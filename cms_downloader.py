"""
CMS Hospital Data Downloader
=============================
Downloads all datasets tagged with the "Hospitals" theme from the CMS Provider
Data Catalog metastore API.  Column headers are normalized to snake_case, CSV
files are downloaded in parallel, and only files modified since the last run
are re-downloaded (incremental / idempotent daily job).

Usage
-----
    python cms_downloader.py [--output-dir OUTPUT_DIR] [--workers N] [--force] [--limit N]

    --output-dir  : Directory where CSV files are saved  (default: ./output)
    --workers     : Number of parallel download threads   (default: 8)
    --force       : Ignore last-run timestamps, re-download everything
    --limit       : Download only the first N datasets (useful for testing)
"""

import argparse
import csv
import io
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


METASTORE_URL = "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items"
TARGET_THEME  = "Hospitals"
METADATA_FILE = Path("run_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def to_snake_case(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download_dataset(dataset, output_dir, metadata, session, force):
    ds_id    = dataset.get("identifier", "unknown")
    title    = dataset.get("title", ds_id)
    modified = dataset.get("modified") or dataset.get("issued")

    if not force and metadata.get(ds_id, {}).get("modified") == modified:
        logger.info("[%s]  Skipping - not modified since last run.", ds_id)
        return {"id": ds_id, "title": title, "status": "skipped"}

    # Find a CSV download URL, preferring explicit mediaType over .csv extension
    csv_url = None
    for dist in (dataset.get("distribution") or []):
        media = (dist.get("mediaType") or dist.get("format") or "").lower()
        url   = dist.get("downloadURL") or dist.get("accessURL") or ""
        if "csv" in media and url:
            csv_url = url
            break
        if url.lower().endswith(".csv") and not csv_url:
            csv_url = url

    if not csv_url:
        logger.warning("[%s]  No CSV URL found - skipping.", ds_id)
        return {"id": ds_id, "title": title, "status": "no_url"}

    logger.info("[%s]  Starting download: %s", ds_id, title)
    try:
        raw = session.get(csv_url, timeout=120).content
        logger.info("[%s]  Downloaded %.1f KB - normalizing headers ...", ds_id, len(raw) / 1024)
    except requests.RequestException as exc:
        logger.error("[%s]  Download failed: %s", ds_id, exc)
        return {"id": ds_id, "title": title, "status": "error", "error": str(exc)}

    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig", errors="replace"))))
        if rows:
            rows[0] = [to_snake_case(h) for h in rows[0]]
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        raw = out.getvalue().encode("utf-8")
    except Exception as exc:
        logger.error("[%s]  Header normalization failed: %s", ds_id, exc)

    safe_title = re.sub(r"[^\w-]", "_", title)[:80]
    filepath   = output_dir / (safe_title + "__" + ds_id + ".csv")
    filepath.write_bytes(raw)

    logger.info("[%s]  Saved -> %s", ds_id, filepath.name)
    return {"id": ds_id, "title": title, "status": "downloaded",
            "modified": modified, "file": str(filepath), "bytes": len(raw)}


def run(output_dir, workers, force, limit=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = datetime.now(timezone.utc).isoformat()
    metadata  = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}

    if metadata:
        prior = len([k for k in metadata if not k.startswith("__")])
        logger.info("Loaded metadata for %d previously downloaded datasets.", prior)

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )))
    session.headers["User-Agent"] = "cms-hospital-downloader/1.0"

    # Paginate through the metastore, stopping when a page is empty or
    # contains no identifiers we haven't already seen. The seen-ID check
    # guards against an infinite loop if the API ignores the page= parameter
    # and returns the same response on every call.
    logger.info("Fetching dataset catalog from CMS metastore ...")
    datasets, seen_ids, page = [], set(), 1
    try:
        while True:
            logger.info("  Fetching catalog page %d ...", page)
            batch = session.get(METASTORE_URL,
                                params={"page-size": 100, "page": page},
                                timeout=60).json()
            if not batch:
                break
            new = [ds for ds in batch if ds.get("identifier") not in seen_ids]
            if not new:
                logger.warning("  Page %d returned no new datasets - stopping.", page)
                break
            seen_ids.update(ds.get("identifier") for ds in new)
            datasets.extend(new)
            logger.info("  Page %d: %d new datasets (%d total).", page, len(new), len(datasets))
            if len(batch) < 100:
                break
            page += 1
    except requests.RequestException as exc:
        logger.error("Failed to fetch metastore: %s", exc)
        sys.exit(1)

    hospital_datasets = [
        ds for ds in datasets
        if any(TARGET_THEME.lower() in t.lower() for t in (ds.get("theme") or []))
    ]

    if limit:
        hospital_datasets = hospital_datasets[:limit]

    total = len(hospital_datasets)
    logger.info("Found %d Hospital-themed datasets%s. Starting downloads with %d workers ...",
                total, f" (limited to {limit})" if limit else "", workers)

    results, completed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_dataset, ds, output_dir, metadata, session, force)
                   for ds in hospital_datasets]
        for future in as_completed(futures):
            try:
                result     = future.result()
                completed += 1
                results.append(result)
                if result["status"] == "downloaded":
                    logger.info("Progress [%d/%d]  DOWNLOADED   %.1f KB  %s",
                                completed, total, result.get("bytes", 0) / 1024, result["title"])
                else:
                    logger.info("Progress [%d/%d]  %-10s  %s",
                                completed, total, result["status"].upper(), result["title"])
            except Exception as exc:
                completed += 1
                logger.error("Progress [%d/%d]  THREAD ERROR  %s", completed, total, exc)

    for result in results:
        if result["status"] == "downloaded":
            metadata[result["id"]] = {"title": result["title"], "modified": result.get("modified"),
                                       "file": result.get("file"), "last_downloaded": run_start}
    metadata["__last_run__"] = run_start
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("downloaded", "skipped", "error", "no_url")}
    logger.info("Done - downloaded: %d  skipped: %d  errors: %d  no_url: %d",
                counts["downloaded"], counts["skipped"], counts["error"], counts["no_url"])


def main():
    p = argparse.ArgumentParser(description="Download CMS Hospital datasets with snake_case headers.")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--workers",    type=int,  default=8)
    p.add_argument("--force",      action="store_true")
    p.add_argument("--limit",      type=int,  default=None,
                   help="Download only the first N datasets (useful for testing)")
    args = p.parse_args()
    run(output_dir=args.output_dir, workers=args.workers, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()