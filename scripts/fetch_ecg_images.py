"""Download the PTB-XL part (D0) of ECG-Image-Database: real printouts of PTB-XL records that were
scanned, photographed with phones, photographed off monitors, stained, creased and so on.

Kaggle `physionet/ecg-image-database` (v2, CC BY-ND 4.0, Reyna et al. 2024, arXiv 2409.16612).
D0_<ecg_id> is the PTB-XL ecg_id. Per-file downloads work anonymously, so only the records
of the requested PTB-XL folds are fetched (all 977 D0 records are about 22 GB; fold 10 is 2.5 GB).
Files that already exist with the right size are skipped, so re-running resumes.

    python scripts/fetch_ecg_images.py --folds 9 10            # dev (fold 9) + test (fold 10)
    python scripts/fetch_ecg_images.py --folds 9 --debug       # 3 records
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

API = "https://www.kaggle.com/api/v1/datasets"
REF = "physionet/ecg-image-database"


def list_files(cache: Path) -> list[tuple[str, int]]:
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    out, tok = [], None
    while True:
        q = {"pageSize": 1000, **({"pageToken": tok} if tok else {})}
        with urllib.request.urlopen(f"{API}/list/{REF}?{urllib.parse.urlencode(q)}", timeout=60) as r:
            d = json.load(r)
        out += [(f["name"], int(f["totalBytes"])) for f in d.get("datasetFiles", [])]
        tok = d.get("nextPageTokenNullable")
        if not tok:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def fetch(name: str, dest: Path, size: int, retries: int = 6) -> int:
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{API}/download/{REF}/{urllib.parse.quote(name, safe='')}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size mismatch for {name}")
            tmp.replace(dest)
            return size
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True, help="PTB-XL strat_fold(s) to fetch")
    ap.add_argument("--dest", default="data/raw/ecg_image_db")
    ap.add_argument("--ptbxl", default="data/raw/ptbxl/ptbxl_database.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="first 3 records only")
    args = ap.parse_args()

    dest = Path(args.dest)
    files = [(n, s) for n, s in list_files(dest / "files.json") if n.startswith("D0/")]
    fold = pd.read_csv(args.ptbxl, index_col="ecg_id")["strat_fold"]
    ecg_id = lambda n: int(n.split("/")[1][3:])
    ids = sorted({ecg_id(n) for n, _ in files if fold.get(ecg_id(n)) in args.folds})
    if args.debug:
        ids = ids[:3]
    keep = set(ids)
    objs = [(n, s) for n, s in files if ecg_id(n) in keep]
    total = sum(s for _, s in objs)
    print(f"D0 folds {args.folds}: {len(ids)} records, {len(objs)} files, {total / 1e9:.2f} GB -> {dest}", flush=True)
    done = got = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, n, dest / n, s) for n, s in objs]
        for f in as_completed(futs):
            got += f.result()
            done += 1
            if done % 200 == 0 or done == len(futs):
                rate = got / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {done}/{len(futs)} files, {got / 1e9:.2f} GB new, {rate:.1f} MB/s", flush=True)
    bad = [n for n, s in objs if not (dest / n).exists() or (dest / n).stat().st_size != s]
    if bad:
        sys.exit(f"{len(bad)} files missing or incomplete, e.g. {bad[:3]}; re-run to resume")
    print("complete")


if __name__ == "__main__":
    main()
