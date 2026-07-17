import argparse
import sys
import urllib.request
import urllib.error

from config import ALL_RECORDS, MITDB_MIRROR, RAW_DIR

EXTENSIONS = ["hea", "dat", "atr"]


def download_record(record, dest_dir, overwrite=False):
    written = []
    for ext in EXTENSIONS:
        target = dest_dir / f"{record}.{ext}"
        if target.exists() and not overwrite:
            continue
        url = f"{MITDB_MIRROR}/{record}.{ext}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to download {url}: {exc}") from exc
        target.write_bytes(data)
        written.append(target.name)
    return written


def main():
    parser = argparse.ArgumentParser(description="Download the MIT-BIH Arrhythmia Database WFDB files.")
    parser.add_argument("--records", nargs="*", default=ALL_RECORDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0
    for record in args.records:
        written = download_record(record, RAW_DIR, overwrite=args.overwrite)
        total_written += len(written)
        status = "downloaded " + ", ".join(written) if written else "already present"
        print(f"{record}: {status}")

    present = sum(1 for r in args.records for e in EXTENSIONS if (RAW_DIR / f"{r}.{e}").exists())
    expected = len(args.records) * len(EXTENSIONS)
    print(f"\n{total_written} files downloaded; {present}/{expected} record files present in {RAW_DIR}")
    if present != expected:
        print("WARNING: some record files are missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
