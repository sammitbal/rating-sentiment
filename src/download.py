"""Download the All_Beauty reviews and product metadata (Amazon Reviews '23) and verify them.

Files already present with the expected checksum are not downloaded again.

Usage:  python src/download.py [--no-verify]
"""
import argparse
import hashlib
import os
import stat
import ssl
import urllib.request

from tqdm import tqdm

from config import DATA_URL, META_RAW, RAW, REVIEWS_RAW

# SHA-256 of the files the project was built with. The shipped RoBERTa scores in
# data/precomputed/ are keyed to these exact files.
FILES = {
    REVIEWS_RAW: (
        "review_categories/All_Beauty.jsonl.gz",
        "ee00e66835567c3f12fde6a482f8d7055c22cac2d0924f677263affdd8a0e349",
    ),
    META_RAW: (
        "meta_categories/meta_All_Beauty.jsonl.gz",
        "51f8255c2794afd60e274c10d3e2d09dc1f671eba4ba35f74f748d8631216d05",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, no_verify=False):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rating-sentiment research download"},
    )

    context = None
    if no_verify:
        context = ssl._create_unverified_context()

    with urllib.request.urlopen(
        request,
        timeout=60,
        context=context,
    ) as response, open(dest, "wb") as out:

        total = int(response.headers.get("Content-Length") or 0) or None

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name.removesuffix(".part"),
        ) as bar:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                bar.update(len(block))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="disable SSL verification and accept files whose checksum differs",
    )
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    for path, (remote, expected) in FILES.items():

        if path.exists() and (args.no_verify or sha256(path) == expected):
            print(f"{path.name}: already downloaded")
            continue

        part = path.with_name(path.name + ".part")

        try:
            fetch(
                f"{DATA_URL}/{remote}",
                part,
                args.no_verify,
            )

        except OSError as e:
            raise SystemExit(
                f"Could not download {DATA_URL}/{remote}: {e}\n"
                "Check your internet connection (or proxy settings) and run again."
            )

        actual = sha256(part)

        if actual != expected and not args.no_verify:
            part.unlink()

            raise SystemExit(
                f"{path.name}: checksum mismatch (got {actual}).\n"
                "The file on the server differs from the one this project was built with. "
                "To continue anyway, run with --no-verify and recompute the RoBERTa scores "
                "(python run_pipeline.py --no-verify --recompute-roberta)."
            )

        if path.exists():
            os.chmod(path, stat.S_IWRITE)
            path.unlink()

        part.replace(path)

        if actual == expected:
            print(f"{path.name}: downloaded and verified")
        else:
            print(f"{path.name}: downloaded (checksum not verified)")


if __name__ == "__main__":
    main()