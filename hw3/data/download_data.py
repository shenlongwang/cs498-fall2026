"""Download and verify the three full HW3 data splits.

Run from this directory with: python download_data.py
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent
ARCHIVE_URL = "https://huggingface.co/datasets/hongchi/cs498/resolve/main/hw3_data.zip?download=true"
SPLITS = ("train.npz", "val.npz", "test.npz")
MAX_SPLIT_BYTES = 500_000_000
PREVIOUS_SPLIT_SHA256 = {
    "train.npz": "6af3fd01e6fbad1da23e3dab50f9da365f748004b6eefcb8a1b593c28d4ed8df",
    "val.npz": "dec4dfb0a303b4332d4ded4376f1588e71f275016082fa60461f5103726f6ef1",
    "test.npz": "429f15432d2d6284626cb2bb558d41af7289be918065092a09cb751bca6258c3",
}
PREVIOUS_ARCHIVE_SHA256 = "85ee72881a56883c2af8984e6e6448794388b834a74abb6daf1742f9ca8f762c"


def expected_hashes() -> dict[str, str]:
    entries = {}
    for line in (DATA_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return {name: entries[name] for name in (*SPLITS, "hw3_data.zip")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url: str, path: Path) -> None:
    """Keep an interrupted transfer in a .partial file for the next run."""
    partial = path.with_name(path.name + ".partial")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                continued = offset and response.status == 206
                if continued:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected resume response: {content_range}")
                mode = "ab" if continued else "wb"
                expected_bytes = response.headers.get("Content-Length")
                received = 0
                with partial.open(mode) as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                if expected_bytes is not None and received != int(expected_bytes):
                    raise OSError(f"Incomplete transfer: {received}/{expected_bytes} bytes")
            partial.replace(path)
            return
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            if attempt == 4:
                raise RuntimeError(f"Failed to download {url}") from error
            print(f"Download interrupted; retrying ({attempt + 1}/5)...", flush=True)
            time.sleep(min(2 ** attempt, 15))


def install_archive(archive_path: Path, output: Path, hashes: dict[str, str]) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(SPLITS) or set(names) != set(SPLITS):
            raise ValueError(f"Unexpected ZIP contents: {names}")
        for name in SPLITS:
            info = archive.getinfo(name)
            if info.file_size > MAX_SPLIT_BYTES:
                raise ValueError(f"Unreasonable size for {name}: {info.file_size}")
            target = output / name
            if target.exists():
                current = sha256(target)
                if current == hashes[name]:
                    print(f"Already verified: {name}", flush=True)
                    continue
                if current != PREVIOUS_SPLIT_SHA256[name]:
                    raise ValueError(f"Existing {target} has an unknown hash; move it aside and rerun")
                print(f"Upgrading previous course split: {name}", flush=True)
                print("Retrain Task 3; preview checkpoints are incompatible with this data.", flush=True)
            partial = output / f"{name}.partial"
            digest = hashlib.sha256()
            with archive.open(info) as source, partial.open("wb") as sink:
                while chunk := source.read(4 * 1024 * 1024):
                    sink.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != hashes[name]:
                raise ValueError(f"SHA-256 mismatch for {name}; expected {hashes[name]}")
            partial.replace(target)
            print(f"Installed and verified: {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = expected_hashes()
    if args.verify_only:
        for name in SPLITS:
            path = output / name
            if not path.is_file() or sha256(path) != hashes[name]:
                raise ValueError(f"Missing or invalid {path}")
            print(f"Verified: {name}", flush=True)
        return
    if all((output / name).is_file() and sha256(output / name) == hashes[name]
           for name in SPLITS):
        print("All three HW3 splits are already verified.", flush=True)
        return

    archive = output / "hw3_data.zip"
    if archive.is_file():
        cached_hash = sha256(archive)
        if cached_hash != hashes["hw3_data.zip"] and cached_hash != PREVIOUS_ARCHIVE_SHA256:
            raise ValueError(f"Existing {archive} has an unknown hash; move it aside and rerun")
        if cached_hash == PREVIOUS_ARCHIVE_SHA256:
            archive.unlink()
            print("Replacing previous course ZIP with the corrected release.", flush=True)
    if not archive.exists():
        print(f"Downloading {ARCHIVE_URL}", flush=True)
        download_resumable(ARCHIVE_URL, archive)
    if sha256(archive) != hashes["hw3_data.zip"]:
        archive.unlink()
        raise ValueError("Downloaded ZIP failed SHA-256; rerun to retry")
    install_archive(archive, output, hashes)
    archive.unlink()
    print("HW3 data ready. Removed the ZIP after verified extraction.", flush=True)


if __name__ == "__main__":
    main()
