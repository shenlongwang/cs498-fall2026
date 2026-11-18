#!/usr/bin/env python3
"""Download, verify, and extract the official Push-T demonstration archive."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_NAME = "pusht.zip"
ARCHIVE_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ARCHIVE_SHA256 = "63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9"
RELATIVE_DATASET = Path("pusht/pusht_cchi_v7_replay.zarr")
EXPECTED_TRANSITIONS = 25650
EXPECTED_EPISODES = 206


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest used to verify an archive."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    """Reject a downloaded Push-T archive whose digest is not approved."""
    actual = sha256(path)
    if actual != ARCHIVE_SHA256:
        raise RuntimeError(
            f"archive checksum mismatch for {path}: {actual}; expected {ARCHIVE_SHA256}"
        )


def verify_dataset(path: Path) -> None:
    """Check the extracted Zarr layout, array shapes, and episode metadata."""
    required = [
        path / ".zgroup",
        path / "data/action/.zarray",
        path / "data/keypoint/.zarray",
        path / "data/state/.zarray",
        path / "meta/episode_ends/.zarray",
    ]
    missing = [item for item in required if not item.is_file()]
    if missing:
        raise RuntimeError(f"incomplete Push-T dataset; missing {missing[0]}")

    handout_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(handout_root))
    try:
        from hw5_utils import load_lowdim_arrays

        obs, action, episode_ends = load_lowdim_arrays(path, obs_dim=20)
    finally:
        sys.path.pop(0)
    if obs.shape != (EXPECTED_TRANSITIONS, 20):
        raise RuntimeError(f"unexpected observation shape: {obs.shape}")
    if action.shape != (EXPECTED_TRANSITIONS, 2):
        raise RuntimeError(f"unexpected action shape: {action.shape}")
    if len(episode_ends) != EXPECTED_EPISODES or int(episode_ends[-1]) != EXPECTED_TRANSITIONS:
        raise RuntimeError(
            f"unexpected episode metadata: {len(episode_ends)} episodes, "
            f"last end {int(episode_ends[-1])}"
        )
    print(
        f"verified {path}: {EXPECTED_TRANSITIONS} transitions, "
        f"{EXPECTED_EPISODES} episodes, obs {obs.shape}, action {action.shape}"
    )


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    """Extract an archive only after ensuring every member stays in the target."""
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if target != output_root and output_root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def download_and_extract(output_dir: Path, url: str = ARCHIVE_URL) -> Path:
    """Install or verify Push-T data and return the extracted dataset path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    dataset_path = output_dir / RELATIVE_DATASET
    if dataset_path.exists():
        if archive_path.exists():
            verify_archive(archive_path)
        verify_dataset(dataset_path)
        return dataset_path

    if not archive_path.exists():
        partial = output_dir / f"{ARCHIVE_NAME}.part"
        print(f"downloading {url} -> {partial}")
        with urllib.request.urlopen(url) as response, partial.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        verify_archive(partial)
        partial.replace(archive_path)
    else:
        verify_archive(archive_path)

    print(f"extracting {archive_path} -> {output_dir}")
    safe_extract(archive_path, output_dir)
    verify_dataset(dataset_path)
    return dataset_path


def default_data_dir() -> Path:
    """Choose the handout data directory shared by student and reference code."""
    here = Path(__file__).resolve().parent
    hw5_root = here.parent.parent
    handout_data = hw5_root / "handout/data"
    if here.parent.name == "solution" and handout_data.is_dir():
        return handout_data
    return here


def main() -> None:
    """Run the command-line download, extraction, or verification workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--url", default=ARCHIVE_URL)
    args = parser.parse_args()

    output_dir = (args.output_dir or default_data_dir()).resolve()
    dataset_path = output_dir / RELATIVE_DATASET
    if args.verify_only:
        archive_path = output_dir / ARCHIVE_NAME
        if archive_path.exists():
            verify_archive(archive_path)
        verify_dataset(dataset_path)
        return
    download_and_extract(output_dir, args.url)


if __name__ == "__main__":
    main()
