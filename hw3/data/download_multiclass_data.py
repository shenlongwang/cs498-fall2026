"""Download the optional Car/Pedestrian/Cyclist label companion for HW3.

Run the core `download_data.py` first. This small archive adds labels only;
it does not replace the frozen core BEV tensors, checkpoint, or test protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from download_data import DATA_ROOT, download_resumable, sha256


URL = "https://huggingface.co/datasets/hongchi/cs498/resolve/main/hw3_multiclass_data.zip?download=true"
NAMES = ("multiclass_train.npz", "multiclass_val.npz", "multiclass_test.npz",
         "multiclass_manifest.json")
CORE_NAMES = ("train.npz", "val.npz", "test.npz", "split_manifest.json")


def hashes():
    entries = {}
    for line in (DATA_ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    return {name: entries[name] for name in (*NAMES, "hw3_multiclass_data.zip", *CORE_NAMES)}


def validate(output, expected):
    for name in CORE_NAMES:
        if not (output / name).is_file() or sha256(output / name) != expected[name]:
            raise ValueError(f"Missing or invalid core {name}; run download_data.py first")
    for name in NAMES:
        if not (output / name).is_file() or sha256(output / name) != expected[name]:
            raise ValueError(f"Missing or invalid optional {name}")
    manifest = json.loads((output / "multiclass_manifest.json").read_text())
    if (manifest["classes"] != ["Car", "Pedestrian", "Cyclist"]
            or manifest["core_manifest_sha256"] != expected["split_manifest.json"]):
        raise ValueError("Optional labels do not align with the frozen core splits")
    for split in ("train", "val", "test"):
        if manifest["splits"][split]["core_npz_sha256"] != expected[f"{split}.npz"]:
            raise ValueError(f"Optional {split} labels do not match the core BEV split")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = hashes()
    if args.verify_only:
        validate(output, expected)
        print("Optional multiclass labels verified")
        return
    for name in CORE_NAMES:
        if not (output / name).is_file() or sha256(output / name) != expected[name]:
            raise ValueError(f"Run download_data.py to install the verified core {name} first")
    if all((output / name).is_file() and sha256(output / name) == expected[name]
           for name in NAMES):
        validate(output, expected)
        print("Optional multiclass labels are already verified")
        return
    archive = output / "hw3_multiclass_data.zip"
    if archive.exists() and sha256(archive) != expected[archive.name]:
        raise ValueError(f"Existing {archive} has an unknown checksum; move it aside")
    if not archive.exists():
        print(f"Downloading {URL}", flush=True)
        download_resumable(URL, archive)
    if sha256(archive) != expected[archive.name]:
        raise ValueError("Downloaded optional ZIP failed SHA-256")
    with zipfile.ZipFile(archive) as package:
        if package.namelist() != list(NAMES):
            raise ValueError("Unexpected optional ZIP contents")
        for name in NAMES:
            if package.getinfo(name).file_size > 10_000_000:
                raise ValueError(f"Unreasonable optional label file size: {name}")
            target = output / name
            if target.exists():
                if sha256(target) == expected[name]:
                    continue
                raise ValueError(f"Existing {target} has an unknown checksum; move it aside")
            partial = output / f"{name}.partial"
            digest = hashlib.sha256()
            with package.open(name) as source, partial.open("wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Optional split checksum mismatch: {name}")
            partial.replace(target)
    validate(output, expected)
    archive.unlink()
    print("Optional multiclass labels installed and verified")


if __name__ == "__main__":
    main()
