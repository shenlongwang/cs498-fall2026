# HW5 data

From the extracted homework root (the directory containing `task3_policy.py`),
download, verify, and extract the official Diffusion Policy Push-T
demonstrations with either equivalent command:

```bash
python data/download_data.py
python task3_policy.py --mode download
```

The expected dataset path is:

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

The downloader verifies `pusht.zip` against `SHA256SUMS`, rejects unsafe archive
paths, and validates the extracted observation/action shapes and episode
metadata. You can rerun only the verification with:

```bash
python data/download_data.py --verify-only
```

Do not add the downloaded ZIP, extracted Zarr store, generated toy data,
checkpoints, rollout videos, or other outputs to Git. Only this README, the
downloader, and `SHA256SUMS` belong in the release.
