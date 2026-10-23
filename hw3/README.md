# HW3 student handout

Release: Friday, October 23, 2026

Due: Friday, November 6, 2026 at 11:59 p.m. America/Chicago

Build a compact CenterPoint-style 3D vehicle detector from LiDAR. The four
task files expose the complete learning pipeline without a separate runner.
For a quick smoke check with the included debug data, run:

```bash
python -m pip install -r requirements.txt
python task1_bev.py
python task2_targets.py
python task3_detector.py --debug
python task4_inference.py --debug
```

Each file has its own `main()` and named checkpoint. Edit only within the
blocks marked `Modify your code below` and `DO NOT MODIFY CODE OUTSIDE THE
BLOCK`.

Graded functions:

- `task1_bev.py`: `rasterize_points`
- `task2_targets.py`: `encode_targets`, `decode_targets`
- `task3_detector.py`: `CenterDetector.forward`, `focal_loss`, `regression_l1`
- `task4_inference.py`: `suppress_nonmaxima`, `decode_predictions`

The required total is 15 points: Task 1 (2.0), Task 2 (3.0), Task 3 (6.0),
and Task 4 (2.5), plus the report and reproducibility (1.5).

The repository contains real KITTI debug archives for these checks.
`--debug` runs only two epochs and verifies code flow; it is not a final
experiment. The full release supplies `train.npz`,
`val.npz`, and `test.npz` using the schema in `data/README.md`. The three splits
contain 5,977/730/774 frames from 123/8/10 non-overlapping raw drives. The
drive-level split balances object range, occlusion, empty frames, 2D box size,
and capture date; the tracking drive is test-only.
Before the final run, download and verify the full splits from the public
[CS 498 dataset repository](https://huggingface.co/datasets/hongchi/cs498):

```bash
python data/download_data.py
python task3_detector.py
python task4_inference.py
```

The large split archives are intentionally not stored in Git. See
`data/README.md` for the archive schema and verification details.
The default Task 3 and Task 4 commands require full data. Task 4 also requires
a matching trained checkpoint; final test requires a validation-selected
threshold. Missing prerequisites cause an error instead of a debug or
untrained-model fallback. The saved threshold records the validation split and
checkpoint hash; `--test` checks both before opening the test data and refuses
a repeat run.
Full training is GPU-oriented. On the staff RTX 4080, batch size 16 used about
11.4 GiB of GPU memory; the default batch size 8 used about 6.7 GiB. Use
`python task3_detector.py --batch-size 4` if your GPU has less memory, and
allow more time. The explicit `--debug` smoke run also works on CPU.
The full detector run uses at most 20 epochs, stops after validation AP@0.5
fails to improve for eight epochs, and keeps the highest-AP checkpoint.
Use training data to fit the detector, validation data for every checkpoint and
threshold decision, and test data exactly once after all choices are frozen.
Then run `python task4_inference.py --test` for the final result.
The course AP uses the supplied split and all in-range `Car` labels; it is not
directly comparable to the official KITTI leaderboard metric.

Expected outputs:

- Task 1: `bev_channels.png`, `bev_boxes.png`, `metrics.json`
- Task 2: `target_roundtrip.png`, `metrics.json`
- Task 3: `detector.pt`, `training_curves.png`, `metrics.json`
- Task 4: `prediction_0.png`, `prediction_1.png`, `pr_curve.png`,
  `selected_threshold.json`, `metrics.json`, and final-only `test_metrics.json`

Submit the revised four task scripts and one PDF report. Do not submit data,
checkpoints, or the full output folder.

## Optional tracking bonus [+2.0 points]

Use `data/tracking_sequence.npz` and your frozen detector to implement
`bonus_tracking.py` yourself; no tracker or metric helper is supplied. Maintain
constant-velocity Kalman tracks, associate detections with Hungarian matching,
and manage track births and deaths. Run all 154 frames in order and assign each
predicted ID a stable, distinct color throughout the video.

Run your completed pipeline with `python bonus_tracking.py`. Video encoding
requires the `ffmpeg` command-line program in addition to the Python packages
in `requirements.txt`.

Report MOTA, mean matched IoU (MOTP), IDF1, and ID switches at rotated BEV IoU
0.5. Submit `bonus_tracking.py`, `tracking_comparison.mp4`, and the corresponding
bonus report section. Do not submit the tracking dataset.

## Optional multiclass bonus [+1.0 point]

The required data contains only Car labels. To add Pedestrian and Cyclist,
first run `python data/download_multiclass_data.py` after the core download.
The small public companion provides class IDs and LiDAR boxes aligned to the
same train/validation/test BEV rows; see `data/README.md` for the schema and
class counts. It does not change the required detector or core data.

Implement `bonus_multiclass.py` yourself, including three-class targets,
heatmaps, class-wise decoding and NMS, validation per-class PR/AP, and at
least two qualitative validation views. Keep all checkpoint and threshold
choices on validation; if reporting optional final test metrics, evaluate test
once after those choices are frozen. Submit the code and a bonus report
section. No multiclass helper implementation is supplied.
