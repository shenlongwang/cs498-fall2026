# HW3 data

## Download the full splits

From this directory, run:

```bash
python download_data.py
```

The script downloads [`hw3_data.zip` from the course dataset repository](https://huggingface.co/datasets/hongchi/cs498), extracts `train.npz`, `val.npz`, and
`test.npz` here, and checks each file against `SHA256SUMS`. An interrupted
download can be resumed by running the same command again. Run
`python download_data.py --verify-only` to check files already present. Keep
the final test split unused until model and threshold choices are frozen.
If you downloaded the September preview package, the script recognizes those
three checksums and atomically upgrades them to the corrected grid-aligned
release. Retrain Task 3 after upgrading; a detector checkpoint trained on the
preview targets is not compatible with the corrected package.

The required assignment uses all 7,481 labeled training frames from the
[KITTI 3D object-detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
Camera-frame `Car` labels are converted to the LiDAR frame: x forward, y left,
and z up. Every box is
`[x,y,z,length,width,height,yaw]`.

KITTI's object labels cover objects visible in the front color camera, while
the Velodyne sweep contains 360-degree returns. The translucent blue wedge in
the supplied BEV visualizations marks the approximate camera-label field of
view; its exact boundary varies slightly with image size and calibration.

Small real-data archives included for immediate checkpoints:

- `debug_frames.npz`: eight variable-length LiDAR sweeps (`points`), their
  `Car` boxes (`boxes`), and source frame indices (`frame_ids`). Tasks 1 and
  2 use this archive.
- `debug_train.npz` and `debug_val.npz`: quantized processed tensors for a
  two-epoch end-to-end smoke run of Tasks 3 and 4.

Final split archives use the same processed schema:

The release uses preprocessing version `grid-aligned-v2` (recorded in
`split_manifest.json`). Its continuous output-grid row is
`v = 100 - (y + 40) / 0.8`; use the supplied coordinate helpers for exact
half-open boundaries. In particular, `floor(v)` agrees with the corresponding
stride-4 BEV row.

| Key | Shape | Stored dtype | Meaning |
|---|---|---|---|
| `bev` | `[F,4,400,352]` | `uint8` | four BEV channels divided by 255 in the loader |
| `heatmap` | `[F,1,100,88]` | `uint8` | Gaussian center heatmap divided by 255 |
| `inds` | `[F,64]` | `int32` | flat center indices |
| `mask` | `[F,64]` | `uint8` | valid object slots |
| `reg` | `[F,64,2]` | `float16` | sub-cell offsets |
| `height` | `[F,64,1]` | `float16` | metric center height |
| `dims` | `[F,64,3]` | `float16` | log box dimensions |
| `rot` | `[F,64,2]` | `float16` | sine/cosine yaw |
| `boxes` | `[F,64,7]` | `float16` | padded metric boxes for evaluation |
| `box_mask` | `[F,64]` | `uint8` | valid evaluation boxes |

Debug data is for correctness checks only. The final 5,977-frame training,
730-frame validation, and 774-frame test archives are grouped with KITTI's
official object-to-raw mapping. Their 123/8/10 source drives do not overlap.
The deterministic drive-level split balances frame and box counts together
with empty frames, object range, occlusion, 2D box size, and capture date. The
tracking drive appears only in the test split.
Use validation for every model and threshold decision, then evaluate the test
split once after all choices are frozen.

The three full split archives are distributed together as `hw3_data.zip`
because they are too large for ordinary Git. The download script above
extracts them into this directory; no code or path changes are required.
The optional tracking sequence is already included in the handout. Expected
hashes are listed in `SHA256SUMS`.

## Optional multiclass labels

The +1.0 multiclass extension needs Pedestrian and Cyclist annotations, which
are not in the required Car-only NPZs. After installing the core splits, run:

```bash
python download_multiclass_data.py
python download_multiclass_data.py --verify-only
```

This downloads the public, 478 KB `hw3_multiclass_data.zip` companion and
checks its SHA-256 and each extracted file against `SHA256SUMS`. It does not
replace any core file. The three `multiclass_train.npz`,
`multiclass_val.npz`, and `multiclass_test.npz` archives align row-for-row with
the core BEV splits via `frame_ids`. Their Car boxes match the required data
exactly. The label schema is:

| Key | Shape | Stored dtype | Meaning |
|---|---|---|---|
| `frame_ids` | `[F]` | `int32` | original KITTI labeled-frame index |
| `boxes` | `[F,64,7]` | `float16` | LiDAR boxes in the core coordinate convention |
| `classes` | `[F,64]` | `uint8` | 0 Car, 1 Pedestrian, 2 Cyclist |
| `box_mask` | `[F,64]` | `uint8` | valid box/class slots |

`multiclass_manifest.json` records the frozen core split checksums and class
counts. Train/validation/test contain 22,401/2,958/2,974 Cars,
3,077/757/647 Pedestrians, and 1,254/297/62 Cyclists. The held-out test
Cyclist count is small, so report validation per-class AP and PR for the bonus;
if you also report test metrics, evaluate them once after freezing every
multiclass checkpoint and threshold. No multiclass helper implementation is
supplied.

## Optional tracking sequence

`tracking_sequence.npz` contains all 154 consecutive frames, numbered 0--153,
from KITTI raw drive `2011_09_26_drive_0005_sync`, which was used in the
original CS588 detector assignment. It stores 527 labeled boxes from 12
vehicle tracklets. KITTI `Car`, `Van`, `Truck`, `Tram`, and `Bus` tracklets are
merged into the assignment's single vehicle class.

| Key | Shape | Stored dtype | Meaning |
|---|---|---|---|
| `bev` | `[154,4,400,352]` | `uint8` | sequential detector inputs divided by 255 |
| `boxes` | `[154,64,7]` | `float16` | padded metric ground-truth boxes |
| `track_ids` | `[154,64]` | `int16` | persistent ground-truth identities |
| `box_mask` | `[154,64]` | `uint8` | valid box and identity slots |
| `frame_ids` | `[154]` | `int16` | temporal frame indices 0--153 |

SHA-256: `ba7ab903d6abb79c2c31e7a6b4e23c8f4661d24c15952a1490c3b30142ad8dcb`.
Ground-truth boxes and IDs are supplied only for final metric computation and
comparison visualization. Do not use them for detector filtering,
association, tracker tuning, or track birth/death decisions.
