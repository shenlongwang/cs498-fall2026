# HW2 compact PartNeXt data

This folder contains a fixed chair-part subset derived from
[PartNeXt](https://huggingface.co/datasets/AuWang/PartNeXt), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The corresponding
paper and project are [PartNeXt: A Next-Generation Dataset for Fine-Grained
and Hierarchical 3D Part Understanding](https://arxiv.org/abs/2510.20155) and
the [PartNeXt project page](https://authoritywang.github.io/partnext/).

| File | Objects | Labels |
| --- | ---: | --- |
| `train.npz` | 2,400 | included |
| `val.npz` | 400 | included |
| `student_test.npz` | 600 | included; final evaluation only |

Each object has 2,048 deterministically sampled surface points. The NPZ keys
are:

- `xyz`: `float16 [M, 2048, 3]`, centered and uniformly scaled to the range
  `[-1, 1]` along the largest full-mesh extent;
- `rgb`: `uint8 [M, 2048, 3]` in `[0, 255]`;
- `normal`: `float16 [M, 2048, 3]` geometric unit normals; and
- `labels`: `uint8 [M, 2048]`, present in all three files.

Use the training split to fit models and validation to make every model,
hyperparameter, epoch, and checkpoint decision. Use the test split only after
all choices are frozen. The intended use is one final test run for PointNet and
one for the Transformer---two test runs total. Do not inspect test labels,
select test examples, tune from test results, or retrain after seeing them.

Cast `xyz` and `normal` to `float32` when loading, and divide `rgb` by 255.
All released point clouds are z-up. Source meshes with different coordinate
conventions were rotated deterministically before normalization. Normal signs
follow the source GLB face winding.

RGB is the source GLB albedo sampled at exactly the same barycentric surface
locations as XYZ. No scene lighting is applied, so some point clouds naturally
look dark, nearly gray, or uniform even though their colors are aligned.

The coarse labels are:

| ID | Part |
| ---: | --- |
| 0 | seat |
| 1 | backrest |
| 2 | armrest |
| 3 | base/support |
| 4 | other/accessory |

Leaves under the PartNeXt `Seat`, `Backrest`, `Armrest`, and `Base` hierarchy
branches map to their corresponding coarse classes. Every other direct chair
branch, including headrests and footrests, maps to `other/accessory`.

The adjacent JSON manifests map each NPZ row to its PartNeXt model ID and
source annotation record. The annotation revision is
`2723a6430a612203f514e560bfd6e86d47399b12`; the mesh revision is
`62f25b9b9313713f5ff854701b0f0446a096c51f`.

The three student NPZ archives occupy 71,882,645 bytes in total. Float16
quantization has maximum observed coordinate error `2.4414e-4`. Normals add
41,779,200 bytes before compression across the 3,400 unique labeled objects;
in the student archives they add 23,502,388 compressed bytes.
