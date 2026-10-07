# HW2 student handout

Release: Wednesday, October 7, 2026

Due: Wednesday, October 21, 2026 at 11:59 p.m. America/Chicago

In this homework, you will label every point of a 3D chair as a semantic part.
You will first prepare and evaluate compact PartNeXt-derived point clouds,
then build a PointNet baseline, and finally train a small Transformer on
the same split.

Use Python 3.12. Create and activate a virtual environment, then install the
dependencies from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -c "import numpy, matplotlib, torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

For CUDA, install the PyTorch build appropriate to your driver using the
[official installer](https://pytorch.org/get-started/locally/) before installing
the requirements. Colab normally supplies these packages; run the import check
there as well. No external attention package is required.

Work only between `INSERT YOUR CODE BELOW` and
`DO NOT MODIFY CODE OUTSIDE THIS BLOCK`. Each graded function has its own edit
block; do not modify code elsewhere. Run each task file directly from this folder:

```bash
python task1_data.py
python task2_pointnet.py
python task3_transformer.py
```

If you want to run Task 2 or Task 3 in Google Colab, first generate the
Colab-friendly notebooks from this folder:

```bash
python make_colab_notebooks.py
```

This creates `task2_pointnet_colab.ipynb` and
`task3_transformer_colab.ipynb`. Upload the handout folder, or a zip of the
handout folder, to Colab so the notebook working directory contains `data/`,
`hw2_utils.py`, `task1_data.py`, `task2_pointnet.py`, and
`task3_transformer.py`. The notebooks use the Python files as the source of
truth, so edit the `.py` files first and regenerate the notebooks if your code
changes. In Colab, choose a GPU runtime when possible, run Task 2 before Task 3
to provide the required PointNet checkpoint, and keep the final test cells commented
out until all validation-based choices are frozen.

Each file contains its complete pipeline in `main()` and prints a checkpoint
after each subtask. Implement one block, rerun that same file, and inspect the
saved figure or metric before continuing. Complete and test Task 1 before
training either model. If its metric functions are incomplete, Tasks 2 and 3
stop before training with a labeled diagnostic in `metrics.json`. Run Task 2
before Task 3: a saved PointNet checkpoint is required for the comparison.

The `tests/` folder contains small local sanity tests for the student-code
blocks. They use synthetic inputs and do not train models, so they should run
quickly on CPU. After editing a task function, run all tests from this folder:

```bash
python -m unittest discover -s tests
```

You can also run one task's tests at a time:

```bash
python -m unittest tests.test_task1_data
python -m unittest tests.test_task2_pointnet
python -m unittest tests.test_task3_transformer
```

The starter code is expected to fail several tests because it contains
placeholder implementations. Use the failure messages to find shape, weighting,
gradient-flow, attention, confusion-matrix, or IoU mistakes. Passing these tests
is not a substitute for the required figures, metrics, or report, but it can
catch common implementation bugs early.

Student-facing files:

- `task1_data.py`: implement point-cloud augmentation, a confusion matrix, and
  per-class IoU.
- `task2_pointnet.py`: implement `PointNetSegmenter.forward`.
- `task3_transformer.py`: implement scaled dot-product self-attention, the
  Transformer segmentation forward pass, and weighted segmentation loss.
- `hw2_utils.py`: low-level loading, normalization, and plotting; do not edit.
- `report_template.tex`: report structure and AI-use declaration.
- `data/`: compact fixed splits. XYZ and optional bonus normals use `float16`,
  RGB and semantic labels use `uint8`, and model inputs are converted to
  `float32` when loaded. Required models use only XYZ+RGB.

The required models use the same supplied XYZ+RGB points, fixed splits, loss
weighting, and evaluation code. This makes the comparison about architecture
rather than a different data pipeline. Both training pipelines run for 100
epochs and save the checkpoint with the highest validation mean IoU.

Fixed architecture and tensor contract:

- Let `B` be batch size, `N=2048` points, and `C=5` classes. Both models take
  `xyz` and `rgb` with shape `[B,N,3]` and return raw logits `[B,N,C]` without
  changing point order.
- PointNet uses shared `1x1 Conv1d` layers `6→64→128→256`, max-pools over the
  `N` points, repeats the `[B,256,1]` global feature, concatenates local and
  global features to `[B,512,N]`, and classifies with `512→128→5`.
- The Transformer uses width 96, four heads of width 24, four pre-LN blocks,
  MLP width 384, and dropout 0.1. Its packed QKV projection is `96→288`, with
  each Q/K/V reshaped to `[B,4,N,24]`. The final head is `LayerNorm` followed
  by `96→5`.

These architectures are fixed for the required comparison. Implement the
declared forward passes; do not add, remove, or resize layers. The handout PDF
contains a labeled architecture figure and a complete stage-by-stage shape
table for each model.

The weighted loss uses `F.cross_entropy(..., weight=class_weights,
reduction="mean")`: divide the weighted sum of negative log probabilities by
the sum of the target-class weights across all batch/point entries.

Use training data to fit the models and validation data for every design,
hyperparameter, epoch, and checkpoint decision. After all choices are frozen,
make at most one final test evaluation per required model (two total):

```bash
python task2_pointnet.py --test
python task3_transformer.py --test
```

Test labels are included only to make these final results reproducible. Do not
inspect or tune from them, and do not rerun training after seeing test results.
All qualitative figures and error analysis use validation samples.

Task 3 should be run in a CUDA environment when possible. A staff Apple M4
MacBook Air needed about 13 minutes for the full PointNet run, while each
Transformer epoch took about 5 minutes under MPS. The 100-epoch Transformer
reference run took about 2.5 minutes on the staff CUDA workstation.

Expected task checkpoints:

- `outputs/task1/ground_truth.png`, `class_frequency.png`, `augmentation.png`,
  `confusion_matrix.txt`, `metrics.json`
- `outputs/task2/training_curves.png`, `predictions.png`, `metrics.json`,
  `pointnet.pt`, and final-only `test_metrics.json`. The curve shows
  training/validation loss and validation mIoU; the prediction figure contains
  two fixed validation chairs.
- `outputs/task3/training_curves.png`, `predictions.png`,
  `model_comparison.png`, `metrics.json`, `transformer.pt`, and final-only
  `test_metrics.json`. The comparison uses the same two fixed validation
  chairs for input RGB, ground truth, PointNet, and the Transformer.

The optional Utonia bonus has no supplied helper. A complete attempt must save
`feature_pca.png`, `training_curves.png`, `predictions.png`, and `metrics.json`
under `outputs/bonus/`, following the handout's two-sample visualization and
metric requirements. Obtain the pretrained encoder from the
[official Utonia source](https://github.com/Pointcept/Utonia) and
[official checkpoint](https://huggingface.co/Pointcept/Utonia); document the
exact revision and checkpoint you used. Its sparse-convolution dependencies
may need a separate, compatible environment from the Python 3.12 required
work. A frozen-feature cache and streamed linear-head batches can lower GPU
memory demand without changing the bonus's linear-probe rule.

Submit exactly these required files:

1. your revised `task1_data.py`;
2. your revised `task2_pointnet.py`;
3. your revised `task3_transformer.py`; and
4. one PDF report.

Do not submit supplied data, helper files, or model checkpoints. Your report
figures and values must be reproducible by running the submitted scripts.
Run Task 2 before Task 3 so the shared comparison can load your PointNet
checkpoint.
