# HW5 student handout

Release: To be announced

Due: To be announced at 11:59 p.m. America/Chicago

In this homework you will build a compact low-dimensional Diffusion Policy for
the Push-T manipulation task. You will complete **six focused required
implementation blocks** covering episode-aware sampling, normalization
statistics, deterministic DDIM, FiLM conditioning, the training objective, and
policy inference. Supporting schedule, padding, embedding, U-Net wiring,
training, and evaluation code is provided so you can focus on these concepts.

Work only between `INSERT YOUR CODE BELOW` and
`DO NOT MODIFY CODE OUTSIDE THIS BLOCK`. Every graded function has its own edit
block. The provided code outside those blocks handles endpoint padding,
normalization application, diffusion schedules and forward noising, timestep
embeddings, U-Net data flow, model assembly, data loading, plotting, training,
EMA, checkpointing, rollout evaluation, and video export.

Required implementation map:

| File | Student block | Main contract |
|---|---|---|
| `task1_data.py` | `create_sequence_indices` | Map each clipped episode slice into a fixed-length sample without crossing episodes |
| `task1_data.py` | `LinearNormalizer.fit` | Store one midpoint and safe half-range per final-axis feature |
| `task2_diffusion.py` | `DDIMScheduler.step` | Apply one deterministic reverse update using the scheduler's actual next timestep |
| `task2_diffusion.py` | `FiLMResidualBlock.forward` | Modulate temporal features with condition-derived channel scale and bias |
| `task3_policy.py` | `DiffusionPolicy.compute_loss` | Normalize, noise, predict epsilon, and return scalar MSE |
| `task3_policy.py` | `DiffusionPolicy.predict_action` | Run every DDIM step and return the aligned receding-horizon action chunk |

Run the three task files directly from this directory:

```bash
python task1_data.py
python task2_diffusion.py
python task3_policy.py --mode check
```

The starter implementations are shape-correct placeholders, so all three
commands run before the assignment is complete. The tests intentionally fail
until the corresponding student blocks are implemented.

## Setup

```bash
cd /path/to/hw5
conda create -n cs498-hw5 python=3.11
conda activate cs498-hw5
pip install -r requirements.txt
```

Here, `/path/to/hw5` means the extracted homework root containing
`task1_data.py`, `task2_diffusion.py`, and `task3_policy.py`.

Run all dataset-free CPU tests with:

```bash
python -m unittest discover -s tests
```

## Task 1: Demonstration sequences and normalization [3.0 points]

Implement in `task1_data.py`:

- `create_sequence_indices`
- `LinearNormalizer.fit`

For an episode of length `L`, enumerate local starts from `-pad_before`
through `L - horizon + pad_after`, inclusive. For each start, clip the real
buffer interval to `[0, L]`, compute how far the clipped interval is shifted
inside the length-`horizon` sample, and then convert local buffer indices to
global replay-buffer indices. The real buffer length and sample length must
match. The provided `_slice_with_padding` then repeats the first or last real
row wherever the sample extends beyond the episode.

For normalization, flatten all leading dimensions but keep the final feature
axis. For each feature, store
`center = (minimum + maximum) / 2` and
`scale = (maximum - minimum) / 2`. Replace scales below `self.eps` with one.
The provided normalization and inverse methods preserve NumPy/Torch type,
dtype, device, shape, and broadcasting.

Run:

```bash
python task1_data.py
python -m unittest tests.test_task1_data
```

The task script generates a toy replay buffer and writes:

- `outputs/task1/sequence_windows.png`
- `outputs/task1/normalization.png`

The first figure must show fixed-horizon windows padded by repeating episode
endpoint values. The normalized observation and action extrema should be close
to -1 and 1 per nonconstant feature.

## Task 2: DDIM and conditional temporal U-Net [4.0 points]

Implement in `task2_diffusion.py`:

- `DDIMScheduler.step`
- `FiLMResidualBlock.forward`

The beta schedules, forward noising equation, sinusoidal embedding, and full
conditional temporal U-Net data flow are provided. You implement the two
central operations used repeatedly during denoising:

1. In `DDIMScheduler.step`, convert the scalar timestep to an integer, obtain
   the scheduler's actual previous inference timestep with
   `_previous_timestep`, recover and clamp the clean estimate, and combine it
   with the predicted epsilon. If there is no previous timestep, return the
   clean estimate directly. Do not sample new noise.
2. In `FiLMResidualBlock.forward`, run `block1`, split the encoded condition
   into `(scale, bias)`, add a final singleton time dimension to each, apply
   `output * (1 + scale) + bias`, run `block2`, and add the projected residual.

Run:

```bash
python task2_diffusion.py
python -m unittest tests.test_task2_diffusion
```

The task script writes the graded visual checkpoint
`outputs/task2/task2_visual_checkpoint.png`. It shows the diffusion schedule,
one synthetic action coordinate at several noise levels, an oracle-noise DDIM
reverse reconstruction, and the temporal U-Net output. Increasing timesteps
should progressively destroy the clean trajectory structure; the oracle
reverse curve should align with the clean curve with near-zero error. Include
this figure and a brief explanation in the report before training the policy.

## Task 3: Diffusion Policy, training, and rollout [5.0 points]

Implement in `task3_policy.py`:

- `DiffusionPolicy.compute_loss`
- `DiffusionPolicy.predict_action`

The policy constructor already assembles the temporal U-Net, scheduler, and
normalizer. Use this checklist for the two remaining blocks:

- `compute_loss`: normalize observations and actions; flatten exactly the
  first `n_obs_steps` observations; draw one integer timestep per batch item;
  draw action noise with `torch.randn_like`; call `q_sample`; predict noise;
  return scalar mean-squared error against the same sampled noise.
- `predict_action`: switch to evaluation mode; normalize and flatten the same
  observation prefix; initialize a `(B, horizon, action_dim)` Gaussian tensor
  on the observation device/dtype; call `set_timesteps`; apply every timestep
  in descending order; unnormalize once after the loop; return the full
  trajectory and indices `n_obs_steps - 1` through the next
  `n_action_steps` actions.

First run the dataset-free API and unit checks:

```bash
python task3_policy.py --mode check
python -m unittest tests.test_task3_policy
```

Download the official Push-T demonstrations:

```bash
python task3_policy.py --mode download
```

This creates `data/pusht/pusht_cchi_v7_replay.zarr`. The data is used locally
and must not be added to Git.

Run a short end-to-end training check:

```bash
python task3_policy.py --mode smoke
```

Then run the required full training process:

```bash
python task3_policy.py --mode train
```

Start this run early. On staff hardware, the required 30,000 steps took about
33 minutes on CUDA and roughly 2.5--4 hours on CPU; a complete four-checkpoint
CUDA rollout sweep took about 10 minutes. Hardware and system load can change
these times substantially. The optional 15,000-step flow run took about 22
minutes on CUDA. CUDA is recommended but not required.

The command uses CUDA when available and otherwise uses CPU. Exact learned
weights and the best checkpoint can differ across devices even with the fixed
seed, so select from all four required checkpoints using your own fixed-seed
results rather than copying a staff checkpoint number. The staff
reproducibility run uses `--device cpu`; this slower choice is not required for
students.

The full run writes four periodic checkpoints (steps 7,500, 15,000, 22,500,
and 30,000), metrics, and the loss curve under
`outputs/task3/pusht_train/`. Required inference uses 50 deterministic DDIM
steps. On CUDA, the provided evaluator fixes deterministic kernels and cuDNN
TF32 explicitly and records those settings in `eval_metrics.json`; this keeps
the numerical backend auditable across machines. Evaluate the final checkpoint
over the fixed 20-seed validation suite:

```bash
python task3_policy.py --mode rollout \
  --checkpoint outputs/task3/pusht_train/model.pt \
  --num-rollouts 20
```

Evaluate all saved checkpoints and select the best by success rate, then mean
maximum reward:

```bash
python task3_policy.py --mode sweep \
  --checkpoint outputs/task3/pusht_train \
  --num-rollouts 20
```

After selecting the best checkpoint, generate the required report videos for
one successful seed and the lowest-reward seed (prefer an unsuccessful seed
when one exists):

```bash
python task3_policy.py --mode visualize \
  --checkpoint outputs/task3/pusht_train/ckpt_BEST_STEP.pt \
  --seeds SUCCESS_SEED,LOWEST_REWARD_SEED \
  --output-dir outputs/task3/pusht_train/report_rollouts
```

`--mode visualize` always saves one MP4 and one start/middle/end contact sheet
per seed under `report_rollouts/media/`, together with `eval_metrics.json`.
Report the selected checkpoint step, 20-seed success rate, mean maximum reward,
and the behavior visible in both videos. Include both contact sheets and name
the corresponding MP4 files in the report.

For the rollout-quality score, 15 or more successes out of 20 earns full
performance credit and 10/20 earns half performance credit. Between 10 and 15
successes, credit interpolates evenly; below 10, it scales proportionally from
zero to half credit. Required metrics and visual evidence are checked
separately, so report the actual result even when it is below these anchors.

## Optional bonus: conditional flow matching [+1.0 point]

The bonus reuses the provided model, data, normalization, training, EMA,
checkpoint, rollout, plotting, and video infrastructure. Implement only the two
marked blocks in `bonus_flow_matching.py`:

- `FlowMatchingPolicy.compute_loss`
- `FlowMatchingPolicy.predict_action`

The loss interpolates between Gaussian source trajectories and normalized
expert actions and predicts the constant target velocity. Inference starts from
Gaussian noise and integrates the learned velocity field with the provided
Euler time grid. Run the dataset-free checks before optional training:

```bash
python bonus_flow_matching.py --mode check
python -m unittest tests.test_bonus_flow_matching
```

If attempted, complete the supplied smoke and full training workflows, sweep
the saved flow checkpoints through the shared evaluator, and visualize at
least two selected seeds:

```bash
python bonus_flow_matching.py --mode smoke
python bonus_flow_matching.py --mode train
python task3_policy.py --mode sweep \
  --checkpoint outputs/bonus_flow/pusht_train \
  --num-rollouts 20
python bonus_flow_matching.py --mode visualize \
  --checkpoint outputs/bonus_flow/pusht_train/ckpt_BEST_STEP.pt \
  --seeds SUCCESS_SEED,LOWEST_REWARD_SEED \
  --output-dir outputs/bonus_flow/pusht_train/report_rollouts
```

Include the bonus loss curve, fixed-seed metrics, and one generated contact
sheet in the report. Bonus credit depends on a correct loss and sampler plus
complete evidence; it does not require outperforming the required DDIM policy.

## Report and submission

Use `report_template.tex`. Submit one archive named `netid_hw5.zip` containing
the three completed task files, `report.pdf`, and the following lightweight
canonical evidence tree:

```text
outputs/task1/{sequence_windows.png,normalization.png}
outputs/task2/task2_visual_checkpoint.png
outputs/task3/pusht_train/{loss_curve.png,train_metrics.csv}
outputs/task3/pusht_train/{checkpoint_eval_summary.json,checkpoint_eval_summary.csv,best_checkpoint.json}
outputs/task3/pusht_train/report_rollouts/eval_metrics.json
outputs/task3/pusht_train/report_rollouts/media/rollout_seed_*.{mp4,png}
```

The rollout bundle must contain at least the two seeds analyzed in the report.
If you attempt the bonus, also submit `bonus_flow_matching.py` and the analogous
loss, metric, selection, and two-rollout evidence under
`outputs/bonus_flow/pusht_train/`.

Do not submit `hw5_utils.py`, tests, downloaded data, generated toy data,
checkpoints (`*.pt`), virtual environments, `__pycache__/`, smoke outputs, or
unselected evaluation directories. Keep all generated artifacts untracked in
Git even though the listed lightweight files belong in the LMS archive.

Point distribution:

```text
Task 1: demonstration sequences and normalization       3.0
Task 2: DDIM and conditional temporal U-Net              4.0
Task 3: Diffusion Policy, training, and rollout           5.0
Report quality and reproducibility                       1.0
Required total                                          13.0
Optional conditional flow-matching bonus                +1.0
```
