"""Task 3: train the compact CenterPoint-style detector and its losses.

Run directly with: python task3_detector.py
Use `python task3_detector.py --debug` for the supplied tiny split.
"""

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hw3_utils import (
    DATA_ROOT, NUM_CLASSES, OUTPUT_ROOT, DetectionDataset, evaluate_frames,
    gather_map, save_training_curves, write_json,
)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
            if stride != 1 or input_channels != output_channels else nn.Identity()
        )

    def forward(self, features):
        hidden = F.relu(self.conv1(features), inplace=True)
        return F.relu(self.conv2(hidden) + self.skip(features), inplace=True)


class PredictionHead(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 1),
        )

    def forward(self, features):
        return self.layers(features)


class CenterDetector(nn.Module):
    """Map `[B,4,400,352]` BEV input to five stride-4 dense heads."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1, bias=False), nn.ReLU())
        self.block1 = ResidualBlock(64, 64)
        self.block2 = ResidualBlock(64, 64)
        self.down1 = ResidualBlock(64, 128, stride=2)
        self.block3 = ResidualBlock(128, 128)
        self.block4 = ResidualBlock(128, 128)
        self.down2 = ResidualBlock(128, 128, stride=2)
        self.block5 = ResidualBlock(128, 128)
        self.block6 = ResidualBlock(128, 128)
        self.heads = nn.ModuleDict({
            "heatmap": PredictionHead(NUM_CLASSES), "reg": PredictionHead(2),
            "height": PredictionHead(1), "dims": PredictionHead(3),
            "rot": PredictionHead(2),
        })
        nn.init.constant_(self.heads["heatmap"].layers[-1].bias, -2.19)

    def forward(self, bev):
        """Return heatmap/reg/height/dims/rot maps at `[B,*,100,88]`.

        Follow the exact left-to-right architecture declared in `__init__`:
        stem, block1, block2, down1, block3, block4, down2, block5, block6.
        Then apply every prediction head to the final `[B,128,100,88]` map.
        """
        # ----------------------- Task 3A Modify your code below -----------------
        features = bev.new_zeros((bev.shape[0], 128, bev.shape[2] // 4, bev.shape[3] // 4))
        return {name: head(features) for name, head in self.heads.items()}
        # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def focal_loss(probability, target):
    """CenterNet focal loss for sigmoid probabilities and Gaussian targets.

    Pixels with target exactly 1 are positives. Other pixels are negatives
    weighted by `(1-target)^4`. Use focusing power 2 and normalize by the
    number of positive centers, or return only the negative term if none exist.
    """
    # ----------------------- Task 3B Modify your code below -----------------
    return F.binary_cross_entropy(probability, target)
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def regression_l1(prediction_map, target, indices, mask):
    """Gather sparse center predictions and return masked L1 loss.

    `prediction_map` is `[B,C,100,88]`, `target` is `[B,M,C]`, `indices` is
    `[B,M]`, and `mask` is `[B,M]`. Normalize the summed absolute error by the
    number of valid objects (not by the number of regression channels).
    """
    # ----------------------- Task 3C Modify your code below -----------------
    return (prediction_map * 0.0).sum()
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def compute_losses(predictions, targets):
    probability = torch.sigmoid(predictions["heatmap"]).clamp(1e-4, 1 - 1e-4)
    losses = {"heatmap": focal_loss(probability, targets["heatmap"])}
    for name in ("reg", "height", "dims", "rot"):
        losses[name] = regression_l1(
            predictions[name], targets[name], targets["inds"], targets["mask"]
        )
    losses["total"] = sum(losses.values())
    return losses


def move_targets(targets, device):
    return {name: value.to(device) for name, value in targets.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    from task4_inference import decode_predictions

    model.eval()
    losses, predictions, ground_truth = [], [], []
    for bev, targets, boxes, box_mask in loader:
        bev, targets = bev.to(device), move_targets(targets, device)
        outputs = model(bev)
        losses.append(float(compute_losses(outputs, targets)["total"].item()))
        decoded = decode_predictions(outputs, score_threshold=0.05)
        predictions.extend(decoded)
        for frame in range(len(boxes)):
            ground_truth.append(boxes[frame][box_mask[frame]].numpy())
    metrics = evaluate_frames(predictions, ground_truth)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    output = OUTPUT_ROOT / "task3"
    output.mkdir(parents=True, exist_ok=True)
    debug = args.debug
    if not debug:
        missing = [name for name in ("train.npz", "val.npz", "test.npz")
                   if not (DATA_ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing full splits {missing}. Run python data/download_data.py "
                "or explicitly choose --debug for the tiny smoke run."
            )
    train_path = DATA_ROOT / ("debug_train.npz" if debug else "train.npz")
    val_path = DATA_ROOT / ("debug_val.npz" if debug else "val.npz")
    epochs = args.epochs or (2 if debug else 20)
    batch_size = args.batch_size or (2 if debug else 8)
    early_stopping_patience = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(498)

    train_loader = DataLoader(
        DetectionDataset(train_path), batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(498),
    )
    val_loader = DataLoader(
        DetectionDataset(val_path), batch_size=batch_size,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(499),
    )
    model = CenterDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_ap_50": []}
    best_ap, best_epoch, epochs_without_improvement = -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for bev, targets, _, _ in train_loader:
            bev, targets = bev.to(device), move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_losses(model(bev), targets)["total"]
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        validation = evaluate(model, val_loader, device)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(validation["loss"])
        history["val_ap_50"].append(validation["ap_50"])
        if validation["ap_50"] > best_ap:
            best_ap, best_epoch = validation["ap_50"], epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output / "detector.pt")
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch + 1:02d}/{epochs}: train={history['train_loss'][-1]:.4f}, "
            f"val={validation['loss']:.4f}, val AP@0.5={validation['ap_50']:.3f}"
        )
        if not debug and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping: validation AP did not improve for {early_stopping_patience} epochs.")
            break

    save_training_curves(history, output / "training_curves.png")
    metrics = {
        "mode": "debug" if debug else "full",
        "max_epochs": epochs,
        "epochs_completed": len(history["train_loss"]),
        "early_stopping_patience": early_stopping_patience,
        "best_epoch": best_epoch,
        "best_validation_ap_50": best_ap,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_time_seconds": time.perf_counter() - start,
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 3: detector.pt, training_curves.png, and metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 3 complete:", main())
