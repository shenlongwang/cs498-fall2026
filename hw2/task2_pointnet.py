"""Task 2: train a compact PointNet part segmenter.

Run directly with: python task2_pointnet.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hw2_utils import PART_NAMES, PartDataset, normalize_points, plot_point_clouds, plot_training, write_json
from task1_data import augment_points, compute_confusion_matrix, iou_from_confusion


class PointNetSegmenter(nn.Module):
    def __init__(self, num_classes=len(PART_NAMES)):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(6, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.2), nn.Conv1d(128, num_classes, 1),
        )

    # Student guidance:
    # Fixed architecture (B=batch, N=2048 points, C=5 classes):
    # xyz/rgb (B,N,3) -> concatenate (B,N,6) -> transpose (B,6,N)
    # -> shared 1x1 Conv1d 6->64->128->256 -> local (B,256,N)
    # -> max over N (B,256,1) -> repeat (B,256,N)
    # -> concatenate local+global (B,512,N)
    # -> shared 1x1 Conv1d 512->128->C -> transpose logits (B,N,C).
    # Keep the declared layers unchanged and preserve the input point order.
    def forward(self, xyz, rgb):
        """Return logits with shape (batch, points, classes).

        Math: h_i = point_mlp([xyz_i, rgb_i]), g = max_i h_i, and
        logits_i = classifier([h_i, g]). The max is over points, and the same
        global feature g is copied back to every point.

        Hint: after arranging features as (batch, channels, points), a Conv1d
        with kernel size 1 applies the same linear map independently to all N
        points. Stacking these layers with nonlinearities gives a shared MLP.
        Reference: Qi et al., "PointNet" (2017),
        https://arxiv.org/abs/1612.00593.
        """
        # ---------------- Task 2: INSERT YOUR CODE BELOW -----------------
        batch, points, _ = xyz.shape
        empty = xyz.new_zeros((batch, 512, points))
        return self.classifier(empty).transpose(1, 2)
        # ------------- DO NOT MODIFY CODE OUTSIDE THIS BLOCK -------------


def metric_functions_ready():
    """Check Task 1's perfect-prediction example before expensive training."""
    labels = np.arange(len(PART_NAMES))
    confusion = compute_confusion_matrix(labels, labels, len(PART_NAMES))
    iou = iou_from_confusion(confusion)
    return (confusion.shape == (len(PART_NAMES), len(PART_NAMES))
            and np.array_equal(confusion, np.eye(len(PART_NAMES), dtype=np.int64))
            and np.shape(iou) == (len(PART_NAMES),)
            and np.allclose(iou, 1.0))


@torch.no_grad()
def evaluate(model, loader, device, class_weights):
    model.eval()
    predictions, targets = [], []
    loss_sum = 0.0
    weight_sum = 0.0
    for xyz, rgb, labels in loader:
        xyz = normalize_points(xyz.to(device))
        rgb, labels = rgb.to(device), labels.to(device)
        logits = model(xyz, rgb)
        loss_sum += nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.flatten(), weight=class_weights, reduction="sum"
        ).item()
        weight_sum += class_weights[labels].sum().item()
        predictions.append(logits.argmax(-1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    confusion = compute_confusion_matrix(prediction, target, len(PART_NAMES))
    per_class_iou = iou_from_confusion(confusion)
    if confusion.sum() != target.size or not np.isfinite(per_class_iou).any():
        raise ValueError("Invalid Task 1 metrics: verify confusion counts and IoU with the public tests.")
    return (
        float(np.nanmean(per_class_iou)),
        per_class_iou,
        prediction,
        loss_sum / weight_sum,
        float(np.mean(prediction == target)),
    )


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs/task2"
    output.mkdir(parents=True, exist_ok=True)
    if not metric_functions_ready():
        metrics = {"status": "starter_placeholder",
                   "reason": "Complete and test Task 1 confusion matrix and IoU before training or testing."}
        write_json(output / "metrics.json", metrics)
        print(metrics["reason"])
        return metrics
    torch.manual_seed(498)
    rng = np.random.default_rng(498)

    train_data = PartDataset(root / "data/train.npz", lambda xyz: augment_points(xyz, rng))
    val_data = PartDataset(root / "data/val.npz")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = PointNetSegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train_data.labels.reshape(-1), minlength=len(PART_NAMES))
    class_weights = torch.tensor(np.sqrt(counts.sum() / counts), dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()

    # Final test mode: run this only after validation has selected the checkpoint.
    # Do not change the model, checkpoint, or hyperparameters after seeing test results.
    if "--test" in sys.argv[1:]:
        test_data = PartDataset(root / "data/student_test.npz")
        test_loader = DataLoader(test_data, batch_size=16)
        checkpoint = output / "pointnet.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("Run python task2_pointnet.py before the final test.")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_miou, per_class_iou, _, test_loss, accuracy = evaluate(
            model, test_loader, device, class_weights
        )
        metrics = {
            "split": "test",
            "checkpoint_selected_by": "highest validation_miou",
            "test_miou": test_miou,
            "test_loss": test_loss,
            "overall_point_accuracy": accuracy,
            "per_class_iou": per_class_iou.tolist(),
        }
        write_json(output / "test_metrics.json", metrics)
        print("Final test complete: test_metrics.json. Do not tune using this result.")
        return metrics

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, best_epoch = -1.0, 0
    epochs = 100
    start = time.perf_counter()

    # Checkpoint 2A: train PointNet and plot its learning curves.
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xyz, rgb, labels in train_loader:
            xyz = normalize_points(xyz.to(device))
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(xyz, rgb)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=class_weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        val_miou, _, _, val_loss, _ = evaluate(model, val_loader, device, class_weights)
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        if val_miou > best_miou:
            best_miou, best_epoch = val_miou, epoch + 1
            torch.save(model.state_dict(), output / "pointnet.pt")
        print(
            f"epoch {epoch + 1:03d}/{epochs}: train loss={train_loss:.4f}, "
            f"val loss={val_loss:.4f}, val mIoU={val_miou:.3f}"
        )

    training_time = time.perf_counter() - start
    plot_training(history, output / "training_curves.png")
    print("Checkpoint 2A: training_curves.png")

    # Checkpoint 2B: evaluate the best checkpoint and visualize two chairs.
    model.load_state_dict(torch.load(output / "pointnet.pt", map_location=device, weights_only=True))
    val_miou, per_class_iou, prediction, val_loss, accuracy = evaluate(
        model, val_loader, device, class_weights
    )
    example_indices = [0, 7]
    example_xyz = normalize_points(torch.from_numpy(val_data.xyz[example_indices])).numpy()
    plot_point_clouds(
        [example_xyz[0], example_xyz[0], example_xyz[1], example_xyz[1]],
        [val_data.labels[0], prediction[0], val_data.labels[7], prediction[7]],
        [
            "Sample 1: ground truth", "Sample 1: PointNet",
            "Sample 2: ground truth", "Sample 2: PointNet",
        ],
        output / "predictions.png",
    )
    metrics = {
        "validation_miou": val_miou,
        "validation_loss": val_loss,
        "overall_point_accuracy": accuracy,
        "per_class_iou": per_class_iou.tolist(),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_time_seconds": training_time,
        "checkpoint_selection": "highest validation_miou",
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 2B: predictions.png, metrics.json, and pointnet.pt")
    return metrics


if __name__ == "__main__":
    print("Task 2 complete:", main())
