"""Task 3: train a Transformer part segmenter.

Run directly with: python task3_transformer.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hw2_utils import PART_NAMES, PartDataset, normalize_points, plot_model_comparison, plot_point_clouds, plot_training, write_json
from task1_data import augment_points
from task2_pointnet import PointNetSegmenter, evaluate, metric_functions_ready


class SelfAttention(nn.Module):
    def __init__(self, width, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_width = width // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    # Student guidance:
    # Fixed attention shapes (N=2048, width=96, heads=4, head_width=24):
    # features (B,N,96) -> packed qkv (B,N,288)
    # -> Q, K, V each (B,4,N,24)
    # -> scaled dot-product attention (B,4,N,24)
    # -> merge heads (B,N,96) -> output projection (B,N,96).
    def forward(self, features):
        """Apply multi-head attention and return shape (batch, points, width).

        Math: Q = X W_Q, K = X W_K, V = X W_V, and
        Attention(Q,K,V) = softmax(Q K^T / sqrt(head_width)) V.
        F.scaled_dot_product_attention performs the scaling and softmax.

        Hint: self.qkv produces Q, K, and V with one packed linear projection.
        Its output has shape (batch, points, 3 * width); reshape and permute are
        useful for exposing the Q/K/V and attention-head dimensions.
        Reference: Vaswani et al., "Attention Is All You Need" (2017),
        https://arxiv.org/abs/1706.03762.
        """
        # ----------- SelfAttention.forward: INSERT YOUR CODE BELOW -----------
        return torch.zeros_like(features)
        # ------------ DO NOT MODIFY CODE OUTSIDE THIS BLOCK --------------


class TransformerBlock(nn.Module):
    def __init__(self, width, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = SelfAttention(width, num_heads, dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, features):
        features = features + self.attention(self.attention_norm(features))
        return features + self.mlp(self.mlp_norm(features))


class TransformerSegmenter(nn.Module):
    def __init__(self, num_classes=len(PART_NAMES), width=96, num_heads=4, num_blocks=4, dropout=0.1):
        super().__init__()
        self.width = width
        self.input_projection = nn.Linear(6, width)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))

    # Student guidance:
    # Fixed architecture (B=batch, N=2048 points, C=5 classes):
    # concatenate xyz/rgb (B,N,6) -> input Linear 6->96 (B,N,96)
    # + position MLP xyz 3->96->96 (B,N,96)
    # -> 4 pre-LN TransformerBlocks, each preserving (B,N,96)
    #    [4-head attention; MLP 96->384->96; two residual connections]
    # -> LayerNorm + Linear 96->C -> raw logits (B,N,C).
    # Add position features; do not concatenate them. Preserve point order.
    def forward(self, xyz, rgb):
        """Embed XYZ+RGB, apply all Transformer blocks, and classify each point.

        Math: Z_0 = W_in [XYZ, RGB] + MLP_pos(XYZ),
        Z_l = TransformerBlock_l(Z_{l-1}), and logits = Head(Z_L).
        Preserve point order and return raw logits, not probabilities.
        """
        # ------- TransformerSegmenter.forward: INSERT YOUR CODE BELOW -------
        empty = xyz.new_zeros((*xyz.shape[:2], self.width))
        return self.head(empty)
        # ------------ DO NOT MODIFY CODE OUTSIDE THIS BLOCK --------------


# Student guidance:
# Inputs: logits has shape (B, N, C), labels has shape (B, N), and
# class_weights has shape (C,). Output should be one scalar loss tensor.
# Rough idea: flatten the batch and point dimensions of logits and labels,
# then call F.cross_entropy with the provided class weights.
def segmentation_loss(logits, labels, class_weights):
    """Return class-weighted cross entropy over every point.

    Math: L = -sum_i w[y_i] log softmax(logits_i)[y_i] / sum_i w[y_i].
    Use reduction="mean"; i ranges over all batch and point entries.
    Flatten only the batch and point axes before calling F.cross_entropy.
    """
    # ------------ segmentation_loss: INSERT YOUR CODE BELOW ------------
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    # ------------ DO NOT MODIFY CODE OUTSIDE THIS BLOCK --------------


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs/task3"
    output.mkdir(parents=True, exist_ok=True)
    if not metric_functions_ready():
        metrics = {"status": "starter_placeholder",
                   "reason": "Complete and test Task 1 confusion matrix and IoU before training or testing."}
        write_json(output / "metrics.json", metrics)
        print(metrics["reason"])
        return metrics
    pointnet_path = output.parent / "task2/pointnet.pt"
    if "--test" not in sys.argv[1:] and not pointnet_path.exists():
        metrics = {"status": "incomplete",
                   "reason": "Run task2_pointnet.py first; Task 3 requires its trained checkpoint for comparison."}
        write_json(output / "metrics.json", metrics)
        print(metrics["reason"])
        return metrics
    torch.manual_seed(498)
    rng = np.random.default_rng(498)

    train_data = PartDataset(root / "data/train.npz", lambda xyz: augment_points(xyz, rng))
    val_data = PartDataset(root / "data/val.npz")
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = TransformerSegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    counts = np.bincount(train_data.labels.reshape(-1), minlength=len(PART_NAMES))
    class_weights = torch.tensor(np.sqrt(counts.sum() / counts), dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()

    # Final test mode: run this only after validation has selected the checkpoint.
    # Do not change the model, checkpoint, or hyperparameters after seeing test results.
    if "--test" in sys.argv[1:]:
        test_data = PartDataset(root / "data/student_test.npz")
        test_loader = DataLoader(test_data, batch_size=8)
        checkpoint = output / "transformer.pt"
        if not checkpoint.exists():
            raise FileNotFoundError("Run python task3_transformer.py before the final test.")
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

    # Checkpoint 3A: train the Transformer and plot its curves.
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xyz, rgb, labels in train_loader:
            xyz = normalize_points(xyz.to(device))
            rgb, labels = rgb.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = segmentation_loss(model(xyz, rgb), labels, class_weights)
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
            torch.save(model.state_dict(), output / "transformer.pt")
        print(
            f"epoch {epoch + 1:03d}/{epochs}: train loss={train_loss:.4f}, "
            f"val loss={val_loss:.4f}, val mIoU={val_miou:.3f}"
        )

    training_time = time.perf_counter() - start
    plot_training(history, output / "training_curves.png")
    print("Checkpoint 3A: training_curves.png")

    # Checkpoint 3B: evaluate the best checkpoint and visualize two chairs.
    model.load_state_dict(torch.load(output / "transformer.pt", map_location=device, weights_only=True))
    val_miou, per_class_iou, prediction, val_loss, accuracy = evaluate(
        model, val_loader, device, class_weights
    )
    example_indices = [0, 7]
    example_xyz = normalize_points(torch.from_numpy(val_data.xyz[example_indices])).numpy()
    plot_point_clouds(
        [example_xyz[0], example_xyz[0], example_xyz[1], example_xyz[1]],
        [val_data.labels[0], prediction[0], val_data.labels[7], prediction[7]],
        [
            "Sample 1: ground truth", "Sample 1: Transformer",
            "Sample 2: ground truth", "Sample 2: Transformer",
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
    print("Checkpoint 3B: predictions.png, metrics.json, and transformer.pt")

    # Checkpoint 3C: compare both models on two fixed validation chairs.
    pointnet = PointNetSegmenter().to(device)
    pointnet.load_state_dict(torch.load(pointnet_path, map_location=device, weights_only=True))
    pointnet.eval()
    comparison_indices = [7, 12]
    xyz = normalize_points(torch.from_numpy(val_data.xyz[comparison_indices]).to(device))
    rgb = torch.from_numpy(val_data.rgb[comparison_indices]).to(device)
    with torch.no_grad():
        pointnet_prediction = pointnet(xyz, rgb).argmax(-1).cpu().numpy()
        transformer_prediction = model(xyz, rgb).argmax(-1).cpu().numpy()
    plot_model_comparison(
        xyz.cpu().numpy(),
        val_data.rgb[comparison_indices],
        val_data.labels[comparison_indices],
        pointnet_prediction,
        transformer_prediction,
        output / "model_comparison.png",
    )
    print("Checkpoint 3C: model_comparison.png")
    return metrics


if __name__ == "__main__":
    print("Task 3 complete:", main())
