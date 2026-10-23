"""Small shared helpers for HW3 data, evaluation, and visualization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


X_MIN, X_MAX = 0.0, 70.4
Y_MIN, Y_MAX = -40.0, 40.0
Z_MIN, Z_MAX = -3.0, 1.0
RESOLUTION = 0.2
OUTPUT_STRIDE = 4
GRID_H = int(round((Y_MAX - Y_MIN) / RESOLUTION))
GRID_W = int(round((X_MAX - X_MIN) / RESOLUTION))
OUT_H, OUT_W = GRID_H // OUTPUT_STRIDE, GRID_W // OUTPUT_STRIDE
OUTPUT_RESOLUTION = RESOLUTION * OUTPUT_STRIDE
NUM_CLASSES = 1
MAX_OBJECTS = 64
CLASS_NAMES = ["Car"]
CHANNEL_NAMES = ["max height", "mean height", "intensity", "density"]
CAMERA_FOV_HALF_ANGLE_DEG = 41.0

CODE_ROOT = Path(__file__).resolve().parent
HW3_ROOT = CODE_ROOT.parent
DATA_ROOT = HW3_ROOT / "handout/data" if CODE_ROOT.name == "solution" else CODE_ROOT / "data"
OUTPUT_ROOT = CODE_ROOT / "outputs"


def metric_to_grid(x: np.ndarray, y: np.ndarray):
    """Map metric LiDAR x/y to integer BEV row/column coordinates."""
    columns = np.floor((x - X_MIN) / RESOLUTION).astype(np.int64)
    y_bins = np.floor((y - Y_MIN) / RESOLUTION).astype(np.int64)
    rows = (GRID_H - 1) - y_bins
    valid = (rows >= 0) & (rows < GRID_H) & (columns >= 0) & (columns < GRID_W)
    return rows, columns, valid


def metric_to_output_grid(x: np.ndarray, y: np.ndarray):
    """Map metric LiDAR x/y to continuous stride-4 output coordinates u/v."""
    u = ((x - X_MIN) / OUTPUT_RESOLUTION).astype(np.float32)
    # Output rows run opposite to metric y. A center at the lower edge of a
    # row belongs to that row, so move exact integer boundaries one float32
    # step downward before taking floor(v). This also keeps y == Y_MIN inside.
    v = (OUT_H - (y - Y_MIN) / OUTPUT_RESOLUTION).astype(np.float32)
    v = np.nextafter(v, np.float32(-np.inf))
    valid = (
        (x >= X_MIN) & (x < X_MAX) & (y >= Y_MIN) & (y < Y_MAX)
        & (u >= 0.0) & (u < OUT_W) & (v >= 0.0) & (v < OUT_H)
    )
    return u, v, valid


def output_grid_to_metric(u: np.ndarray, v: np.ndarray):
    """Invert continuous output-grid coordinates to metric LiDAR x/y."""
    x = X_MIN + u * OUTPUT_RESOLUTION
    y = Y_MIN + (OUT_H - v) * OUTPUT_RESOLUTION
    return x.astype(np.float32), y.astype(np.float32)


def gaussian2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    half_y, half_x = [(size - 1.0) / 2.0 for size in shape]
    y, x = np.ogrid[-half_y : half_y + 1, -half_x : half_x + 1]
    gaussian = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def gaussian_radius(height: float, width: float, min_overlap: float = 0.1) -> int:
    """CenterNet radius for an object footprint measured in output cells."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + np.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / 2.0
    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))) / 2.0
    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    r3 = (b3 + np.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / (2.0 * a3)
    return max(1, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    """Draw a truncated 2D Gaussian in place using a maximum merge."""
    diameter = 2 * radius + 1
    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6.0)
    x, y = center
    left, right = min(x, radius), min(heatmap.shape[1] - x - 1, radius)
    top, bottom = min(y, radius), min(heatmap.shape[0] - y - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    target = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    source = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(target, source, out=target)


def gather_map(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a BxCxHxW feature map at BxM flat spatial indices."""
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.shape[0], -1, features.shape[-1])
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return features.gather(1, gather_indices)


class DetectionDataset(Dataset):
    """Load a compact processed split with quantized BEV inputs."""

    def __init__(self, path: Path):
        archive = np.load(path, allow_pickle=False)
        self.bev = archive["bev"]
        self.heatmap = archive["heatmap"]
        self.inds = archive["inds"]
        self.mask = archive["mask"]
        self.reg = archive["reg"]
        self.height = archive["height"]
        self.dims = archive["dims"]
        self.rot = archive["rot"]
        self.boxes = archive["boxes"]
        self.box_mask = archive["box_mask"]

    def __len__(self):
        return len(self.bev)

    def __getitem__(self, index):
        target = {
            "heatmap": torch.from_numpy(self.heatmap[index].astype(np.float32) / 255.0),
            "inds": torch.from_numpy(self.inds[index].astype(np.int64)),
            "mask": torch.from_numpy(self.mask[index].astype(np.float32)),
            "reg": torch.from_numpy(self.reg[index].astype(np.float32)),
            "height": torch.from_numpy(self.height[index].astype(np.float32)),
            "dims": torch.from_numpy(self.dims[index].astype(np.float32)),
            "rot": torch.from_numpy(self.rot[index].astype(np.float32)),
        }
        return (
            torch.from_numpy(self.bev[index].astype(np.float32) / 255.0),
            target,
            torch.from_numpy(self.boxes[index].astype(np.float32)),
            torch.from_numpy(self.box_mask[index].astype(bool)),
        )


def load_debug_frames(path: Path):
    """Return variable-length real LiDAR sweeps and boxes from a debug archive."""
    archive = np.load(path, allow_pickle=True)
    return archive["points"], archive["boxes"], archive["frame_ids"]


def bev_to_rgb(bev: np.ndarray) -> np.ndarray:
    """Colorize occupied BEV cells using height, intensity, and density."""
    occupied = np.max(bev, axis=0) > 0
    saliency = np.clip(0.25 * bev[0] + 0.15 * bev[2] + 0.60 * bev[3], 0.0, 1.0)
    rgb = (plt.get_cmap("inferno")(saliency)[..., :3] * 255).astype(np.uint8)
    rgb[~occupied] = 0
    return rgb


def box_corners(box: np.ndarray) -> np.ndarray:
    """Return four counter-clockwise BEV corners for [x,y,z,l,w,h,yaw]."""
    x, y, _, length, width, _, yaw = box
    local = np.array(
        [[length / 2, width / 2], [-length / 2, width / 2],
         [-length / 2, -width / 2], [length / 2, -width / 2]],
        dtype=np.float32,
    )
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rotation.T + np.array([x, y])


def metric_to_pixel(points: np.ndarray, shape: tuple[int, int]):
    columns = (points[:, 0] - X_MIN) * shape[1] / (X_MAX - X_MIN)
    rows = (shape[0] - 1) - (points[:, 1] - Y_MIN) * shape[0] / (Y_MAX - Y_MIN)
    return columns, rows


def draw_camera_fov(axis, shape: tuple[int, int]):
    """Shade the approximate front-camera region covered by KITTI labels."""
    side_x = Y_MAX / np.tan(np.deg2rad(CAMERA_FOV_HALF_ANGLE_DEG))
    polygon = np.array([
        [X_MIN, 0.0], [side_x, Y_MAX], [X_MAX, Y_MAX],
        [X_MAX, Y_MIN], [side_x, Y_MIN],
    ])
    columns, rows = metric_to_pixel(polygon, shape)
    axis.fill(columns, rows, color="deepskyblue", alpha=0.08,
              label="Approx. camera-label FOV")
    for y in (Y_MIN, Y_MAX):
        columns, rows = metric_to_pixel(np.array([[X_MIN, 0.0], [side_x, y]]), shape)
        axis.plot(columns, rows, color="deepskyblue", linewidth=1.0,
                  linestyle="--", alpha=0.85)


def draw_boxes(axis, boxes: np.ndarray, shape: tuple[int, int], color: str, label: str, scores=None):
    for index, box in enumerate(boxes):
        corners = box_corners(box)
        corners = np.concatenate([corners, corners[:1]], axis=0)
        columns, rows = metric_to_pixel(corners, shape)
        axis.plot(columns, rows, color=color, linewidth=1.4)
        center = box[:2]
        front = center + 0.5 * box[3] * np.array([np.cos(box[6]), np.sin(box[6])])
        head_columns, head_rows = metric_to_pixel(np.stack([center, front]), shape)
        axis.plot(head_columns, head_rows, color=color, linewidth=1.2)
        if scores is not None:
            axis.text(columns[0], rows[0], f"{scores[index]:.2f}", color=color, fontsize=6)
    if len(boxes):
        axis.plot([], [], color=color, label=label)


def save_bev_boxes(bev, gt_boxes, path: Path, pred_boxes=None, pred_scores=None, title=""):
    """Save a metric BEV with green GT and optional cyan predictions."""
    image = bev_to_rgb(bev)
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    axis.imshow(image)
    draw_camera_fov(axis, image.shape[:2])
    draw_boxes(axis, np.asarray(gt_boxes), image.shape[:2], "lime", "GT")
    if pred_boxes is not None:
        draw_boxes(axis, np.asarray(pred_boxes), image.shape[:2], "cyan", "Prediction", pred_scores)
    axis.set_title(title)
    axis.set_xlabel("x forward")
    axis.set_ylabel("y left")
    if len(gt_boxes) or (pred_boxes is not None and len(pred_boxes)):
        axis.legend(loc="upper right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_bev_channels(bev: np.ndarray, path: Path):
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for index, axis in enumerate(axes):
        image = axis.imshow(bev[index], cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(CHANNEL_NAMES[index])
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_target_debug(heatmap: np.ndarray, bev: np.ndarray, gt_boxes, decoded_boxes, path: Path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].imshow(heatmap[0], cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("Center heatmap")
    axes[0].axis("off")
    image = bev_to_rgb(bev)
    axes[1].imshow(image)
    draw_camera_fov(axes[1], image.shape[:2])
    draw_boxes(axes[1], np.asarray(gt_boxes), image.shape[:2], "lime", "GT")
    draw_boxes(axes[1], np.asarray(decoded_boxes), image.shape[:2], "cyan", "Decoded")
    axes[1].set_title("Encode/decode round trip")
    axes[1].axis("off")
    if len(gt_boxes) or len(decoded_boxes):
        axes[1].legend(loc="upper right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                     np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) * 0.5)


def polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip one convex counter-clockwise polygon by another."""
    def cross2d(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    output = [point for point in subject]
    for index in range(len(clip)):
        start, end = clip[index], clip[(index + 1) % len(clip)]
        input_points, output = output, []
        if not input_points:
            break

        def inside(point):
            return cross2d(end - start, point - start) >= -1e-8

        def intersection(first, second):
            edge, segment = end - start, second - first
            denominator = cross2d(segment, edge)
            if abs(denominator) < 1e-8:
                return second
            return first + segment * (cross2d(start - first, edge) / denominator)

        previous = input_points[-1]
        for current in input_points:
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
            previous = current
    return np.asarray(output, dtype=np.float32)


def rotated_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_polygon, second_polygon = box_corners(first), box_corners(second)
    intersection = polygon_area(polygon_clip(first_polygon, second_polygon))
    union = polygon_area(first_polygon) + polygon_area(second_polygon) - intersection
    return intersection / union if union > 0 else 0.0


def rotated_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.1):
    order, keep = np.argsort(-scores), []
    if len(order) == 0:
        return np.zeros(0, dtype=np.int64)
    corners = np.asarray([box_corners(box) for box in boxes])
    lower, upper = corners.min(axis=1), corners.max(axis=1)
    while len(order):
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        overlaps = np.all(
            np.minimum(upper[current], upper[remaining])
            > np.maximum(lower[current], lower[remaining]),
            axis=1,
        )
        suppress = np.zeros(len(remaining), dtype=bool)
        for position in np.flatnonzero(overlaps):
            suppress[position] = rotated_iou(boxes[current], boxes[remaining[position]]) > threshold
        order = remaining[~suppress]
    return np.asarray(keep, dtype=np.int64)


def evaluate_frames(predictions: list[dict], gt_boxes: list[np.ndarray], iou_threshold=0.5):
    """Compute one-class precision/recall and interpolated AP over all frames."""
    ranked = []
    total_gt = sum(len(boxes) for boxes in gt_boxes)
    matched = [np.zeros(len(boxes), dtype=bool) for boxes in gt_boxes]
    for frame, prediction in enumerate(predictions):
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            ranked.append((float(score), frame, box))
    ranked.sort(key=lambda item: item[0], reverse=True)
    true_positive, false_positive = [], []
    for _, frame, box in ranked:
        overlaps = np.array([rotated_iou(box, gt) for gt in gt_boxes[frame]])
        if len(overlaps):
            best = int(overlaps.argmax())
            is_match = overlaps[best] >= iou_threshold and not matched[frame][best]
        else:
            best, is_match = -1, False
        if is_match:
            matched[frame][best] = True
        true_positive.append(float(is_match))
        false_positive.append(float(not is_match))
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / max(total_gt, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall_curve = np.concatenate([[0.0], recall, [1.0]])
    precision_curve = np.concatenate([[0.0], precision, [0.0]])
    for index in range(len(precision_curve) - 2, -1, -1):
        precision_curve[index] = max(precision_curve[index], precision_curve[index + 1])
    changes = np.where(recall_curve[1:] != recall_curve[:-1])[0]
    average_precision = float(np.sum(
        (recall_curve[changes + 1] - recall_curve[changes]) * precision_curve[changes + 1]
    ))
    return {
        "ap_50": average_precision,
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "ground_truth_boxes": int(total_gt),
        "predicted_boxes": int(len(ranked)),
    }


def save_pr_curve(metrics: dict, path: Path):
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.plot(metrics["recall_curve"], metrics["precision_curve"], linewidth=2)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
             title=f"Validation PR curve, AP@0.5={metrics['ap_50']:.3f}")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_training_curves(history: dict, path: Path):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Total loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_ap_50"], color="tab:orange")
    axes[1].set(xlabel="Epoch", ylabel="AP@0.5", title="Validation detection")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
