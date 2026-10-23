"""Task 4: decode dense detector outputs, select a threshold, and evaluate.

Run directly with: python task4_inference.py
After every choice is frozen, run once with: python task4_inference.py --test
"""

import argparse
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hw3_utils import (
    DATA_ROOT, OUT_H, OUTPUT_RESOLUTION, OUTPUT_ROOT, DetectionDataset,
    X_MIN, Y_MIN, evaluate_frames, gather_map, rotated_nms, save_bev_boxes,
    save_pr_curve, write_json,
)


def suppress_nonmaxima(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local maxima in a `[B,C,H,W]` probability heatmap."""
    # ----------------------- Task 4A Modify your code below -----------------
    return heatmap
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def topk_centers(heatmap: torch.Tensor, count: int):
    batch, classes, height, width = heatmap.shape
    count = min(count, height * width)
    class_scores, class_indices = torch.topk(heatmap.view(batch, classes, -1), count)
    class_indices = class_indices % (height * width)
    rows = (class_indices // width).float()
    columns = (class_indices % width).float()
    scores, ranking = torch.topk(class_scores.view(batch, -1), count)
    class_ids = (ranking // count).int()
    flat_indices = class_indices.view(batch, -1).gather(1, ranking)
    rows = rows.view(batch, -1).gather(1, ranking)
    columns = columns.view(batch, -1).gather(1, ranking)
    return scores, flat_indices, class_ids, rows, columns


def decode_predictions(predictions: dict[str, torch.Tensor], score_threshold=0.05, topk=100):
    """Decode five dense heads into scored metric boxes for every batch item.

    Apply sigmoid and `suppress_nonmaxima`, select the top-k centers, gather
    regression values, add sub-cell offsets, and convert output-grid locations
    back to metric LiDAR coordinates. Exponentiate log-dimensions and recover
    yaw with `atan2(sin,cos)`. Finally apply score filtering and rotated NMS.
    """
    # ----------------------- Task 4B Modify your code below -----------------
    batch = predictions["heatmap"].shape[0]
    return [
        {"boxes": np.zeros((0, 7), dtype=np.float32),
         "scores": np.zeros(0, dtype=np.float32),
         "classes": np.zeros(0, dtype=np.int64)}
        for _ in range(batch)
    ]
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


@torch.no_grad()
def predict_split(model, loader, device, threshold):
    model.eval()
    predictions, ground_truth, bev_images = [], [], []
    for bev, _, boxes, box_mask in loader:
        decoded = decode_predictions(model(bev.to(device)), score_threshold=threshold)
        predictions.extend(decoded)
        for frame in range(len(boxes)):
            ground_truth.append(boxes[frame][box_mask[frame]].numpy())
            bev_images.append(bev[frame].numpy())
    return predictions, ground_truth, bev_images


def filter_predictions(predictions, threshold):
    """Keep scored boxes at one operating threshold without changing AP ranking."""
    filtered = []
    for prediction in predictions:
        keep = prediction["scores"] >= threshold
        filtered.append({name: values[keep] for name, values in prediction.items()})
    return filtered


def main():
    from task3_detector import CenterDetector

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true", help="use the tiny smoke-run split")
    mode.add_argument("--test", action="store_true", help="run the frozen final test once")
    args = parser.parse_args()

    output = OUTPUT_ROOT / "task4"
    output.mkdir(parents=True, exist_ok=True)
    test_mode = args.test
    checkpoint = OUTPUT_ROOT / "task3/detector.pt"
    training_metrics = OUTPUT_ROOT / "task3/metrics.json"
    if not checkpoint.is_file() or not training_metrics.is_file():
        raise FileNotFoundError("Train Task 3 before running inference")
    expected_mode = "debug" if args.debug else "full"
    if json.loads(training_metrics.read_text())["mode"] != expected_mode:
        raise ValueError(f"Task 3 checkpoint must be from a {expected_mode} run")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if test_mode:
        if (output / "test_metrics.json").exists():
            raise RuntimeError("The final test has already been evaluated; do not rerun it")
        threshold_path = output / "selected_threshold.json"
        if not threshold_path.is_file():
            raise FileNotFoundError("Run full validation before the final test")
        selection = json.loads(threshold_path.read_text())
        if selection.get("source_split") != "val" or selection.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("The frozen threshold must come from validation of this checkpoint")
        frozen_threshold = selection["score_threshold"]
    split = "debug_val" if args.debug else "test" if test_mode else "val"
    path = DATA_ROOT / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python data/download_data.py or use --debug")
    if not args.debug and not all((DATA_ROOT / f"{name}.npz").is_file()
                                  for name in ("train", "val", "test")):
        raise FileNotFoundError("The full train/val/test package is required for final evaluation")
    loader = DataLoader(DetectionDataset(path), batch_size=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CenterDetector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    prediction, truth, images = predict_split(model, loader, device, threshold=0.01)
    ranked_metrics = evaluate_frames(prediction, truth)

    if test_mode:
        threshold = frozen_threshold
    else:
        best = (-1.0, 0.05)
        for threshold in (0.01, 0.03, 0.05, 0.10, 0.20):
            operating = evaluate_frames(filter_predictions(prediction, threshold), truth)
            precision, recall = operating["precision"], operating["recall"]
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best[0]:
                best = (f1, threshold)
        _, threshold = best
        write_json(output / "selected_threshold.json", {
            "score_threshold": threshold,
            "source_split": split,
            "checkpoint_sha256": checkpoint_hash,
        })

    displayed_prediction = filter_predictions(prediction, threshold)
    operating = evaluate_frames(displayed_prediction, truth)
    metrics = ranked_metrics
    metrics["precision"] = operating["precision"]
    metrics["recall"] = operating["recall"]
    metrics["predicted_boxes"] = operating["predicted_boxes"]
    metrics["split"] = split
    metrics["score_threshold"] = threshold

    if not test_mode:
        save_pr_curve(metrics, output / "pr_curve.png")
        example_indices = sorted(range(len(truth)), key=lambda index: -len(truth[index]))[:2]
        for output_index, frame_index in enumerate(example_indices):
            save_bev_boxes(
                images[frame_index], truth[frame_index], output / f"prediction_{output_index}.png",
                displayed_prediction[frame_index]["boxes"], displayed_prediction[frame_index]["scores"],
                title=f"Validation frame {frame_index}: GT and detections",
            )
        print("Checkpoint 4: predictions, pr_curve.png, metrics.json, and selected_threshold.json")
    else:
        print("Final test complete: test_metrics.json. Do not tune after this run.")
    summary = {name: value for name, value in metrics.items() if not name.endswith("_curve")}
    write_json(output / ("test_metrics.json" if test_mode else "metrics.json"), summary)
    return summary


if __name__ == "__main__":
    print("Task 4 complete:", main())
