"""Task 2: encode metric 3D boxes as CenterPoint supervision targets.

Run directly with: python task2_targets.py
"""

import numpy as np

from hw3_utils import (
    DATA_ROOT, MAX_OBJECTS, NUM_CLASSES, OUT_H, OUT_W, OUTPUT_RESOLUTION,
    OUTPUT_ROOT, draw_gaussian, gaussian_radius, load_debug_frames,
    metric_to_output_grid, output_grid_to_metric, save_target_debug, write_json,
)
from task1_bev import rasterize_points


def encode_targets(boxes: np.ndarray) -> dict[str, np.ndarray]:
    """Encode `[M,7]` boxes into one dense heatmap and sparse regressions.

    Each box is `[x,y,z,length,width,height,yaw]`. Draw one Gaussian on the
    `[1,100,88]` center heatmap. At the integer center cell, store the flat
    index, sub-cell `(du,dv)`, center z, log-dimensions, and `(sin(yaw),
    cos(yaw))`. The mask marks valid rows among `MAX_OBJECTS` slots.
    """
    # ------------------------ Task 2A Modify your code below --------------------
    return {
        "heatmap": np.zeros((NUM_CLASSES, OUT_H, OUT_W), dtype=np.float32),
        "inds": np.zeros(MAX_OBJECTS, dtype=np.int64),
        "mask": np.zeros(MAX_OBJECTS, dtype=np.uint8),
        "reg": np.zeros((MAX_OBJECTS, 2), dtype=np.float32),
        "height": np.zeros((MAX_OBJECTS, 1), dtype=np.float32),
        "dims": np.zeros((MAX_OBJECTS, 3), dtype=np.float32),
        "rot": np.zeros((MAX_OBJECTS, 2), dtype=np.float32),
    }
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def decode_targets(targets: dict[str, np.ndarray]) -> np.ndarray:
    """Invert `encode_targets` and return decoded metric boxes `[M,7]`."""
    # ------------------------ Task 2B Modify your code below --------------------
    return np.zeros((0, 7), dtype=np.float32)
    # ---------------------- DO NOT MODIFY CODE OUTSIDE THE BLOCK ----------------


def main():
    points, boxes, frame_ids = load_debug_frames(DATA_ROOT / "debug_frames.npz")
    output = OUTPUT_ROOT / "task2"
    targets = encode_targets(boxes[0])
    decoded = decode_targets(targets)
    bev = rasterize_points(points[0])
    save_target_debug(targets["heatmap"], bev, boxes[0], decoded, output / "target_roundtrip.png")

    count = min(len(boxes[0]), len(decoded))
    center_error = (
        np.linalg.norm(boxes[0][:count, :2] - decoded[:count, :2], axis=1)
        if count else np.array([], dtype=np.float32)
    )
    metrics = {
        "frame_id": str(frame_ids[0]),
        "heatmap_shape": list(targets["heatmap"].shape),
        "heatmap_max": float(targets["heatmap"].max()),
        "ground_truth_boxes": int(len(boxes[0])),
        "encoded_boxes": int(targets["mask"].sum()),
        "decoded_boxes": int(len(decoded)),
        "mean_center_error_m": float(center_error.mean()) if len(center_error) else None,
    }
    write_json(output / "metrics.json", metrics)
    print("Checkpoint 2: target_roundtrip.png and metrics.json")
    return metrics


if __name__ == "__main__":
    print("Task 2 complete:", main())
