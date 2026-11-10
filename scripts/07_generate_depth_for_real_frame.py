"""生成真实帧 DA2 深度，应用人工掩码并保存质量记录及掩码指纹。"""

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.da2_engine import get_estimator, save_depth_preview, save_raw_depth
from shared.intraop_masks import (
    depth_quality,
    mask_fingerprints,
    quality_summary,
    validate_case_masks,
    validate_depth_quality,
)


def main():
    case_dir = case_config.case_dir()
    depth_dir = case_config.depth_dir()
    os.makedirs(depth_dir, exist_ok=True)

    lap_path = os.path.join(case_dir, "image", "lap_undist.png")
    raw_npy = os.path.join(depth_dir, "da2_intraop_raw.npy")
    raw_vis = os.path.join(depth_dir, "da2_intraop_raw_preview.png")
    out_npy = os.path.join(depth_dir, "da2_intraop.npy")
    out_vis = os.path.join(depth_dir, "da2_intraop_preview.png")
    quality_path = os.path.join(depth_dir, "da2_intraop_quality.json")

    print("=== 07. Depth Anything on intra-op lap_undist ===")
    print(f"  frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")

    lap = cv2.imread(lap_path, cv2.IMREAD_COLOR)
    if lap is None:
        raise FileNotFoundError(f"Run scripts/01_prepare_frame_data_and_models.py first. Missing: {lap_path}")

    full_liver_mask, depth_valid_mask, mask_quality = validate_case_masks(
        case_dir,
        expected_shape=lap.shape[:2],
    )
    print(f"[OK] masks: {quality_summary(mask_quality)}")

    estimator = get_estimator()
    raw_depth = estimator.infer(lap).astype(np.float32)
    if raw_depth.shape != lap.shape[:2]:
        raise RuntimeError(
            f"DA2 depth shape mismatch: got {raw_depth.shape}, expected {lap.shape[:2]}."
        )

    masked_depth = raw_depth.copy()
    masked_depth[depth_valid_mask <= 0] = 0.0
    depth_stats = depth_quality(raw_depth, masked_depth, full_liver_mask, depth_valid_mask)
    validate_depth_quality(depth_stats)

    save_raw_depth(raw_depth, raw_npy)
    save_depth_preview(raw_depth, raw_vis)
    save_raw_depth(masked_depth, out_npy)
    save_depth_preview(masked_depth, out_vis, mask=depth_valid_mask > 0)
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mask_quality": mask_quality,
                "mask_fingerprints": mask_fingerprints(full_liver_mask, depth_valid_mask),
                "depth_quality": depth_stats,
                "semantics": {
                    "da2_intraop_raw.npy": "Unmasked raw DA2 output retained for audit only.",
                    "da2_intraop.npy": (
                        "DA2 depth with all pixels outside the manual outer liver mask set to zero."
                    ),
                },
            },
            f,
            indent=2,
        )

    print(
        f"[OK] audit raw depth: {raw_npy}  shape={raw_depth.shape}  "
        f"range=[{raw_depth.min():.4f}, {raw_depth.max():.4f}]"
    )
    print(
        "[OK] inference depth: %s  valid/image=%.4f valid/full=%.4f valid/outer_mask=%.4f"
        % (
            out_npy,
            depth_stats["masked_depth_valid_ratio"],
            depth_stats["masked_depth_valid_to_full_ratio"],
            depth_stats["masked_depth_valid_to_depth_valid_mask_ratio"],
        )
    )
    print(f"[OK] quality report: {quality_path}")


if __name__ == "__main__":
    main()
