"""保存和校验人工肝脏掩码、质量记录及文件指纹。"""

import hashlib
import json
import os
from datetime import datetime

import cv2
import numpy as np


FULL_MASK_NAME = "full_liver_mask.png"
DEPTH_VALID_MASK_NAME = "depth_valid_mask.png"
OCCLUSION_MASK_NAME = "occlusion_mask.png"
ANNOTATION_NAME = "mask_annotation.json"
QUALITY_NAME = "mask_quality.json"
OVERLAY_NAME = "mask_quality_overlay.png"

DEFAULT_FULL_MASK_MIN_RATIO = float(os.environ.get("AR_FULL_MASK_MIN_RATIO", "0.05"))
DEFAULT_FULL_MASK_MAX_RATIO = float(os.environ.get("AR_FULL_MASK_MAX_RATIO", "0.95"))
DEFAULT_FULL_MASK_LARGEST_COMPONENT_MIN = float(
    os.environ.get("AR_FULL_MASK_LARGEST_COMPONENT_MIN", "0.95")
)
DEFAULT_DEPTH_VALID_TO_FULL_MIN = float(os.environ.get("AR_DEPTH_VALID_TO_FULL_MIN", "1.00"))
DEFAULT_DEPTH_COVERAGE_MIN = float(os.environ.get("AR_DEPTH_COVERAGE_MIN", "0.95"))


class MaskValidationError(ValueError):
    """掩码或深度质量校验失败。"""


def mask_dir(case_dir):
    return os.path.join(case_dir, "masks")


def mask_paths(case_dir):
    root = mask_dir(case_dir)
    return {
        "root": root,
        "full_liver_mask": os.path.join(root, FULL_MASK_NAME),
        "depth_valid_mask": os.path.join(root, DEPTH_VALID_MASK_NAME),
        "occlusion_mask": os.path.join(root, OCCLUSION_MASK_NAME),
        "annotation": os.path.join(root, ANNOTATION_NAME),
        "quality": os.path.join(root, QUALITY_NAME),
        "overlay": os.path.join(root, OVERLAY_NAME),
    }


def _binary_uint8(mask):
    if mask is None:
        raise MaskValidationError("Mask is None.")
    if mask.ndim != 2:
        raise MaskValidationError(f"Mask must be single-channel, got shape={mask.shape}.")
    return (mask > 0).astype(np.uint8) * 255


def mask_fingerprints(full_liver_mask, depth_valid_mask):
    full = np.ascontiguousarray(_binary_uint8(full_liver_mask))
    depth_valid = np.ascontiguousarray(_binary_uint8(depth_valid_mask))
    return {
        "full_liver_mask_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
        "depth_valid_mask_sha256": hashlib.sha256(depth_valid.tobytes()).hexdigest(),
    }


def _load_binary(path, expected_shape=None):
    if not os.path.exists(path):
        raise MaskValidationError(
            f"Required mask is missing: {path}. "
            "Run scripts/04_annotate_full_liver_outer_mask.py for this frame."
        )
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise MaskValidationError(f"Cannot read required mask: {path}")
    if expected_shape is not None and tuple(raw.shape) != tuple(expected_shape):
        raise MaskValidationError(
            f"Mask shape mismatch for {path}: got {raw.shape}, expected {tuple(expected_shape)}. "
            "Re-annotate masks after preprocessing."
        )
    return _binary_uint8(raw)


def _component_metrics(mask):
    binary = (mask > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(x) for x in stats[1:, cv2.CC_STAT_AREA].tolist()] if count > 1 else []
    foreground = int(np.count_nonzero(binary))
    largest = max(areas) if areas else 0
    return {
        "component_count": int(max(0, count - 1)),
        "component_areas_px": sorted(areas, reverse=True),
        "largest_component_px": int(largest),
        "largest_component_fraction": float(largest / foreground) if foreground else 0.0,
    }


def compute_mask_quality(full_liver_mask, depth_valid_mask):
    full = _binary_uint8(full_liver_mask)
    depth_valid = _binary_uint8(depth_valid_mask)
    if full.shape != depth_valid.shape:
        raise MaskValidationError(
            f"full_liver_mask and depth_valid_mask shapes differ: {full.shape} vs {depth_valid.shape}."
        )

    full_bool = full > 0
    depth_bool = depth_valid > 0
    pixels = int(full.size)
    full_pixels = int(np.count_nonzero(full_bool))
    depth_pixels = int(np.count_nonzero(depth_bool))
    outside_pixels = int(np.count_nonzero(depth_bool & ~full_bool))
    occlusion_pixels = int(np.count_nonzero(full_bool & ~depth_bool))

    return {
        "shape_hw": [int(full.shape[0]), int(full.shape[1])],
        "image_pixels": pixels,
        "full_liver_pixels": full_pixels,
        "full_liver_ratio": float(full_pixels / pixels),
        "depth_valid_pixels": depth_pixels,
        "depth_valid_ratio": float(depth_pixels / pixels),
        "depth_valid_to_full_ratio": float(depth_pixels / full_pixels) if full_pixels else 0.0,
        "occlusion_pixels": occlusion_pixels,
        "occlusion_to_full_ratio": float(occlusion_pixels / full_pixels) if full_pixels else 0.0,
        "depth_valid_outside_full_pixels": outside_pixels,
        "full_liver_components": _component_metrics(full),
        "depth_valid_components": _component_metrics(depth_valid),
    }


def validate_mask_arrays(
    full_liver_mask,
    depth_valid_mask,
    full_mask_min_ratio=DEFAULT_FULL_MASK_MIN_RATIO,
    full_mask_max_ratio=DEFAULT_FULL_MASK_MAX_RATIO,
    largest_component_min=DEFAULT_FULL_MASK_LARGEST_COMPONENT_MIN,
    depth_valid_to_full_min=DEFAULT_DEPTH_VALID_TO_FULL_MIN,
):
    full = _binary_uint8(full_liver_mask)
    depth_valid = _binary_uint8(depth_valid_mask)
    quality = compute_mask_quality(full, depth_valid)
    errors = []

    full_ratio = quality["full_liver_ratio"]
    if not full_mask_min_ratio <= full_ratio <= full_mask_max_ratio:
        errors.append(
            f"full_liver_mask area ratio {full_ratio:.4f} is outside "
            f"[{full_mask_min_ratio:.4f}, {full_mask_max_ratio:.4f}]"
        )
    if quality["full_liver_components"]["component_count"] < 1:
        errors.append("full_liver_mask is empty")
    if quality["full_liver_components"]["largest_component_fraction"] < largest_component_min:
        errors.append(
            "full_liver_mask is fragmented: largest component fraction "
            f"{quality['full_liver_components']['largest_component_fraction']:.4f} "
            f"< {largest_component_min:.4f}"
        )
    if quality["depth_valid_outside_full_pixels"] > 0:
        errors.append(
            f"depth_valid_mask has {quality['depth_valid_outside_full_pixels']} "
            "pixels outside full_liver_mask"
        )
    if not np.array_equal(full > 0, depth_valid > 0):
        errors.append(
            "depth_valid_mask must equal full_liver_mask under the single outer-mask policy"
        )
    if quality["depth_valid_to_full_ratio"] < depth_valid_to_full_min:
        errors.append(
            "depth_valid_mask retains too little liver area: "
            f"{quality['depth_valid_to_full_ratio']:.4f} < {depth_valid_to_full_min:.4f}"
        )

    quality["thresholds"] = {
        "full_mask_min_ratio": float(full_mask_min_ratio),
        "full_mask_max_ratio": float(full_mask_max_ratio),
        "largest_component_min": float(largest_component_min),
        "depth_valid_to_full_min": float(depth_valid_to_full_min),
    }
    quality["validation_errors"] = errors
    quality["valid"] = not errors
    if errors:
        raise MaskValidationError("Invalid intra-operative masks: " + "; ".join(errors))
    return full, depth_valid, quality


def validate_case_masks(case_dir, expected_shape=None):
    paths = mask_paths(case_dir)
    full = _load_binary(paths["full_liver_mask"], expected_shape)
    depth_valid = _load_binary(paths["depth_valid_mask"], expected_shape)
    full, depth_valid, quality = validate_mask_arrays(full, depth_valid)

    for name in ("annotation", "quality", "overlay"):
        if not os.path.exists(paths[name]):
            raise MaskValidationError(
                f"Required mask QA artifact is missing: {paths[name]}. "
                "Re-save this frame with scripts/04_annotate_full_liver_outer_mask.py."
            )
    try:
        with open(paths["annotation"], "r", encoding="utf-8") as f:
            saved_annotation = json.load(f)
        with open(paths["quality"], "r", encoding="utf-8") as f:
            saved_quality = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaskValidationError(f"Cannot read mask QA metadata: {exc}") from exc

    expected_fingerprints = mask_fingerprints(full, depth_valid)
    if (
        saved_annotation.get("mask_fingerprints") != expected_fingerprints
        or saved_quality.get("mask_fingerprints") != expected_fingerprints
    ):
        raise MaskValidationError(
            "Mask files no longer match mask annotation/quality metadata. "
            "Do not edit mask PNGs manually; re-save them with "
            "scripts/04_annotate_full_liver_outer_mask.py."
        )
    overlay = cv2.imread(paths["overlay"], cv2.IMREAD_COLOR)
    if overlay is None or overlay.shape[:2] != full.shape:
        raise MaskValidationError(
            f"Mask QA overlay is unreadable or has the wrong shape: {paths['overlay']}"
        )
    quality["mask_fingerprints"] = expected_fingerprints
    return full, depth_valid, quality


def make_quality_overlay(image_bgr, full_liver_mask, depth_valid_mask):
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    full = _binary_uint8(full_liver_mask)
    depth_valid = _binary_uint8(depth_valid_mask)
    if image_bgr.shape[:2] != full.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_bgr.shape[:2]} vs {full.shape}")

    full_bool = full > 0
    valid_bool = depth_valid > 0

    overlay = image_bgr.copy()
    tint = overlay.copy()
    tint[valid_bool] = (70, 150, 70)
    overlay = cv2.addWeighted(overlay, 0.70, tint, 0.30, 0.0)

    full_contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, full_contours, -1, (0, 255, 0), 2)
    return overlay


def _write_image_checked(path, image):
    if not cv2.imwrite(path, image):
        raise OSError(f"Failed to write image: {path}")


def save_mask_bundle(case_dir, image_bgr, full_liver_mask, depth_valid_mask, annotation):
    full, depth_valid, quality = validate_mask_arrays(full_liver_mask, depth_valid_mask)
    fingerprints = mask_fingerprints(full, depth_valid)
    quality["mask_fingerprints"] = fingerprints
    paths = mask_paths(case_dir)
    os.makedirs(paths["root"], exist_ok=True)

    occlusion = np.zeros_like(full, dtype=np.uint8)
    overlay = make_quality_overlay(image_bgr, full, depth_valid)
    _write_image_checked(paths["full_liver_mask"], full)
    _write_image_checked(paths["depth_valid_mask"], depth_valid)
    _write_image_checked(paths["occlusion_mask"], occlusion)
    _write_image_checked(paths["overlay"], overlay)

    annotation_out = dict(annotation or {})
    annotation_out.update(
        {
            "format_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mask_fingerprints": fingerprints,
            "mask_semantics": {
                "full_liver_mask": "Complete manually annotated outer liver region.",
                "depth_valid_mask": "Same pixels as full_liver_mask; instruments are not separately removed.",
                "occlusion_mask": "Always empty under the single outer-mask policy; retained for path compatibility.",
            },
        }
    )
    with open(paths["annotation"], "w", encoding="utf-8") as f:
        json.dump(annotation_out, f, indent=2)
    with open(paths["quality"], "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)
    return paths, quality


def depth_quality(raw_depth, masked_depth, full_liver_mask, depth_valid_mask):
    raw = np.asarray(raw_depth)
    masked = np.asarray(masked_depth)
    full = _binary_uint8(full_liver_mask) > 0
    valid_mask = _binary_uint8(depth_valid_mask) > 0
    if raw.shape != full.shape or masked.shape != full.shape:
        raise ValueError(
            f"Depth/mask shape mismatch: raw={raw.shape}, masked={masked.shape}, mask={full.shape}"
        )
    raw_valid = np.isfinite(raw) & (raw > 0)
    masked_valid = np.isfinite(masked) & (masked > 0)
    return {
        "shape_hw": [int(raw.shape[0]), int(raw.shape[1])],
        "raw_depth_valid_ratio": float(np.count_nonzero(raw_valid) / raw.size),
        "masked_depth_valid_ratio": float(np.count_nonzero(masked_valid) / raw.size),
        "masked_depth_valid_to_full_ratio": float(
            np.count_nonzero(masked_valid & full) / max(1, np.count_nonzero(full))
        ),
        "masked_depth_valid_to_depth_valid_mask_ratio": float(
            np.count_nonzero(masked_valid & valid_mask) / max(1, np.count_nonzero(valid_mask))
        ),
        "nonzero_outside_depth_valid_mask": int(np.count_nonzero(masked_valid & ~valid_mask)),
    }


def validate_depth_quality(stats, min_depth_coverage=DEFAULT_DEPTH_COVERAGE_MIN):
    errors = []
    if stats["nonzero_outside_depth_valid_mask"] != 0:
        errors.append(
            "masked depth has "
            f"{stats['nonzero_outside_depth_valid_mask']} non-zero pixels outside depth_valid_mask"
        )
    coverage = stats["masked_depth_valid_to_depth_valid_mask_ratio"]
    if coverage < min_depth_coverage:
        errors.append(
            f"depth coverage inside depth_valid_mask {coverage:.4f} < {min_depth_coverage:.4f}"
        )
    stats["depth_coverage_min"] = float(min_depth_coverage)
    stats["validation_errors"] = errors
    stats["valid"] = not errors
    if errors:
        raise MaskValidationError("Invalid intra-operative depth: " + "; ".join(errors))
    return stats


def quality_summary(quality):
    return (
        f"full_ratio={quality['full_liver_ratio']:.4f}, "
        f"depth_valid_ratio={quality['depth_valid_ratio']:.4f}, "
        f"depth_valid/full={quality['depth_valid_to_full_ratio']:.4f}, "
        f"occlusion/full={quality['occlusion_to_full_ratio']:.4f}, "
        f"full_components={quality['full_liver_components']['component_count']}"
    )
