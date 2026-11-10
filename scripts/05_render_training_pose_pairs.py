"""渲染 A/B 位姿配对：base_to_target 用于较大修正，local_refine 用于局部修正。"""

import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.pose_render import PoseSampleRenderer, euler_to_matrix


class Config:
    """配对采样配置：平移单位 mm，旋转角单位度。"""

    PATIENT_ID = case_config.PATIENT_ID
    FRAME_ID = case_config.FRAME_ID
    CASE_ROOT = case_config.CASE_ROOT

    # 训练和验证配对数量。
    NUM_PAIRS = int(os.environ.get("AR_NUM_SAMPLES", "5000"))
    TRAIN_RATIO = float(os.environ.get("AR_TRAIN_RATIO", "0.9"))
    SEED = case_config.SEED

    # 目标位姿的整体扰动范围。
    GLOBAL_TRANS_RANGE_MM = float(os.environ.get("AR_TRANS_RANGE_MM", "50.0"))
    GLOBAL_ROT_RANGE_DEG = float(os.environ.get("AR_ROT_RANGE_DEG", "20.0"))

    # local_refine 配对中 A 相对 B 的扰动范围。
    LOCAL_TRANS_RANGE_MM = float(os.environ.get("AR_LOCAL_TRANS_RANGE_MM", "5.0"))
    LOCAL_ROT_RANGE_DEG = float(os.environ.get("AR_LOCAL_ROT_RANGE_DEG", "3.0"))
    LOCAL_REFINE_RATIO = float(os.environ.get("AR_LOCAL_REFINE_RATIO", "0.7"))

    # 渲染可见性阈值。
    MIN_VISIBLE_TYPES = 2
    MIN_MASK_RATIO = 0.02
    MAX_MASK_RATIO = 0.95
    RENDER_CONTOUR_THICKNESS = case_config.CONTOUR_THICKNESS

    # 0 表示最多尝试 NUM_PAIRS * 50 次。
    MAX_ATTEMPTS = int(os.environ.get("AR_MAX_ATTEMPTS", "0"))

    # 全分辨率样本的存储类型。
    SAMPLE_CONTOUR_DTYPE = os.environ.get("AR_SAMPLE_CONTOUR_DTYPE", "uint8").lower()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sample_delta(trans_range_mm, rot_range_deg):
    rx, ry, rz = np.random.uniform(-rot_range_deg, rot_range_deg, 3)
    tx, ty, tz = np.random.uniform(-trans_range_mm, trans_range_mm, 3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_to_matrix(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def sample_global_delta(cfg):
    return sample_delta(cfg.GLOBAL_TRANS_RANGE_MM, cfg.GLOBAL_ROT_RANGE_DEG)


def sample_local_delta(cfg):
    return sample_delta(cfg.LOCAL_TRANS_RANGE_MM, cfg.LOCAL_ROT_RANGE_DEG)


def render_state(renderer, delta_T, total_pixels, cfg):
    contours, mask, rgb_bgr, T_render, visible, _ = renderer.render(delta_T)
    mask_ratio = np.count_nonzero(mask) / total_pixels
    valid = (
        len(visible) >= cfg.MIN_VISIBLE_TYPES
        and cfg.MIN_MASK_RATIO <= mask_ratio <= cfg.MAX_MASK_RATIO
    )
    state = {
        "contours": contours,
        "mask": mask,
        "rgb_bgr": rgb_bgr,
        "Delta_T": delta_T,
        "T_render": T_render,
        "visible_types": visible,
        "mask_ratio": float(mask_ratio),
    }
    return valid, state


def save_state(images_dir, labels, prefix, state):
    storage = labels.get(prefix, {}).get("storage", {})
    if prefix not in labels:
        contours_path = os.path.join(images_dir, f"{prefix}_contours.npy")
        if Config.SAMPLE_CONTOUR_DTYPE == "uint8":
            np.save(contours_path, state["contours"].astype(np.uint8))
            storage["contours"] = "uint8_0_255"
        elif Config.SAMPLE_CONTOUR_DTYPE == "float32":
            np.save(contours_path, state["contours"].astype(np.float32) / 255.0)
            storage["contours"] = "float32_0_1"
        else:
            raise ValueError(
                "Unsupported AR_SAMPLE_CONTOUR_DTYPE="
                f"{Config.SAMPLE_CONTOUR_DTYPE!r}; use uint8 or float32."
            )
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_mask.png"), state["mask"])
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_rgb.png"), state["rgb_bgr"])

    labels[prefix] = {
        "Delta_T": state["Delta_T"].tolist(),
        "T_render": state["T_render"].tolist(),
        "visible_types": state["visible_types"],
        "mask_ratio": state["mask_ratio"],
        "storage": storage,
    }


def save_pair(split_dir, labels, pairs, pair_id, pair_type, state_A, state_B):
    images_dir = os.path.join(split_dir, "images")
    if pair_type == "base_to_target":
        prefix_A = "base"
        prefix_B = f"{pair_id}_B"
    else:
        prefix_A = f"{pair_id}_A"
        prefix_B = f"{pair_id}_B"

    save_state(images_dir, labels, prefix_A, state_A)
    save_state(images_dir, labels, prefix_B, state_B)
    pairs[pair_id] = {
        "pair_type": pair_type,
        "A": prefix_A,
        "B": prefix_B,
        "Delta_A": labels[prefix_A]["Delta_T"],
        "Delta_B": labels[prefix_B]["Delta_T"],
    }


def contour_overlay(rgb_bgr, mask, contours):
    out = rgb_bgr.copy()
    tint = out.copy()
    tint[mask > 0] = (70, 120, 70)
    out = cv2.addWeighted(out, 0.75, tint, 0.25, 0.0)
    if contours.dtype != np.uint8:
        contours = np.clip(contours * 255.0, 0, 255).astype(np.uint8)
    colors = [(0, 255, 0), (255, 0, 255), (255, 255, 0)]
    for channel, color in enumerate(colors):
        if channel >= contours.shape[0]:
            break
        pts = contours[channel] > 0
        out[pts] = (0.35 * out[pts] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def load_preview_state(split_dir, prefix):
    images_dir = os.path.join(split_dir, "images")
    rgb = cv2.imread(os.path.join(images_dir, f"{prefix}_rgb.png"), cv2.IMREAD_COLOR)
    mask = cv2.imread(os.path.join(images_dir, f"{prefix}_mask.png"), cv2.IMREAD_GRAYSCALE)
    contours = np.load(os.path.join(images_dir, f"{prefix}_contours.npy"))
    if rgb is None or mask is None:
        raise FileNotFoundError(f"Missing preview state images for {prefix} in {images_dir}")
    return contour_overlay(rgb, mask, contours)


def make_tile(image, title, size=(360, 220)):
    tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size[0], 30), (0, 0, 0), -1)
    cv2.putText(tile, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def write_preview_sheet(split_data, max_pairs=5):
    rows = []
    for split in ("train", "val"):
        data = split_data[split]
        for pair_id, pair in data["pairs"].items():
            state_a = load_preview_state(data["dir"], pair["A"])
            state_b = load_preview_state(data["dir"], pair["B"])
            tile_a = make_tile(state_a, f"{split} {pair_id} A  {pair['pair_type']}")
            tile_b = make_tile(state_b, f"{split} {pair_id} B  target")
            rows.append(np.hstack([tile_a, tile_b]))
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            break
    if not rows:
        return None
    pad = np.full((12, rows[0].shape[1], 3), 245, dtype=np.uint8)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.vstack([sheet, pad, row])
    out_dir = os.path.join(case_config.training_dir(), "sample_preview")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "sample_preview_contact_sheet.jpg")
    cv2.imwrite(out_path, sheet)
    return out_path


def make_pair(renderer, base_state, total_pixels, cfg):
    pair_type = "local_refine" if np.random.rand() < cfg.LOCAL_REFINE_RATIO else "base_to_target"

    if pair_type == "base_to_target":
        valid_B, state_B = render_state(renderer, sample_global_delta(cfg), total_pixels, cfg)
        if not valid_B:
            return None
        return pair_type, base_state, state_B

    delta_B = sample_global_delta(cfg)
    delta_A = delta_B @ sample_local_delta(cfg)
    valid_A, state_A = render_state(renderer, delta_A, total_pixels, cfg)
    if not valid_A:
        return None
    valid_B, state_B = render_state(renderer, delta_B, total_pixels, cfg)
    if not valid_B:
        return None
    return pair_type, state_A, state_B


def main():
    cfg = Config()
    np.random.seed(cfg.SEED)
    if cfg.NUM_PAIRS < 2:
        raise ValueError("AR_NUM_SAMPLES must be at least 2 so train and val are both non-empty.")
    train_pair_count = int(round(cfg.NUM_PAIRS * cfg.TRAIN_RATIO))
    train_pair_count = min(max(train_pair_count, 1), cfg.NUM_PAIRS - 1)
    split_plan = np.array(
        ["train"] * train_pair_count + ["val"] * (cfg.NUM_PAIRS - train_pair_count),
        dtype=object,
    )
    np.random.default_rng(cfg.SEED).shuffle(split_plan)

    case_dir = case_config.case_dir(cfg.CASE_ROOT)
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    height, width = img.shape[:2]
    total_pixels = width * height

    renderer = PoseSampleRenderer(
        os.path.join(case_dir, "models", "Liver_obj.ply"),
        os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
        os.path.join(case_dir, "contours", "model_contours.json"),
        os.path.join(case_dir, "camera", "camera_new.txt"),
        os.path.join(case_dir, "T_view.txt"),
        width,
        height,
        contour_thickness=cfg.RENDER_CONTOUR_THICKNESS,
    )

    train_dir = os.path.join(case_dir, "sample", "train")
    val_dir = os.path.join(case_dir, "sample", "val")
    for split_dir in (train_dir, val_dir):
        ensure_dir(os.path.join(split_dir, "images"))

    valid_base, base_state = render_state(renderer, np.eye(4, dtype=np.float32), total_pixels, cfg)
    if not valid_base:
        raise RuntimeError("Base T_view pose is not visible enough for training pairs.")

    print("=== 05. Render explicit A/B pose pairs (RGB for DA2, no depth model) ===")
    print(f"  frame: {cfg.PATIENT_ID}/{cfg.FRAME_ID}")
    print(f"  seed: {cfg.SEED}")
    print(f"  pairs: {cfg.NUM_PAIRS}")
    print(
        "  pair mix: base_to_target=%.0f%%, local_refine=%.0f%%"
        % ((1.0 - cfg.LOCAL_REFINE_RATIO) * 100.0, cfg.LOCAL_REFINE_RATIO * 100.0)
    )
    print(
        "  local refine range: +/-%.1f mm, +/-%.1f deg"
        % (cfg.LOCAL_TRANS_RANGE_MM, cfg.LOCAL_ROT_RANGE_DEG)
    )

    split_data = {
        "train": {"dir": train_dir, "labels": {}, "pairs": {}},
        "val": {"dir": val_dir, "labels": {}, "pairs": {}},
    }
    max_attempts = cfg.MAX_ATTEMPTS if cfg.MAX_ATTEMPTS > 0 else cfg.NUM_PAIRS * 50
    accepted = 0
    rejected = 0
    attempts = 0
    pbar = tqdm(total=cfg.NUM_PAIRS, desc="Rendering pairs")

    while accepted < cfg.NUM_PAIRS and attempts < max_attempts:
        attempts += 1
        result = make_pair(renderer, base_state, total_pixels, cfg)
        if result is None:
            rejected += 1
            continue

        pair_type, state_A, state_B = result
        split = str(split_plan[accepted])
        pair_id = f"pair_{accepted:05d}"
        data = split_data[split]
        save_pair(data["dir"], data["labels"], data["pairs"], pair_id, pair_type, state_A, state_B)
        accepted += 1
        pbar.update(1)

    pbar.close()
    if accepted < cfg.NUM_PAIRS:
        raise RuntimeError(
            f"Only accepted {accepted}/{cfg.NUM_PAIRS} pairs after {attempts} attempts. "
            "Relax visibility filters or raise AR_MAX_ATTEMPTS."
        )

    for split, data in split_data.items():
        with open(os.path.join(data["dir"], "labels.json"), "w", encoding="utf-8") as f:
            json.dump(data["labels"], f, indent=2)
        with open(os.path.join(data["dir"], "pairs.json"), "w", encoding="utf-8") as f:
            json.dump(data["pairs"], f, indent=2)

    preview_pairs = int(os.environ.get("AR_SAMPLE_PREVIEW_PAIRS", "5"))
    preview_path = write_preview_sheet(split_data, max_pairs=max(1, preview_pairs))

    print(f"[OK] accepted pairs: {accepted}, rejected attempts: {rejected}")
    for split, data in split_data.items():
        print(f"[OK] {split}: states={len(data['labels'])}, pairs={len(data['pairs'])}")
    if preview_path:
        print(f"[OK] sample preview: {preview_path}")
    print("[Next] python scripts/06_generate_depth_for_training_samples.py")


if __name__ == "__main__":
    main()
