"""为渲染样本保存 DA2 原始深度；归一化与随机增强在数据加载时执行。"""

import argparse
import glob
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
from shared.da2_engine import get_estimator, save_raw_depth


def load_prefixes(split_dir):
    labels_path = os.path.join(split_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return sorted(labels.keys())

    images_dir = os.path.join(split_dir, "images")
    rgb_files = sorted(glob.glob(os.path.join(images_dir, "*_rgb.png")))
    return [os.path.basename(path).replace("_rgb.png", "") for path in rgb_files]


def depth_is_current(depth_path, rgb_path):
    if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
        return False
    return os.path.getmtime(depth_path) >= os.path.getmtime(rgb_path)


def process_split(split_dir, estimator, skip_existing: bool):
    images_dir = os.path.join(split_dir, "images")
    prefixes = load_prefixes(split_dir)
    if not prefixes:
        print(f"[WARN] no labeled samples in {split_dir}")
        return 0

    count = 0
    for prefix in tqdm(prefixes, desc=os.path.basename(split_dir)):
        rgb_path = os.path.join(images_dir, f"{prefix}_rgb.png")
        depth_path = os.path.join(images_dir, f"{prefix}_depth.npy")
        if skip_existing and depth_is_current(depth_path, rgb_path):
            continue

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[skip] bad image: {rgb_path}")
            continue

        depth = estimator.infer(bgr)
        depth_dtype = os.environ.get("AR_SAMPLE_DEPTH_DTYPE", "float32").lower()
        if depth_dtype == "float16":
            np.save(depth_path, depth.astype(np.float16))
        elif depth_dtype == "float32":
            save_raw_depth(depth, depth_path)
        else:
            raise ValueError(
                f"Unsupported AR_SAMPLE_DEPTH_DTYPE={depth_dtype!r}; use float32 or float16."
            )
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="为渲染训练样本生成原始 DA2 深度。")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="存在 *_depth.npy 时跳过该样本",
    )
    args = parser.parse_args()

    case_dir = case_config.case_dir()
    train_dir = os.path.join(case_dir, "sample", "train")
    val_dir = os.path.join(case_dir, "sample", "val")

    print("=== 06. Depth Anything on training samples ===")
    print(f"  frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")
    print("  output: raw DA2 float32, no post-processing")

    estimator = get_estimator()
    n_train = process_split(train_dir, estimator, args.skip_existing)
    n_val = process_split(val_dir, estimator, args.skip_existing)
    print(f"[OK] wrote depth: train={n_train}, val={n_val}")
    print("[Next] python scripts/07_generate_depth_for_real_frame.py")


if __name__ == "__main__":
    main()
