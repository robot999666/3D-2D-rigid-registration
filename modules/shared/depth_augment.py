"""深度归一化与训练增强。"""

import os

import cv2
import numpy as np


def depth_for_network_eval(depth_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """将有效深度确定性映射到网络输入范围。"""
    out = depth_raw.astype(np.float32).copy()
    mask_bool = mask > 0
    valid = np.isfinite(out) & (out > 0) & mask_bool
    out[~mask_bool] = 0.0
    if not np.any(valid):
        out[~valid] = 0.0
        return out

    z = out[valid]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin <= 1e-6:
        out[valid] = 0.0
        out[~valid] = 0.0
        return out

    out[valid] = (z - zmin) / (zmax - zmin) * 255.0
    out[~valid] = 0.0
    return out.astype(np.float32)


def rotated_rect_occlusion(depth: np.ndarray, ref_width: float = 1920.0) -> np.ndarray:
    h, w = depth.shape
    occ = np.zeros((h, w), dtype=np.uint8)
    n = np.random.choice([1, 2])
    scale = w / ref_width
    min_len = max(1.0, 100.0 * scale)
    max_len = max(min_len + 1.0, 400.0 * scale)
    min_width = max(1.0, 8.0 * scale)
    max_width = max(min_width + 1.0, 25.0 * scale)
    for _ in range(n):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        length = np.random.uniform(min_len, max_len)
        width = np.random.uniform(min_width, max_width)
        angle = np.random.uniform(-45, 45)
        rect = ((cx, cy), (length, width), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
        cv2.fillPoly(occ, [box], 255)
    out = depth.copy()
    out[occ > 128] = 0
    return out


def random_erase_depth(depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    out = depth.copy()
    target_ratio = np.random.uniform(0.05, 0.25)
    erased = 0
    max_area = h * w * target_ratio
    n = np.random.choice([0, 1, 2])
    for _ in range(n):
        bw = np.random.randint(max(1, w // 20), max(2, w // 8))
        bh = np.random.randint(max(1, h // 20), max(2, h // 8))
        if erased + bw * bh > max_area:
            continue
        x = np.random.randint(0, max(1, w - bw))
        y = np.random.randint(0, max(1, h - bh))
        out[y : y + bh, x : x + bw] = 0
        erased += bw * bh
    return out


def add_depth_noise(depth: np.ndarray) -> np.ndarray:
    out = depth.copy().astype(np.float32)
    valid = out > 0
    if not np.any(valid):
        return out
    sigma = np.random.uniform(0.005, 0.025) * 255.0
    noise = np.random.normal(0, sigma, size=out[valid].shape).astype(np.float32)
    out[valid] = np.clip(out[valid] + noise, 0, 255)
    return out


def _env_prob(name: str, default: float) -> float:
    return float(np.clip(float(os.environ.get(name, str(default))), 0.0, 1.0))


def augment_depth(depth_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """归一化后应用训练用遮挡与噪声增强。"""
    out = depth_for_network_eval(depth_raw, mask)
    if np.random.rand() < _env_prob("AR_DEPTH_OCC_PROB", 0.4):
        out = rotated_rect_occlusion(out)
    if np.random.rand() < _env_prob("AR_DEPTH_ERASE_PROB", 0.4):
        out = random_erase_depth(out)
    if np.random.rand() < _env_prob("AR_DEPTH_NOISE_PROB", 0.6):
        out = add_depth_noise(out)
    out[mask <= 0] = 0.0
    return out.astype(np.float32)
