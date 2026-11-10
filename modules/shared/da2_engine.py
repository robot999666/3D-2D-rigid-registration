"""加载 Depth Anything V2，生成并保存原始深度。"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np

from shared import case_config

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_estimator: Optional["DepthAnythingEstimator"] = None


def _ensure_da2_on_path():
    root = case_config.DA2_PROJECT_DIR
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Depth Anything V2 project not found: {root}\n"
            "Set env DA2_PROJECT_DIR to the directory containing depth_anything_v2."
        )
    if root not in sys.path:
        sys.path.insert(0, root)


def missing_checkpoint_message(encoder, path):
    model = {"vits": "Small", "vitb": "Base", "vitl": "Large"}.get(encoder)
    url = (f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model}/resolve/main/"
           f"depth_anything_v2_{encoder}.pth") if model else "https://github.com/DepthAnything/Depth-Anything-V2"
    return f"缺少 Depth Anything V2 权重，请提前下载。\n下载：{url}\n保存到：{path}"


def resolve_checkpoint(encoder: Optional[str] = None) -> str:
    enc = encoder or case_config.DA2_ENCODER
    root = case_config.DA2_PROJECT_DIR
    for path in (
        case_config.DA2_CHECKPOINT_PATH,
        os.path.join(case_config.WEIGHTS_ROOT, f"depth_anything_v2_{enc}.pth"),
        os.path.join(root, f"depth_anything_v2_{enc}.pth"),
        os.path.join(root, "checkpoints", f"depth_anything_v2_{enc}.pth"),
    ):
        if os.path.isfile(path):
            return path
    expected = os.path.join(case_config.WEIGHTS_ROOT, f"depth_anything_v2_{enc}.pth")
    raise FileNotFoundError(missing_checkpoint_message(enc, expected))


class DepthAnythingEstimator:
    def __init__(
        self,
        encoder: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        input_size: Optional[int] = None,
    ):
        _ensure_da2_on_path()
        self.encoder = encoder or case_config.DA2_ENCODER
        if self.encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder: {self.encoder}")

        ckpt = checkpoint_path or resolve_checkpoint(self.encoder)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(missing_checkpoint_message(self.encoder, ckpt))
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        self.input_size = input_size or case_config.DA2_INPUT_SIZE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DepthAnythingV2(**_MODEL_CONFIGS[self.encoder])
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            depth = self.model.infer_image(image_bgr, self.input_size)
        depth = depth.astype(np.float32)
        if depth.shape[:2] != image_bgr.shape[:2]:
            depth = cv2.resize(
                depth,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth


def get_estimator(**kwargs) -> DepthAnythingEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthAnythingEstimator(**kwargs)
    return _estimator


def save_raw_depth(depth: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, depth.astype(np.float32))


def save_depth_preview(depth: np.ndarray, path: str, mask: Optional[np.ndarray] = None):
    """按有效区域生成深度预览图。"""
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid]
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin > 1e-6:
            vis[valid] = ((depth[valid] - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    if mask is not None:
        vis_color[~np.asarray(mask, dtype=bool)] = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, vis_color)
