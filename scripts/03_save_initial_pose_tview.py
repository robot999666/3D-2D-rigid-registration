"""保存物体到相机的初始位姿 T_view 及其参数来源。"""

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config as _cc
from shared.pose_render import PoseSampleRenderer, euler_to_matrix


class Config:
    """初始位姿：平移单位 mm，Euler 角单位度，旋转顺序 Rz @ Ry @ Rx。

    参数优先级为 AR_TVIEW_*、initial_pose.json、默认值。"""

    PATIENT_ID = _cc.PATIENT_ID
    FRAME_ID = _cc.FRAME_ID
    CASE_ROOT = _cc.CASE_ROOT

    DEBUG_DIR = os.path.join(_cc.result_dir(), "preprocess")
    RENDER_CONTOUR_THICKNESS = _cc.CONTOUR_THICKNESS

    # 未提供 JSON 或环境变量时使用的初值。
    DEFAULT_POSE = {
        "TX": 6.0,
        "TY": 27.0,
        "TZ": 110.0,
        "RX": 45.0,
        "RY": -10.0,
        "RZ": 50.0,
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def overlay_debug(image, gt_maps, render_maps):
    out = image.copy()
    gt_colors = [(0, 180, 0), (180, 0, 0), (0, 180, 180)]
    rd_colors = [(0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for c in range(3):
        gt = gt_maps[c] > 0
        rd = render_maps[c] > 0
        out[gt] = (0.55 * out[gt] + 0.45 * np.array(gt_colors[c])).astype(np.uint8)
        out[rd] = (0.55 * out[rd] + 0.45 * np.array(rd_colors[c])).astype(np.uint8)
    return out


def normalize_pose_key(key):
    return key.strip().upper()


def load_pose_params(case_dir, cfg):
    params = dict(cfg.DEFAULT_POSE)
    sources = {key: "default" for key in params}

    config_path = os.environ.get("AR_TVIEW_CONFIG", os.path.join(case_dir, "initial_pose.json"))
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, value in raw.items():
            norm_key = normalize_pose_key(key)
            if norm_key in params:
                params[norm_key] = float(value)
                sources[norm_key] = f"config:{config_path}"

    env_names = {
        "TX": "AR_TVIEW_TX",
        "TY": "AR_TVIEW_TY",
        "TZ": "AR_TVIEW_TZ",
        "RX": "AR_TVIEW_RX",
        "RY": "AR_TVIEW_RY",
        "RZ": "AR_TVIEW_RZ",
    }
    for key, env_name in env_names.items():
        if env_name in os.environ:
            params[key] = float(os.environ[env_name])
            sources[key] = f"env:{env_name}"

    return params, sources, config_path


def build_transform(params):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_to_matrix(params["RX"], params["RY"], params["RZ"])
    T[:3, 3] = [params["TX"], params["TY"], params["TZ"]]
    return T


def save_pose_metadata(path, params, sources, config_path, T_view):
    metadata = {
        "patient_id": Config.PATIENT_ID,
        "frame_id": Config.FRAME_ID,
        "config_path_checked": config_path,
        "params": params,
        "sources": sources,
        "T_view": T_view.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    cfg = Config()
    case_dir = os.path.join(cfg.CASE_ROOT, cfg.PATIENT_ID, cfg.FRAME_ID)
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    gt_path = os.path.join(case_dir, "contours", "gt_multicontour.npy")
    tview_path = os.path.join(case_dir, "T_view.txt")
    meta_path = os.path.join(case_dir, "T_view_meta.json")

    print("=== 03. Save selected initial T_view ===")
    print(f"  frame: {cfg.PATIENT_ID}/{cfg.FRAME_ID}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    gt_maps = (np.load(gt_path) * 255).astype(np.uint8)

    params, sources, config_path = load_pose_params(case_dir, cfg)
    T_view = build_transform(params)
    np.savetxt(tview_path, T_view, fmt="%.10f")
    save_pose_metadata(meta_path, params, sources, config_path, T_view)

    renderer = PoseSampleRenderer(
        os.path.join(case_dir, "models", "Liver_obj.ply"),
        os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
        os.path.join(case_dir, "contours", "model_contours.json"),
        os.path.join(case_dir, "camera", "camera_new.txt"),
        tview_path,
        width,
        height,
        contour_thickness=cfg.RENDER_CONTOUR_THICKNESS,
    )
    render_maps, _, _, _, _, _ = renderer.render(np.eye(4, dtype=np.float32))

    ensure_dir(cfg.DEBUG_DIR)
    debug = overlay_debug(image, gt_maps, render_maps)
    debug_path = os.path.join(cfg.DEBUG_DIR, f"{cfg.PATIENT_ID}_{cfg.FRAME_ID}_tview_debug.png")
    cv2.imwrite(debug_path, debug)

    print(f"[OK] T_view: {tview_path}")
    print(f"[OK] metadata: {meta_path}")
    print(f"[OK] debug: {debug_path}")


if __name__ == "__main__":
    main()
