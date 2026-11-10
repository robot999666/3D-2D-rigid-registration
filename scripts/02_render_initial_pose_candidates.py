"""从预处理病例渲染初始位姿候选，保存到 preprocess/tview_candidates。"""

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
from shared.pose_render import PoseSampleRenderer, euler_to_matrix


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


def normalize_pose_key(key):
    return key.strip().upper()


def load_pose_params(case_dir):
    params = dict(DEFAULT_POSE)
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


def as_uint8_maps(maps):
    arr = np.asarray(maps)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_tint(image, mask, color, alpha):
    out = image.copy()
    sel = mask > 0
    if np.any(sel):
        out[sel] = ((1.0 - alpha) * out[sel] + alpha * np.array(color)).astype(np.uint8)
    return out


def overlay_debug(image, gt_maps, render_maps, render_mask, candidate_id, params, metrics):
    out = image.copy()
    out = apply_tint(out, render_mask, (70, 70, 170), 0.18)

    gt_colors = [(0, 180, 0), (180, 0, 0), (0, 180, 180)]
    rd_colors = [(0, 0, 255), (255, 0, 255), (255, 255, 0)]
    for c in range(3):
        out = apply_tint(out, gt_maps[c], gt_colors[c], 0.45)
        out = apply_tint(out, render_maps[c], rd_colors[c], 0.65)

    lines = [
        candidate_id,
        "TX/TY/TZ %.1f %.1f %.1f" % (params["TX"], params["TY"], params["TZ"]),
        "RX/RY/RZ %.1f %.1f %.1f" % (params["RX"], params["RY"], params["RZ"]),
        "mask %.3f  ridge_out %.3f  ligament_out %.3f"
        % (
            metrics["render_mask_ratio"],
            metrics["ridge_outside_mask_ratio"],
            metrics["ligament_outside_mask_ratio"],
        ),
    ]
    y = 42
    for line in lines:
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        y += 36
    return out


def candidate_specs():
    return [
        ("candidate_00_base", {}),
        ("candidate_01_tx_m10", {"TX": -10.0}),
        ("candidate_02_tx_p10", {"TX": 10.0}),
        ("candidate_03_ty_m10", {"TY": -10.0}),
        ("candidate_04_ty_p10", {"TY": 10.0}),
        ("candidate_05_tz_m15", {"TZ": -15.0}),
        ("candidate_06_tz_p15", {"TZ": 15.0}),
        ("candidate_07_rx_m10", {"RX": -10.0}),
        ("candidate_08_rx_p10", {"RX": 10.0}),
        ("candidate_09_rz_m10", {"RZ": -10.0}),
        ("candidate_10_rz_p10", {"RZ": 10.0}),
    ]


def add_delta(base_params, delta):
    params = dict(base_params)
    for key, value in delta.items():
        params[key] = float(params[key] + value)
    return params


def contour_outside_ratio(contour, mask, dilation_px):
    count = int(np.count_nonzero(contour))
    if count == 0:
        return 1.0
    ksize = max(3, int(dilation_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    expanded = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1) > 0
    outside = np.count_nonzero((contour > 0) & ~expanded)
    return float(outside / count)


def make_contact_sheet(thumbnails, tile_width):
    if not thumbnails:
        return None
    resized = []
    for image in thumbnails:
        h, w = image.shape[:2]
        scale = tile_width / float(w)
        tile = cv2.resize(image, (tile_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        resized.append(tile)
    cols = min(3, len(resized))
    rows = int(np.ceil(len(resized) / cols))
    tile_h = max(tile.shape[0] for tile in resized)
    sheet = np.zeros((rows * tile_h, cols * tile_width, 3), dtype=np.uint8)
    for i, tile in enumerate(resized):
        r, c = divmod(i, cols)
        y = r * tile_h
        x = c * tile_width
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet


def main():
    case_dir = case_config.case_dir()
    output_dir = os.path.join(case_config.result_dir(), "preprocess", "tview_candidates")
    ensure_dir(output_dir)

    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    gt_path = os.path.join(case_dir, "contours", "gt_multicontour.npy")
    camera_path = os.path.join(case_dir, "camera", "camera_new.txt")
    tview_seed_path = os.path.join(output_dir, "_candidate_seed_tview.txt")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Run scripts/01_prepare_frame_data_and_models.py first. Missing: {image_path}")
    height, width = image.shape[:2]
    gt_maps = as_uint8_maps(np.load(gt_path))

    base_params, sources, config_path = load_pose_params(case_dir)
    np.savetxt(tview_seed_path, build_transform(base_params), fmt="%.10f")
    renderer = PoseSampleRenderer(
        os.path.join(case_dir, "models", "Liver_obj.ply"),
        os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
        os.path.join(case_dir, "contours", "model_contours.json"),
        camera_path,
        tview_seed_path,
        width,
        height,
        contour_thickness=case_config.CONTOUR_THICKNESS,
    )

    max_outside = float(os.environ.get("AR_CANDIDATE_INTERNAL_MAX_OUTSIDE_RATIO", "0.25"))
    mask_dilation_px = int(os.environ.get("AR_CANDIDATE_MASK_DILATION_PX", "12"))
    records = []
    thumbnails = []

    for candidate_id, delta in candidate_specs():
        params = add_delta(base_params, delta)
        T_view = build_transform(params)
        renderer.T_view = T_view.astype(np.float32)
        render_maps, render_mask, _, _, visible, _ = renderer.render(np.eye(4, dtype=np.float32))
        metrics = {
            "render_mask_ratio": float(np.count_nonzero(render_mask) / render_mask.size),
            "ridge_outside_mask_ratio": contour_outside_ratio(render_maps[1], render_mask, mask_dilation_px),
            "ligament_outside_mask_ratio": contour_outside_ratio(render_maps[2], render_mask, mask_dilation_px),
        }
        passed = (
            metrics["ridge_outside_mask_ratio"] <= max_outside
            and metrics["ligament_outside_mask_ratio"] <= max_outside
        )
        overlay = overlay_debug(image, gt_maps, render_maps, render_mask, candidate_id, params, metrics)
        image_name = f"{candidate_id}.png"
        image_path_out = os.path.join(output_dir, image_name)
        if not cv2.imwrite(image_path_out, overlay):
            raise OSError(f"Failed to write candidate image: {image_path_out}")
        thumbnails.append(overlay)
        records.append(
            {
                "id": candidate_id,
                "delta": delta,
                "params": params,
                "sources": sources,
                "config_path_checked": config_path,
                "T_view": T_view.tolist(),
                "image": image_path_out,
                "visible_types": visible,
                "metrics": metrics,
                "passed_internal_contour_check": bool(passed),
            }
        )

    contact_sheet = make_contact_sheet(thumbnails, int(os.environ.get("AR_CANDIDATE_TILE_WIDTH", "560")))
    contact_path = os.path.join(output_dir, "contact_sheet.jpg")
    if contact_sheet is not None and not cv2.imwrite(contact_path, contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write contact sheet: {contact_path}")

    summary = {
        "patient_id": case_config.PATIENT_ID,
        "frame_id": case_config.FRAME_ID,
        "image_shape_hw": [height, width],
        "base_params": base_params,
        "max_internal_contour_outside_mask_ratio": max_outside,
        "contact_sheet": contact_path,
        "candidates": records,
    }
    json_path = os.path.join(output_dir, "candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    failing = [r["id"] for r in records if not r["passed_internal_contour_check"]]
    print("=== 02. Render initial T_view candidates ===")
    print(f"  frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")
    print(f"  full-resolution candidate images: {output_dir}")
    print(f"  contact sheet: {contact_path}")
    print(f"  metadata: {json_path}")
    if len(failing) == len(records):
        raise RuntimeError(
            "All candidates failed the internal contour-vs-mask check. "
            "Review hybrid mesh centering before training."
        )
    if failing:
        print("[WARN] candidates failing internal contour check: " + ", ".join(failing))


if __name__ == "__main__":
    main()
