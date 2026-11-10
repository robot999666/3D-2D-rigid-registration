"""迭代更新刚性位姿并导出 STL。

默认输出最后一个通过检查的位姿；轮廓和 mask 分数用于诊断。"""

import csv
import json
import logging
import os
import shutil
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.depth_augment import depth_for_network_eval
from shared.intraop_masks import (
    depth_quality,
    mask_fingerprints,
    quality_summary,
    validate_case_masks,
    validate_depth_quality,
)


class Config:
    """推理配置；图像分数控制项由 AR_ALLOW_UNVALIDATED_SCORE_CONTROL 开启。"""

    # 平移缩放与训练保持一致。
    INFER_PROFILE = os.environ.get("AR_INFER_PROFILE", "paper_translation_stop1mm_last")
    PATIENT_ID = case_config.PATIENT_ID
    FRAME_ID = case_config.FRAME_ID
    CASE_ROOT = case_config.CASE_ROOT
    WEIGHT_PATH = case_config.refinenet_weight_path()
    IMG_SIZE = case_config.IMG_SIZE
    TRANS_SCALE = 50.0

    MAX_ITER = int(os.environ.get("AR_MAX_ITER", "10"))
    STOP_TRANS_MM = float(os.environ.get("AR_STOP_TRANS_MM", "1.0"))

    # 首步系数为基础 damping × FIRST_STEP_DAMPING；后续使用 REFINE_*。
    TRANS_DAMPING = float(os.environ.get("AR_TRANS_DAMPING", "0.4"))
    ROT_DAMPING = float(os.environ.get("AR_ROT_DAMPING", "0.4"))
    FIRST_STEP_DAMPING = float(os.environ.get("AR_FIRST_STEP_DAMPING", "0.75"))
    REFINE_TRANS_DAMPING = float(os.environ.get("AR_REFINE_TRANS_DAMPING", "0.1"))
    REFINE_ROT_DAMPING = float(os.environ.get("AR_REFINE_ROT_DAMPING", "0.1"))
    MAX_STEP_TRANS_MM = float(os.environ.get("AR_MAX_STEP_TRANS_MM", "0.0"))
    MAX_STEP_ROT_DEG = float(os.environ.get("AR_MAX_STEP_ROT_DEG", "0.0"))

    # 发散阈值；STOP_DRIFT_MM=0 关闭漂移检查。
    DIVERGE_RAW_TRANS_MM = float(os.environ.get("AR_DIVERGE_RAW_TRANS_MM", "120.0"))
    DIVERGE_RAW_ROT_DEG = float(os.environ.get("AR_DIVERGE_RAW_ROT_DEG", "45.0"))
    DIVERGE_POSE_TRANS_MM = float(os.environ.get("AR_DIVERGE_POSE_TRANS_MM", "200.0"))
    STOP_DRIFT_MM = float(os.environ.get("AR_STOP_DRIFT_MM", "0.0"))

    # 图像分数控制参数，默认用于诊断。
    RAW_SCORE_ROT_WEIGHT = float(os.environ.get("AR_RAW_SCORE_ROT_WEIGHT", "5.0"))
    RAW_WORSEN_PATIENCE = int(os.environ.get("AR_RAW_WORSEN_PATIENCE", "0"))
    RAW_WORSEN_MIN_DELTA = float(os.environ.get("AR_RAW_WORSEN_MIN_DELTA", "0.5"))
    SELECT_BEST_BY = os.environ.get("AR_SELECT_BEST_BY", "last").lower()
    ACCEPT_BY_SCORE = os.environ.get("AR_ACCEPT_BY_SCORE", "0").lower() in ("1", "true", "yes")
    SCORE_ACCEPT_TOL = float(os.environ.get("AR_SCORE_ACCEPT_TOL", "0.0"))
    SCORE_PATIENCE = int(os.environ.get("AR_SCORE_PATIENCE", "0"))
    ALLOW_UNVALIDATED_SCORE_CONTROL = os.environ.get(
        "AR_ALLOW_UNVALIDATED_SCORE_CONTROL", "0"
    ).lower() in ("1", "true", "yes")

    # 图像空间诊断指标。
    CONTOUR_SCORE_MAX_DIST_PX = float(os.environ.get("AR_CONTOUR_SCORE_MAX_DIST_PX", "50.0"))
    CONTOUR_SCORE_MASK_WEIGHT = float(os.environ.get("AR_CONTOUR_SCORE_MASK_WEIGHT", "20.0"))

    # 位姿更新方式和缺失深度消融开关。
    UPDATE_MODE = os.environ.get("AR_UPDATE_MODE", "right")
    ALLOW_ZERO_DEPTH = os.environ.get("AR_ALLOW_ZERO_DEPTH", "0").lower() in ("1", "true", "yes")


class DummyCfg:
    use_BN = True
    rot_rep = "6d"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_logger(log_path, name="InferExportSTL"):
    ensure_dir(os.path.dirname(log_path))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def validate_inference_control_config(cfg):
    """检查图像分数控制项与消融开关是否一致。"""
    unsafe = []
    if getattr(cfg, "SELECT_BEST_BY", "last") != "last":
        unsafe.append("AR_SELECT_BEST_BY must be 'last'")
    if bool(getattr(cfg, "ACCEPT_BY_SCORE", False)):
        unsafe.append("AR_ACCEPT_BY_SCORE must be disabled")
    if int(getattr(cfg, "SCORE_PATIENCE", 0)) > 0:
        unsafe.append("AR_SCORE_PATIENCE must be 0")
    if int(getattr(cfg, "RAW_WORSEN_PATIENCE", 0)) > 0:
        unsafe.append("AR_RAW_WORSEN_PATIENCE must be 0")
    if unsafe and not bool(getattr(cfg, "ALLOW_UNVALIDATED_SCORE_CONTROL", False)):
        raise ValueError(
            "Unvalidated proxy-based inference control is disabled. "
            + "; ".join(unsafe)
            + ". Set AR_ALLOW_UNVALIDATED_SCORE_CONTROL=1 only for an explicit ablation."
        )


def _binary_chamfer_score(src, dst, max_dist_px):
    src = src.astype(bool)
    dst = dst.astype(bool)
    if not np.any(src) and not np.any(dst):
        return 0.0
    if not np.any(src) or not np.any(dst):
        return float(max_dist_px)

    dist_to_dst = cv2.distanceTransform((~dst).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_src = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 3)
    src_to_dst = float(np.mean(np.clip(dist_to_dst[src], 0.0, max_dist_px)))
    dst_to_src = float(np.mean(np.clip(dist_to_src[dst], 0.0, max_dist_px)))
    return 0.5 * (src_to_dst + dst_to_src)


def compute_contour_score(rendered_contours, rendered_mask, target_contours, target_mask, max_dist_px, mask_weight):
    channel_scores = [
        _binary_chamfer_score(rendered_contours[ch] > 0, target_contours[ch] > 0, max_dist_px)
        for ch in range(3)
    ]
    rendered_mask = rendered_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    union = np.logical_or(rendered_mask, target_mask)
    iou = (
        float(np.count_nonzero(np.logical_and(rendered_mask, target_mask)) / np.count_nonzero(union))
        if np.any(union)
        else 0.0
    )
    contour_score = float(np.mean(channel_scores))
    score = contour_score + float(mask_weight) * (1.0 - iou)
    return score, {
        "score": score,
        "contour_score": contour_score,
        "mask_iou": iou,
        "silhouette_score": float(channel_scores[0]),
        "ridge_score": float(channel_scores[1]),
        "ligament_score": float(channel_scores[2]),
    }


def pack_modalities(contours, depth, mask):
    import torch

    width, height = Config.IMG_SIZE
    contours_rs = np.stack(
        [cv2.resize(contours[i], (width, height), interpolation=cv2.INTER_NEAREST) for i in range(3)],
        axis=0,
    )
    depth_rs = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    mask_rs = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    x = np.concatenate([contours_rs, depth_rs[None], mask_rs[None]], axis=0)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def compute_rotation_matrix_from_6d(poses):
    import torch
    import torch.nn.functional as functional

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = functional.normalize(x_raw, p=2, dim=-1)
    z_dot_x = (y_raw * x).sum(dim=-1, keepdim=True)
    y = functional.normalize(y_raw - z_dot_x * x, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class InferenceRenderer:
    def __init__(self, case_dir, width, height, da2_estimator):
        from shared.pose_render import PoseSampleRenderer

        self.da2 = da2_estimator
        self.pose_renderer = PoseSampleRenderer(
            os.path.join(case_dir, "models", "Liver_obj.ply"),
            os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
            os.path.join(case_dir, "contours", "model_contours.json"),
            os.path.join(case_dir, "camera", "camera_new.txt"),
            os.path.join(case_dir, "T_view.txt"),
            width,
            height,
        )

    def render_network_input(self, pose):
        contours, mask, rgb_bgr, _, _, _ = self.pose_renderer.render_at_pose(pose)
        depth_raw = self.da2.infer(rgb_bgr)
        mask_float = (mask > 127).astype(np.float32)
        depth_raw[mask_float <= 0] = 0.0
        depth_net = depth_for_network_eval(depth_raw, mask_float)
        return pack_modalities(contours.astype(np.float32) / 255.0, depth_net, mask_float)

    def render_observation(self, pose):
        contours, mask, _, _, _, _ = self.pose_renderer.render_at_pose(pose)
        return contours, (mask > 127).astype(np.float32)


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata: {path}: {exc}") from exc


def load_real_B(case_dir, image_shape, intraop_depth_path, allow_zero_depth=False, logger=None, return_observation=False):
    """读取并校验真实帧的轮廓、掩码和深度。"""
    height, width = image_shape
    contour_path = os.path.join(case_dir, "contours", "gt_multicontour.npy")
    if not os.path.exists(contour_path):
        raise FileNotFoundError(f"Missing real contours: {contour_path}. Run scripts/01_prepare_frame_data_and_models.py first.")
    contours = np.load(contour_path).astype(np.float32)
    if contours.shape[0] != 3:
        raise ValueError(f"Expected three contour channels in {contour_path}, got {contours.shape}.")
    if contours.shape[1:] != (height, width):
        contours = np.stack(
            [cv2.resize(contours[i], (width, height), interpolation=cv2.INTER_NEAREST) for i in range(3)],
            axis=0,
        )

    full_liver_mask, depth_valid_mask, mask_quality = validate_case_masks(
        case_dir,
        expected_shape=(height, width),
    )
    network_mask = (full_liver_mask > 0).astype(np.float32)
    expected_fingerprints = mask_fingerprints(full_liver_mask, depth_valid_mask)

    quality_path = os.path.join(os.path.dirname(intraop_depth_path), "da2_intraop_quality.json")
    if not os.path.exists(intraop_depth_path):
        message = f"Missing intra-op depth: {intraop_depth_path}. Run scripts/07_generate_depth_for_real_frame.py first."
        if not allow_zero_depth:
            raise FileNotFoundError(message)
        if logger is not None:
            logger.warning("[WARN] %s AR_ALLOW_ZERO_DEPTH is enabled; using zeros.", message)
        depth_raw = np.zeros((height, width), dtype=np.float32)
        depth_stats = depth_quality(depth_raw, depth_raw, full_liver_mask, depth_valid_mask)
    else:
        if not os.path.exists(quality_path):
            raise FileNotFoundError(
                f"Missing audited depth metadata: {quality_path}. Re-run scripts/07_generate_depth_for_real_frame.py."
            )
        saved_quality = _read_json(quality_path)
        if saved_quality.get("mask_fingerprints") != expected_fingerprints:
            raise ValueError(
                "The stored intra-operative depth was generated from different masks. "
                "Re-run scripts/07_generate_depth_for_real_frame.py after mask annotation."
            )
        depth_raw = np.load(intraop_depth_path).astype(np.float32)
        if depth_raw.shape != (height, width):
            raise ValueError(
                f"Intra-operative depth shape mismatch: got {depth_raw.shape}, expected {(height, width)}. "
                "Re-run scripts/07_generate_depth_for_real_frame.py."
            )
        depth_stats = depth_quality(depth_raw, depth_raw, full_liver_mask, depth_valid_mask)
        validate_depth_quality(depth_stats)

    depth_net = depth_for_network_eval(depth_raw, network_mask)
    if return_observation:
        return contours, depth_net, network_mask, mask_quality, depth_stats
    return pack_modalities(contours, depth_net, network_mask)


def rotation_angle_deg(rotation):
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def make_transform(rotation, translation):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform


def damp_offset(rotation, translation, trans_factor, rot_factor):
    trans_factor = float(np.clip(trans_factor, 0.0, 1.0))
    rot_factor = float(np.clip(rot_factor, 0.0, 1.0))
    rotvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
    damped_rotation, _ = cv2.Rodrigues(rotvec * rot_factor)
    return damped_rotation.astype(np.float32), translation.astype(np.float32) * trans_factor


def clip_damped_offset(rotation, translation, max_trans_mm, max_rot_deg):
    clipped = False
    translation_out = translation.astype(np.float32).copy()
    trans_norm = float(np.linalg.norm(translation_out))
    if max_trans_mm > 0 and trans_norm > max_trans_mm:
        translation_out *= float(max_trans_mm / max(trans_norm, 1e-8))
        clipped = True

    rotation_out = rotation.astype(np.float32)
    rot_deg = rotation_angle_deg(rotation_out)
    if max_rot_deg > 0 and rot_deg > max_rot_deg:
        rotvec, _ = cv2.Rodrigues(rotation_out.astype(np.float64))
        rotation_out, _ = cv2.Rodrigues(rotvec * float(max_rot_deg / max(rot_deg, 1e-8)))
        rotation_out = rotation_out.astype(np.float32)
        clipped = True
    return rotation_out, translation_out, clipped


def raw_update_score(raw_trans_mm, raw_rot_deg, rot_weight):
    return float(raw_trans_mm + raw_rot_deg * rot_weight)


def apply_update(current_pose, offset, mode):
    if mode == "right":
        return current_pose @ offset
    if mode == "left":
        return offset @ current_pose
    if mode == "inverse_right":
        return current_pose @ np.linalg.inv(offset)
    raise ValueError(f"Unknown UPDATE_MODE: {mode}")


def export_transformed_mesh(mesh_path, pose, out_path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices).astype(np.float64)
    if len(vertices) == 0:
        raise ValueError(f"Cannot read mesh vertices: {mesh_path}")
    vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    mesh.vertices = o3d.utility.Vector3dVector((pose @ vertices_h.T).T[:, :3])
    ensure_dir(os.path.dirname(out_path))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    extension = os.path.splitext(out_path)[1].lower()
    if not o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=(extension != ".stl")):
        raise RuntimeError(f"Failed to write mesh: {out_path}")


def save_debug_state(debug_dir, iter_idx, pose, offset, tumour_path):
    name = f"iter_{iter_idx:02d}"
    np.savetxt(os.path.join(debug_dir, f"{name}_pose.txt"), pose, fmt="%.10f")
    if offset is not None:
        np.savetxt(os.path.join(debug_dir, f"{name}_offset.txt"), offset, fmt="%.10f")
    export_transformed_mesh(tumour_path, pose, os.path.join(debug_dir, f"{name}_tumour.stl"))


def write_history_csv(path, history):
    fieldnames = [
        "iter",
        "accepted",
        "raw_trans_mm",
        "raw_rot_deg",
        "raw_score",
        "damped_trans_mm",
        "damped_rot_deg",
        "trans_factor",
        "rot_factor",
        "step_clipped",
        "pose_drift_mm",
        "contour_score",
        "mask_iou",
        "silhouette_score",
        "ridge_score",
        "ligament_score",
        "is_best_contour",
        "stop_trigger",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            parts = row.get("contour_parts") or {}
            writer.writerow(
                {
                    "iter": row.get("iter"),
                    "accepted": int(bool(row.get("accepted", True))),
                    "raw_trans_mm": row.get("raw_trans_mm"),
                    "raw_rot_deg": row.get("raw_rot_deg"),
                    "raw_score": row.get("raw_score"),
                    "damped_trans_mm": row.get("damped_trans_mm"),
                    "damped_rot_deg": row.get("damped_rot_deg"),
                    "trans_factor": row.get("trans_factor"),
                    "rot_factor": row.get("rot_factor"),
                    "step_clipped": int(bool(row.get("step_clipped", False))),
                    "pose_drift_mm": row.get("pose_drift_mm"),
                    "contour_score": row.get("contour_score"),
                    "mask_iou": parts.get("mask_iou"),
                    "silhouette_score": parts.get("silhouette_score"),
                    "ridge_score": parts.get("ridge_score"),
                    "ligament_score": parts.get("ligament_score"),
                    "is_best_contour": int(bool(row.get("is_best_contour", False))),
                    "stop_trigger": row.get("stop_trigger", ""),
                }
            )


def _check_required_files(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required inference files:\n" + "\n".join(missing))


def main():
    cfg = Config()
    validate_inference_control_config(cfg)

    case_dir = case_config.case_dir(cfg.CASE_ROOT)
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    tview_path = os.path.join(case_dir, "T_view.txt")
    liver_path = os.path.join(case_dir, "models", "Liver_obj.ply")
    tumour_path = os.path.join(case_dir, "models", "Tumour_obj.ply")
    intraop_depth = os.path.join(case_config.depth_dir(cfg.CASE_ROOT), "da2_intraop.npy")
    inference_dir = case_config.inference_dir()
    logs_dir = case_config.logs_dir()
    debug_dir = os.path.join(inference_dir, "debug_iterations")
    ensure_dir(inference_dir)
    ensure_dir(logs_dir)

    preflight_logger = setup_logger(os.path.join(logs_dir, "infer_preflight.log"), "InferPreflight")
    try:
        _check_required_files([image_path, tview_path, liver_path, tumour_path, cfg.WEIGHT_PATH])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        B_contours, B_depth, B_mask, mask_quality, depth_stats = load_real_B(
            case_dir,
            (height, width),
            intraop_depth,
            allow_zero_depth=cfg.ALLOW_ZERO_DEPTH,
            logger=preflight_logger,
            return_observation=True,
        )
        preflight_logger.info("[OK] mask validation: %s", quality_summary(mask_quality))
        preflight_logger.info(
            "[OK] depth validation: coverage=%.4f outside=%d",
            depth_stats["masked_depth_valid_to_depth_valid_mask_ratio"],
            depth_stats["nonzero_outside_depth_valid_mask"],
        )
    except Exception:
        preflight_logger.exception("[FAIL] inference preflight")
        raise

    if os.path.isdir(debug_dir):
        shutil.rmtree(debug_dir)
    ensure_dir(debug_dir)
    logger = setup_logger(os.path.join(logs_dir, "infer.log"))

    import torch
    from learning.models.refine_network import RefineNet
    from shared.da2_engine import get_estimator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefineNet(cfg=DummyCfg(), c_in=5).to(device)
    model.load_state_dict(torch.load(cfg.WEIGHT_PATH, map_location=device))
    model.eval()
    renderer = InferenceRenderer(case_dir, width, height, get_estimator())
    B_tensor = pack_modalities(B_contours, B_depth, B_mask).to(device)

    current_pose = np.loadtxt(tview_path).astype(np.float32)
    initial_pose = current_pose.copy()

    logger.info("=== 09. Iterative rigid pose inference ===")
    logger.info("  profile: %s", cfg.INFER_PROFILE)
    logger.info("  frame: %s/%s", cfg.PATIENT_ID, cfg.FRAME_ID)
    logger.info("  weight: %s", cfg.WEIGHT_PATH)
    logger.info("  real mask: manually annotated full_liver_mask outer contour")
    logger.info("  real depth: audited DA2 depth masked by the same outer contour")
    logger.info("  mask quality: %s", quality_summary(mask_quality))
    logger.info(
        "  depth quality: valid/outer_mask=%.4f, outside=%d",
        depth_stats["masked_depth_valid_to_depth_valid_mask_ratio"],
        depth_stats["nonzero_outside_depth_valid_mask"],
    )
    logger.info(
        "  damping: first_trans=%.3f first_rot=%.3f refine_trans=%.3f refine_rot=%.3f",
        cfg.TRANS_DAMPING * cfg.FIRST_STEP_DAMPING,
        cfg.ROT_DAMPING * cfg.FIRST_STEP_DAMPING,
        cfg.REFINE_TRANS_DAMPING,
        cfg.REFINE_ROT_DAMPING,
    )
    logger.info("  stop: accepted predicted translation residual < %.3f mm", cfg.STOP_TRANS_MM)
    logger.info("  max_iter: %d, update_mode: %s", cfg.MAX_ITER, cfg.UPDATE_MODE)
    logger.info(
        "  selection: %s; contour/mask scores are diagnostic only by default",
        cfg.SELECT_BEST_BY,
    )
    if cfg.ALLOW_UNVALIDATED_SCORE_CONTROL:
        logger.warning("[WARN] unvalidated proxy-based inference control is explicitly enabled")

    save_debug_state(debug_dir, 0, current_pose, None, tumour_path)
    contours0, mask0 = renderer.render_observation(current_pose)
    best_score, best_parts = compute_contour_score(
        contours0,
        mask0,
        B_contours,
        B_mask,
        cfg.CONTOUR_SCORE_MAX_DIST_PX,
        cfg.CONTOUR_SCORE_MASK_WEIGHT,
    )
    best_iter = 0
    best_pose = current_pose.copy()
    best_offset = None
    current_score = best_score
    no_score_improve_count = 0
    best_raw_score = None
    raw_worsen_count = 0
    stop_reason = None
    history = [
        {
            "iter": 0,
            "accepted": True,
            "raw_trans_mm": 0.0,
            "raw_rot_deg": 0.0,
            "raw_score": 0.0,
            "damped_trans_mm": 0.0,
            "damped_rot_deg": 0.0,
            "pose_drift_mm": 0.0,
            "contour_score": float(best_score),
            "contour_parts": best_parts,
            "is_best_contour": True,
            "step_clipped": False,
            "stop_trigger": "",
        }
    ]
    logger.info("Iter 00: diagnostic_contour_score=%.4f mask_iou=%.4f", best_score, best_parts["mask_iou"])

    with torch.no_grad():
        for iteration in range(1, cfg.MAX_ITER + 1):
            A_tensor = renderer.render_network_input(current_pose).to(device)
            output = model(A_tensor, B_tensor)
            rotation = compute_rotation_matrix_from_6d(output["rot"]).squeeze(0).cpu().numpy()
            translation = output["trans"].squeeze(0).cpu().numpy() * cfg.TRANS_SCALE
            raw_trans = float(np.linalg.norm(translation))
            raw_rot = rotation_angle_deg(rotation)
            raw_score = raw_update_score(raw_trans, raw_rot, cfg.RAW_SCORE_ROT_WEIGHT)

            if not np.isfinite(raw_trans) or not np.isfinite(raw_rot):
                stop_reason = "diverged_nan_offset"
                logger.warning("[STOP] Iter %02d produced a non-finite offset.", iteration)
                break
            if raw_trans > cfg.DIVERGE_RAW_TRANS_MM or raw_rot > cfg.DIVERGE_RAW_ROT_DEG:
                stop_reason = "diverged_raw_offset"
                logger.warning(
                    "[STOP] Iter %02d raw offset too large: trans=%.4f mm rot=%.4f deg",
                    iteration,
                    raw_trans,
                    raw_rot,
                )
                break

            if iteration == 1:
                trans_factor = cfg.TRANS_DAMPING * cfg.FIRST_STEP_DAMPING
                rot_factor = cfg.ROT_DAMPING * cfg.FIRST_STEP_DAMPING
            else:
                trans_factor = cfg.REFINE_TRANS_DAMPING
                rot_factor = cfg.REFINE_ROT_DAMPING
            damped_rotation, damped_translation = damp_offset(
                rotation, translation, trans_factor, rot_factor
            )
            damped_rotation, damped_translation, step_clipped = clip_damped_offset(
                damped_rotation,
                damped_translation,
                cfg.MAX_STEP_TRANS_MM,
                cfg.MAX_STEP_ROT_DEG,
            )
            offset = make_transform(damped_rotation, damped_translation)
            candidate_pose = apply_update(current_pose, offset, cfg.UPDATE_MODE)
            damped_trans = float(np.linalg.norm(damped_translation))
            damped_rot = rotation_angle_deg(damped_rotation)
            pose_drift = float(np.linalg.norm(candidate_pose[:3, 3] - initial_pose[:3, 3]))

            if not np.all(np.isfinite(candidate_pose)):
                stop_reason = "diverged_nan_pose"
                logger.warning("[STOP] Iter %02d produced a non-finite pose.", iteration)
                break
            if pose_drift > cfg.DIVERGE_POSE_TRANS_MM:
                stop_reason = "diverged_pose"
                logger.warning("[STOP] Iter %02d pose drift %.4f mm is too large.", iteration, pose_drift)
                break
            if cfg.STOP_DRIFT_MM > 0 and pose_drift > cfg.STOP_DRIFT_MM:
                stop_reason = "stop_drift"
                logger.info("[STOP] Iter %02d reached configured drift limit.", iteration)
                break

            candidate_contours, candidate_mask = renderer.render_observation(candidate_pose)
            contour_score, contour_parts = compute_contour_score(
                candidate_contours,
                candidate_mask,
                B_contours,
                B_mask,
                cfg.CONTOUR_SCORE_MAX_DIST_PX,
                cfg.CONTOUR_SCORE_MASK_WEIGHT,
            )
            is_best_contour = contour_score < best_score
            if is_best_contour:
                best_score = contour_score
                best_parts = contour_parts
                best_iter = iteration
                best_pose = candidate_pose.copy()
                best_offset = offset.copy()
                no_score_improve_count = 0
            else:
                no_score_improve_count += 1

            accepted = not (
                cfg.ACCEPT_BY_SCORE and contour_score > current_score + cfg.SCORE_ACCEPT_TOL
            )
            save_debug_state(debug_dir, iteration, candidate_pose, offset, tumour_path)
            history.append(
                {
                    "iter": iteration,
                    "accepted": bool(accepted),
                    "raw_trans_mm": raw_trans,
                    "raw_rot_deg": raw_rot,
                    "raw_score": raw_score,
                    "damped_trans_mm": damped_trans,
                    "damped_rot_deg": damped_rot,
                    "trans_factor": float(trans_factor),
                    "rot_factor": float(rot_factor),
                    "step_clipped": bool(step_clipped),
                    "pose_drift_mm": pose_drift,
                    "contour_score": float(contour_score),
                    "contour_parts": contour_parts,
                    "is_best_contour": bool(is_best_contour),
                    "stop_trigger": "",
                }
            )
            logger.info(
                "Iter %02d: raw_trans=%.4f mm raw_rot=%.4f deg | damped_trans=%.4f mm "
                "damped_rot=%.4f deg | drift=%.4f mm | diagnostic_contour=%.4f mask_iou=%.4f%s%s",
                iteration,
                raw_trans,
                raw_rot,
                damped_trans,
                damped_rot,
                pose_drift,
                contour_score,
                contour_parts["mask_iou"],
                " [diagnostic best]" if is_best_contour else "",
                " [rejected]" if not accepted else "",
            )

            if not accepted:
                stop_reason = "score_rejected"
                history[-1]["stop_trigger"] = stop_reason
                break

            current_pose = candidate_pose
            current_score = contour_score

            if raw_trans < cfg.STOP_TRANS_MM:
                stop_reason = "converged_translation_residual"
                history[-1]["stop_trigger"] = stop_reason
                logger.info("[OK] accepted predicted translation residual is below %.3f mm.", cfg.STOP_TRANS_MM)
                break

            if best_raw_score is None or raw_score < best_raw_score - cfg.RAW_WORSEN_MIN_DELTA:
                best_raw_score = raw_score
                raw_worsen_count = 0
            else:
                raw_worsen_count += 1
            if cfg.RAW_WORSEN_PATIENCE > 0 and raw_worsen_count >= cfg.RAW_WORSEN_PATIENCE:
                stop_reason = "raw_score_no_improve"
                history[-1]["stop_trigger"] = stop_reason
                break
            if cfg.SCORE_PATIENCE > 0 and no_score_improve_count >= cfg.SCORE_PATIENCE:
                stop_reason = "contour_score_patience"
                history[-1]["stop_trigger"] = stop_reason
                break

    if stop_reason is None:
        stop_reason = "max_iter_reached"
        logger.info("[OK] max_iter reached.")

    with open(os.path.join(debug_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    write_history_csv(os.path.join(debug_dir, "history.csv"), history)

    if cfg.SELECT_BEST_BY == "last":
        selected_pose = current_pose
        selected_iter = max(int(row["iter"]) for row in history if row.get("accepted", True))
        selected_reason = "last_safe_accepted"
        selected_row = next(row for row in history if int(row["iter"]) == selected_iter)
        selected_score = float(selected_row.get("contour_score") or best_score)
    elif cfg.SELECT_BEST_BY == "contour":
        selected_pose = best_pose
        selected_iter = best_iter
        selected_reason = "diagnostic_lowest_contour_score"
        selected_score = float(best_score)
    else:
        raise ValueError(f"Unknown AR_SELECT_BEST_BY={cfg.SELECT_BEST_BY!r}.")

    selection = {
        "selected_iter": int(selected_iter),
        "selected_reason": selected_reason,
        "selected_contour_score_diagnostic": selected_score,
        "diagnostic_best_contour_iter": int(best_iter),
        "diagnostic_best_contour_score": float(best_score),
        "diagnostic_best_contour_parts": best_parts,
        "stop_reason": stop_reason,
        "warning": (
            "Contour and mask overlap are image-space diagnostics, not validated surrogates for TRE/IC. "
            "The default profile does not use them to select the final pose."
        ),
        "mask_quality": mask_quality,
        "depth_quality": depth_stats,
        "config": {
            "infer_profile": cfg.INFER_PROFILE,
            "max_iter": cfg.MAX_ITER,
            "stop_trans_mm": cfg.STOP_TRANS_MM,
            "update_mode": cfg.UPDATE_MODE,
            "trans_damping": cfg.TRANS_DAMPING,
            "rot_damping": cfg.ROT_DAMPING,
            "first_step_damping": cfg.FIRST_STEP_DAMPING,
            "refine_trans_damping": cfg.REFINE_TRANS_DAMPING,
            "refine_rot_damping": cfg.REFINE_ROT_DAMPING,
            "select_best_by": cfg.SELECT_BEST_BY,
            "accept_by_score": cfg.ACCEPT_BY_SCORE,
            "score_patience": cfg.SCORE_PATIENCE,
            "raw_worsen_patience": cfg.RAW_WORSEN_PATIENCE,
        },
    }
    with open(os.path.join(debug_dir, "best_selection.json"), "w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    np.savetxt(os.path.join(inference_dir, "last_pose.txt"), current_pose, fmt="%.10f")
    np.savetxt(os.path.join(inference_dir, "diagnostic_contour_best_pose.txt"), best_pose, fmt="%.10f")
    if best_offset is not None:
        np.savetxt(
            os.path.join(inference_dir, "diagnostic_contour_best_offset.txt"),
            best_offset,
            fmt="%.10f",
        )
    final_pose_path = os.path.join(inference_dir, "final_pose.txt")
    np.savetxt(final_pose_path, selected_pose, fmt="%.10f")
    tumour_pred = os.path.join(inference_dir, "tumour_pred_cam.stl")
    export_transformed_mesh(tumour_path, selected_pose, tumour_pred)
    export_transformed_mesh(liver_path, selected_pose, os.path.join(inference_dir, "liver_pred_cam.ply"))

    logger.info("[OK] stop_reason: %s", stop_reason)
    logger.info("[OK] selected_iter: %02d (%s)", selected_iter, selected_reason)
    logger.info("[OK] final_pose: %s", final_pose_path)
    logger.info("[OK] tumour STL for Python evaluation: %s", tumour_pred)


if __name__ == "__main__":
    main()
