"""使用 Open3D 渲染轮廓、肝脏区域及 DA2 输入图像。"""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d

TYPE_TO_CHANNEL = {"Silhouette": 0, "Ridge": 1, "Ligament": 2}


def load_camera_matrix(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=np.float64)


def euler_to_matrix(rx, ry, rz) -> np.ndarray:
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def generate_rays(K, T_c2w, width, height):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    R = T_c2w[:3, :3]
    dirs_world = dirs @ R.T
    origins = np.broadcast_to(T_c2w[:3, 3], dirs_world.shape)
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    return o3d.core.Tensor(rays)


def project_points(points_obj, K, T_obj_to_cam, width, height):
    if len(points_obj) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pts_h = np.concatenate([points_obj, np.ones((len(points_obj), 1), dtype=np.float32)], axis=1)
    pts_cam = (T_obj_to_cam @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[valid]
    if len(pts_cam) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]
    pts = np.stack([u, v], axis=1)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    return np.round(pts[inside]).astype(np.int32)


def metric_depth_to_pseudo_lap_bgr(depth_metric: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """将几何深度映射为灰度 BGR 图像，供 DA2 使用。"""
    h, w = mask_uint8.shape[:2]
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask_uint8 > 127
    if not np.any(valid):
        return bgr
    d = depth_metric.astype(np.float32)
    d_pos = d[valid]
    d_pos = d_pos[d_pos > 0]
    if len(d_pos) == 0:
        gray = np.full((h, w), 120, dtype=np.uint8)
        bgr[valid] = np.stack([gray[valid]] * 3, axis=-1)
        return bgr
    d_min, d_max = float(d_pos.min()), float(d_pos.max())
    gray = np.zeros((h, w), dtype=np.float32)
    if d_max - d_min > 1e-6:
        gray[valid] = (d[valid] - d_min) / (d_max - d_min) * 200.0 + 40.0
    else:
        gray[valid] = 120.0
    g = np.clip(gray, 0, 255).astype(np.uint8)
    bgr[valid] = np.stack([g[valid], g[valid], g[valid]], axis=-1)
    return bgr


class PoseSampleRenderer:
    def __init__(
        self,
        liver_path: str,
        raw_order_vertices_path: str,
        model_contours_path: str,
        camera_path: str,
        tview_path: str,
        width: int,
        height: int,
        contour_thickness: int = 2,
    ):
        self.width = width
        self.height = height
        self.contour_thickness = contour_thickness
        self.K = load_camera_matrix(camera_path)
        self.T_view = np.loadtxt(tview_path).astype(np.float32)

        mesh = o3d.io.read_triangle_mesh(liver_path)
        if not mesh.has_vertices():
            raise ValueError(f"Cannot load liver mesh: {liver_path}")
        self.verts_for_contours = np.load(raw_order_vertices_path).astype(np.float32)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_t)

        with open(model_contours_path, "r", encoding="utf-8") as f:
            self.model_contours = json.load(f)

    def render(self, delta_T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """渲染一个相对位姿。

        返回：
            contours：形状为 (3,H,W) 的 uint8 多通道轮廓；
            mask：uint8 肝脏掩码；
            rgb_bgr：作为 DA2 输入的 uint8 图像；
            T_render：4x4 object-to-camera 位姿；
            visible_types：当前可见的轮廓类型；
            depth_metric：float32 metric depth。
        """
        T_render = self.T_view @ delta_T.astype(np.float32)
        T_c2w = np.linalg.inv(T_render)
        rays = generate_rays(self.K, T_c2w, self.width, self.height)
        ans = self.scene.cast_rays(rays)
        depth_hit = ans["t_hit"].numpy()

        mask = (depth_hit < np.inf).astype(np.uint8) * 255
        depth_metric = np.where(depth_hit < np.inf, depth_hit, 0.0).astype(np.float32)
        rgb_bgr = metric_depth_to_pseudo_lap_bgr(depth_metric, mask)

        contours = np.zeros((3, self.height, self.width), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        contours[0] = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        for item in self.model_contours:
            ctype = item["type"]
            if ctype not in ("Ridge", "Ligament"):
                continue
            indices = [
                idx for idx in item.get("model_vertices", [])
                if 0 <= idx < len(self.verts_for_contours)
            ]
            pts = project_points(self.verts_for_contours[indices], self.K, T_render, self.width, self.height)
            if len(pts) >= 2:
                ch = TYPE_TO_CHANNEL[ctype]
                cv2.polylines(
                    contours[ch], [pts.reshape(-1, 1, 2)], False, 255, self.contour_thickness
                )

        visible = []
        for name, ch in [("silhouette", 0), ("ridge", 1), ("ligament", 2)]:
            if np.count_nonzero(contours[ch]) > 0:
                visible.append(name)

        return contours, mask, rgb_bgr, T_render, visible, depth_metric

    def render_at_pose(self, T_obj_to_cam: np.ndarray):
        """按物体到相机的绝对位姿渲染。"""
        delta_T = np.linalg.inv(self.T_view) @ T_obj_to_cam.astype(np.float32)
        return self.render(delta_T)
