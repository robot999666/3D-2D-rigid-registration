"""将源图像、相机、轮廓和网格转换为单帧工作数据。"""

import os
import json
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config as _cc


class Config:
    """预处理配置；公共路径和图像参数来自 case_config。"""

    PATIENT_ID = _cc.PATIENT_ID
    FRAME_ID = _cc.FRAME_ID

    DATA_ROOT = _cc.DATA_ROOT
    CASE_ROOT = _cc.CASE_ROOT
    ANNOTATOR = _cc.ANNOTATOR
    LAP_IMAGE_NAME = f"{FRAME_ID}.png"

    # 去畸变 alpha：0 裁剪黑边，1 保留全部像素。
    UNDISTORT_ALPHA = 0.0

    # 图像、轮廓和模型输入使用相同工作尺寸。
    WORK_WIDTH = _cc.WORK_WIDTH
    WORK_HEIGHT = _cc.WORK_HEIGHT

    # 预处理与渲染共用轮廓线宽。
    CONTOUR_THICKNESS = _cc.CONTOUR_THICKNESS

    # 模型来源：
    # koo_obj：OBJ 用于渲染和轮廓索引。
    # dataset_hybrid：PLY 用于渲染，OBJ 保留 XML 顶点索引。
    MODEL_SOURCE = os.environ.get("AR_MODEL_SOURCE", "koo_obj").lower()

    # XML 坐标从 1 开始，OpenCV 坐标从 0 开始。
    XML_COORDINATE_IS_ONE_BASED = True


TYPE_TO_CHANNEL = {
    "Silhouette": 0,
    "Ridge": 1,
    "Ligament": 2,
}

CHANNEL_NAMES = ["silhouette", "ridge", "ligament"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_camera_parameters(path):
    tree = ET.parse(path)
    root = tree.getroot()

    def get_float(name):
        node = root.find(name)
        if node is None:
            raise ValueError(f"CameraParameters 缺少字段: {name}")
        return float(node.text)

    def get_int(name):
        return int(round(get_float(name)))

    width = get_int("width")
    height = get_int("height")
    fx = get_float("fx")
    fy = get_float("fy")
    cx = get_float("cx")
    cy = get_float("cy")
    skew = get_float("skew")

    k1 = get_float("k1")
    k2 = get_float("k2")
    k3 = get_float("k3")
    p1 = get_float("p1")
    p2 = get_float("p2")

    K = np.array([
        [fx, skew, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # OpenCV 5 参数顺序: k1, k2, p1, p2, k3
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist, width, height


def save_matrix_txt(path, mat, header):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in mat:
            f.write(" ".join(f"{v:.10f}" for v in row) + "\n")


def parse_number_list(text, dtype=float):
    if text is None or not text.strip():
        return []
    return [dtype(x.strip()) for x in text.replace("\n", "").split(",") if x.strip()]


def scale_camera_matrix(K, scale_x, scale_y):
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 1] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y
    return K_scaled


def undistort_points(points_xy, K_raw, dist, K_new, scale_x=1.0, scale_y=1.0):
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K_raw, dist, P=K_new).reshape(-1, 2).astype(np.float32)
    undist[:, 0] *= scale_x
    undist[:, 1] *= scale_y
    return undist


def parse_and_undistort_contours(xml_path, K_raw, dist, K_new, scale_x=1.0, scale_y=1.0):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    contours = []

    for contour_node in root.findall("contour"):
        contour_type = contour_node.findtext("contourType")
        if contour_type not in TYPE_TO_CHANNEL:
            continue

        image_node = contour_node.find("imagePoints")
        if image_node is None:
            continue

        xs = parse_number_list(image_node.findtext("x"), float)
        ys = parse_number_list(image_node.findtext("y"), float)
        if len(xs) != len(ys):
            raise ValueError(f"{xml_path} 中 {contour_type} 的 x/y 点数不一致")

        pts_raw = np.stack([xs, ys], axis=1).astype(np.float32)
        if Config.XML_COORDINATE_IS_ONE_BASED:
            pts_raw -= 1.0
        pts_undist = undistort_points(pts_raw, K_raw, dist, K_new, scale_x, scale_y)

        vertices = []
        model_node = contour_node.find("modelPoints")
        if model_node is not None:
            vertices_node = model_node.find("vertices")
            if vertices_node is not None:
                vertices = parse_number_list(vertices_node.text, int)

        contours.append({
            "type": contour_type,
            "image_points_raw": pts_raw.tolist(),
            "image_points_undistorted": pts_undist.tolist(),
            "model_vertices": vertices,
        })

    return contours


def rasterize_contours(contours, width, height, thickness):
    maps = np.zeros((3, height, width), dtype=np.uint8)
    for item in contours:
        channel = TYPE_TO_CHANNEL[item["type"]]
        pts = np.asarray(item["image_points_undistorted"], dtype=np.float32)
        if len(pts) < 2:
            continue
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, width - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, height - 1)
        cv2.polylines(maps[channel], [pts_i.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=thickness)
    return maps


def load_mesh_vertices_faces(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertices():
        raise ValueError(f"无法读取模型: {path}")
    return mesh


def mesh_vertices(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices.")
    return vertices


def parse_obj_vertices(path):
    """按 OBJ 原始顺序读取顶点，保留 XML 中的顶点索引。"""
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ 中未读取到顶点: {path}")
    return np.asarray(vertices, dtype=np.float64)


def resolve_model_paths(data_root, patient_id, model_source):
    contour_root = os.path.join(data_root, "contours_and_models", patient_id)
    dataset_root = os.path.join(data_root, "Dataset", patient_id, "Preoperative Model")
    koo_liver_path = os.path.join(contour_root, "Liver.obj")
    koo_tumour_path = os.path.join(contour_root, "Tumour.obj")

    if model_source == "koo_obj":
        return {
            "model_source": model_source,
            "render_liver_path": koo_liver_path,
            "render_tumour_path": koo_tumour_path,
            "contour_liver_path": koo_liver_path,
        }
    if model_source == "dataset_hybrid":
        return {
            "model_source": model_source,
            "render_liver_path": os.path.join(dataset_root, "Liver.ply"),
            "render_tumour_path": os.path.join(dataset_root, "Tumour.ply"),
            "contour_liver_path": koo_liver_path,
        }
    raise ValueError(
        f"Unsupported AR_MODEL_SOURCE={model_source!r}. "
        "Use 'koo_obj' or 'dataset_hybrid'."
    )


def assert_model_paths_exist(paths):
    missing = [
        path
        for key, path in paths.items()
        if key.endswith("_path") and isinstance(path, str) and path and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing model source files:\n" + "\n".join(missing))


def center_and_save_models(model_paths, output_model_dir):
    render_liver_path = model_paths["render_liver_path"]
    render_tumour_path = model_paths["render_tumour_path"]
    contour_liver_path = model_paths["contour_liver_path"]

    liver = load_mesh_vertices_faces(render_liver_path)
    tumour = load_mesh_vertices_faces(render_tumour_path)

    # 网格用于渲染，OBJ 原始顶点顺序用于轮廓索引。
    contour_vertices_raw_order = parse_obj_vertices(contour_liver_path)
    contour_liver_centroid = contour_vertices_raw_order.mean(axis=0)
    if model_paths["model_source"] == "koo_obj":
        render_liver_centroid = contour_liver_centroid
    else:
        render_liver_centroid = mesh_vertices(liver).mean(axis=0)
    contour_vertices_centered_raw_order = contour_vertices_raw_order - contour_liver_centroid

    liver.vertices = o3d.utility.Vector3dVector(mesh_vertices(liver) - render_liver_centroid)
    tumour.vertices = o3d.utility.Vector3dVector(mesh_vertices(tumour) - render_liver_centroid)

    liver_out = os.path.join(output_model_dir, "Liver_obj.ply")
    tumour_out = os.path.join(output_model_dir, "Tumour_obj.ply")
    centroid_out = os.path.join(output_model_dir, "liver_centroid.txt")
    contour_centroid_out = os.path.join(output_model_dir, "contour_liver_centroid.txt")
    raw_order_vertices_out = os.path.join(output_model_dir, "Liver_vertices_centered_raw_order.npy")
    meta_out = os.path.join(output_model_dir, "model_source_meta.json")

    o3d.io.write_triangle_mesh(liver_out, liver, write_ascii=True)
    o3d.io.write_triangle_mesh(tumour_out, tumour, write_ascii=True)
    np.savetxt(centroid_out, render_liver_centroid.reshape(1, 3), fmt="%.10f")
    np.savetxt(contour_centroid_out, contour_liver_centroid.reshape(1, 3), fmt="%.10f")
    np.save(raw_order_vertices_out, contour_vertices_centered_raw_order.astype(np.float32))

    metadata = {
        "model_source": model_paths["model_source"],
        "render_liver_path": os.path.abspath(render_liver_path),
        "render_tumour_path": os.path.abspath(render_tumour_path),
        "contour_liver_path": os.path.abspath(contour_liver_path),
        "render_liver_centroid": render_liver_centroid.tolist(),
        "contour_liver_centroid": contour_liver_centroid.tolist(),
        "render_minus_contour_liver_centroid": (
            render_liver_centroid - contour_liver_centroid
        ).tolist(),
        "render_mesh_outputs": {
            "liver": os.path.abspath(liver_out),
            "tumour": os.path.abspath(tumour_out),
        },
        "contour_vertices_output": os.path.abspath(raw_order_vertices_out),
        "notes": (
            "Render/export meshes and Koo contour vertices are centered independently "
            "by their own liver centroids before downstream pose rendering."
        ),
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return liver_out, tumour_out, render_liver_centroid, metadata


def make_debug_overlay(image, contour_maps):
    overlay = image.copy()
    colors = [
        (0, 255, 0),    # silhouette: green
        (255, 0, 0),    # ridge: blue
        (0, 255, 255),  # ligament: yellow
    ]
    for c, color in enumerate(colors):
        mask = contour_maps[c] > 0
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
    return overlay


def main():
    cfg = Config()
    case_dir = os.path.join(cfg.CASE_ROOT, cfg.PATIENT_ID, cfg.FRAME_ID)
    image_dir = os.path.join(case_dir, "image")
    camera_dir = os.path.join(case_dir, "camera")
    contour_dir = os.path.join(case_dir, "contours")
    model_dir = os.path.join(case_dir, "models")
    for d in [image_dir, camera_dir, contour_dir, model_dir]:
        ensure_dir(d)

    lap_image_path = os.path.join(cfg.DATA_ROOT, "Dataset", cfg.PATIENT_ID, "Lap Images", cfg.LAP_IMAGE_NAME)
    camera_path = os.path.join(cfg.DATA_ROOT, "Dataset", cfg.PATIENT_ID, "Lap Images", "CameraParameters")
    xml_path = os.path.join(cfg.DATA_ROOT, "contours_and_models", cfg.PATIENT_ID, "Annotations", cfg.FRAME_ID, cfg.ANNOTATOR, "contours.xml")
    model_paths = resolve_model_paths(cfg.DATA_ROOT, cfg.PATIENT_ID, cfg.MODEL_SOURCE)
    assert_model_paths_exist(model_paths)

    print("=== 01. Prepare case ===")
    print(f"Patient/Frame: {cfg.PATIENT_ID}/{cfg.FRAME_ID}")
    print(f"Model source: {cfg.MODEL_SOURCE}")

    K_raw, dist, width, height = parse_camera_parameters(camera_path)
    K_new_full, roi = cv2.getOptimalNewCameraMatrix(K_raw, dist, (width, height), cfg.UNDISTORT_ALPHA, (width, height))
    scale_x = cfg.WORK_WIDTH / float(width)
    scale_y = cfg.WORK_HEIGHT / float(height)
    K_work = scale_camera_matrix(K_new_full, scale_x, scale_y)

    save_matrix_txt(os.path.join(camera_dir, "camera_raw.txt"), K_raw, "# Raw camera matrix")
    save_matrix_txt(os.path.join(camera_dir, "camera_new_full.txt"), K_new_full, "# Full-resolution undistorted camera matrix")
    save_matrix_txt(os.path.join(camera_dir, "camera_new.txt"), K_work, "# Working-resolution undistorted camera matrix")
    np.savetxt(os.path.join(camera_dir, "dist.txt"), dist.reshape(1, -1), fmt="%.10f")

    img_raw = cv2.imread(lap_image_path, cv2.IMREAD_COLOR)
    if img_raw is None:
        raise FileNotFoundError(f"找不到术中图像: {lap_image_path}")
    img_undist_full = cv2.undistort(img_raw, K_raw, dist, None, K_new_full)
    img_undist = cv2.resize(img_undist_full, (cfg.WORK_WIDTH, cfg.WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(image_dir, "lap_raw.png"), img_raw)
    cv2.imwrite(os.path.join(image_dir, "lap_undist_full.png"), img_undist_full)
    cv2.imwrite(os.path.join(image_dir, "lap_undist.png"), img_undist)

    contours = parse_and_undistort_contours(xml_path, K_raw, dist, K_new_full, scale_x, scale_y)
    contour_maps = rasterize_contours(contours, cfg.WORK_WIDTH, cfg.WORK_HEIGHT, cfg.CONTOUR_THICKNESS)

    for i, name in enumerate(CHANNEL_NAMES):
        cv2.imwrite(os.path.join(contour_dir, f"gt_{name}.png"), contour_maps[i])
    np.save(os.path.join(contour_dir, "gt_multicontour.npy"), contour_maps.astype(np.float32) / 255.0)

    with open(os.path.join(contour_dir, "contours_undistorted.json"), "w", encoding="utf-8") as f:
        json.dump(contours, f, indent=2)

    model_contours = [{"type": c["type"], "model_vertices": c["model_vertices"]} for c in contours]
    with open(os.path.join(contour_dir, "model_contours.json"), "w", encoding="utf-8") as f:
        json.dump(model_contours, f, indent=2)

    overlay = make_debug_overlay(img_undist, contour_maps)
    cv2.imwrite(os.path.join(contour_dir, "gt_contour_overlay.png"), overlay)

    liver_out, tumour_out, centroid, model_meta = center_and_save_models(model_paths, model_dir)

    print("[OK] 输出完成：")
    print(f"  工作分辨率: {cfg.WORK_WIDTH}x{cfg.WORK_HEIGHT}")
    print(f"  去畸变 RGB: {os.path.join(image_dir, 'lap_undist.png')}")
    print(f"  工作分辨率新内参: {os.path.join(camera_dir, 'camera_new.txt')}")
    print(f"  去畸变多轮廓: {os.path.join(contour_dir, 'gt_multicontour.npy')}")
    print(f"  Liver: {liver_out}")
    print(f"  Tumour: {tumour_out}")
    print(f"  Liver centroid(raw): {centroid}")
    print(f"  Model source meta: {os.path.join(model_dir, 'model_source_meta.json')}")
    if cfg.MODEL_SOURCE == "dataset_hybrid":
        print(
            "  Dataset-Koo liver centroid delta: "
            f"{np.asarray(model_meta['render_minus_contour_liver_centroid'])}"
        )


if __name__ == "__main__":
    main()
