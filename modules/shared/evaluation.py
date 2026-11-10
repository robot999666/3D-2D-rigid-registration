import csv
import json
import os
import re

import numpy as np
import open3d as o3d
import scipy.io
from scipy.spatial import cKDTree


def patient_no_from_id(patient_id):
    if patient_id.lower().startswith("patient"):
        return int(patient_id[len("Patient"):])
    raise ValueError(f"Cannot parse patient number from {patient_id!r}")


def vec_norm(x, axis=0):
    return np.sqrt(np.sum(x * x, axis=axis))


def sample_profile(points, count):
    perimeter = float(np.sum(vec_norm(points[:, 1:] - points[:, :-1], axis=0)))
    delta = perimeter / count
    sampled = np.zeros((3, count), dtype=np.float64)
    sampled[:, 0] = points[:, 0]
    next_idx = 1
    current = points[:, 0].copy()
    next_point = points[:, 1].copy()
    dist_to_next = float(np.linalg.norm(next_point - current))
    for i in range(1, count):
        remain = delta
        while remain > dist_to_next and next_idx + 1 < points.shape[1]:
            next_idx += 1
            remain -= dist_to_next
            current = points[:, next_idx - 1].copy()
            next_point = points[:, next_idx].copy()
            dist_to_next = float(np.linalg.norm(next_point - current))
        current = current + remain * (next_point - current) / max(float(np.linalg.norm(next_point - current)), 1e-8)
        dist_to_next = float(np.linalg.norm(next_point - current))
        sampled[:, i] = current
    return sampled


def load_us_profile(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects")
    if not objects:
        raise ValueError(f"No objects in JSON: {json_path}")
    tumour_obj = None
    if isinstance(objects, list):
        for obj in objects:
            if str(obj.get("classTitle", "")).lower() == "tumor":
                tumour_obj = obj
                break
        if tumour_obj is None:
            tumour_obj = objects[0]
    elif isinstance(objects, dict):
        tumour_obj = objects
    else:
        raise ValueError(f"Unsupported objects format: {type(objects)}")
    exterior = tumour_obj["points"]["exterior"]
    return np.asarray(exterior, dtype=np.float64).T


def load_profile_points(data_root, patient_id, frame_id, sample_count):
    patient_no = patient_no_from_id(patient_id)
    patient_tag = f"Patient{patient_no}"
    mat_path = os.path.join(data_root, "Dataset", patient_tag, "LUS Calibration and Pose", f"{frame_id}.mat")
    json_path = os.path.join(data_root, "Dataset", patient_tag, "LUS Segmentation", "json", f"{frame_id}.json")
    mat = scipy.io.loadmat(mat_path)
    profile = load_us_profile(json_path)

    slus = float(np.squeeze(mat["SLus"]))
    saus = float(np.squeeze(mat["SAus"]))
    rus = np.asarray(mat["Rus"], dtype=np.float64)
    tus = np.asarray(mat["Tus"], dtype=np.float64)
    rpr = np.asarray(mat["Rpr"], dtype=np.float64)
    tpr = np.asarray(mat["Tpr"], dtype=np.float64)

    us_profile = np.vstack([slus * profile[0, :], np.zeros(profile.shape[1]), saus * profile[1, :]])
    us_probe = rus @ us_profile + tus
    us_cam = rpr @ us_probe + tpr
    return sample_profile(us_cam, sample_count)


def read_stl_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if len(vertices) == 0:
        raise ValueError(f"No vertices in STL: {path}")
    if len(triangles) == 0:
        raise ValueError(f"No triangles in STL: {path}")
    return vertices, triangles


def rigid_fit(source, target):
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    source0 = source - source_center
    target0 = target - target_center
    h = source0 @ target0.T
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_center - r @ source_center
    return r, t


def tre_calc(profile_points, tumour_vertices, iterations):
    tree = cKDTree(tumour_vertices)
    cog_p = profile_points.mean(axis=1, keepdims=True)
    cog_t = tumour_vertices.T.mean(axis=1, keepdims=True)
    r = np.eye(3, dtype=np.float64)
    t = cog_t - cog_p
    closest = None
    for _ in range(iterations):
        query = (r @ profile_points + t).T
        _, idx = tree.query(query, k=1)
        closest = tumour_vertices[idx].T
        r, t = rigid_fit(profile_points, closest)
    return float(np.mean(vec_norm(profile_points - closest, axis=0)))


def ic_calc(profile_points, tumour_vertices, triangles, oncologic_margin):
    center = tumour_vertices.mean(axis=0, keepdims=True)
    direction = tumour_vertices - center
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    vertices_aug = tumour_vertices + oncologic_margin * direction / np.maximum(norm, 1e-8)

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = o3d.core.Tensor(vertices_aug.astype(np.float32))
    mesh.triangle["indices"] = o3d.core.Tensor(triangles.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    query = o3d.core.Tensor(profile_points.T.astype(np.float32))
    occupancy = scene.compute_occupancy(query).numpy()
    return int(np.all(occupancy > 0.5))


def iter_from_name(name):
    match = re.search(r"iter_(\d+)_tumour\.stl$", name)
    if not match:
        return None
    return int(match.group(1))


def eval_debug_dir(debug_dir, data_root, patient_id, frame_id, sample_count, icp_iterations, oncologic_margin):
    profile_points = load_profile_points(data_root, patient_id, frame_id, sample_count)
    rows = []
    for name in sorted(os.listdir(debug_dir)):
        iter_no = iter_from_name(name)
        if iter_no is None:
            continue
        stl_path = os.path.join(debug_dir, name)
        vertices, triangles = read_stl_mesh(stl_path)
        tre = tre_calc(profile_points, vertices, icp_iterations)
        ic = ic_calc(profile_points, vertices, triangles, oncologic_margin)
        rows.append({"iter": iter_no, "TRE_mm": tre, "IC": ic, "file": name})
    rows.sort(key=lambda row: row["iter"])
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "TRE_mm", "IC", "file"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


