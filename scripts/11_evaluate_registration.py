"""按 Evaluation.m 的计算流程评价肿瘤 STL，输出逐帧、合并及迭代 TRE/IC。"""

import argparse
import csv
import json
import math
import os
import re
import struct
from pathlib import Path

import numpy as np
import scipy.io
from scipy.spatial import cKDTree


DEFAULT_FRAMES = "all"
ICP_ITERATIONS = 10
PROFILE_SAMPLES = 50
ONCOLOGIC_MARGIN_MM = 10.0


def patient_no_from_id(patient_id):
    match = re.fullmatch(r"Patient(\d+)", patient_id, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse patient id: {patient_id!r}")
    return int(match.group(1))


def vec_norm(x, axis=0):
    return np.sqrt(np.sum(x * x, axis=axis))


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tumour_exterior(json_path):
    data = load_json(json_path)
    objects = data.get("objects")
    if isinstance(objects, dict):
        candidates = [objects]
    elif isinstance(objects, list):
        candidates = objects
    else:
        raise ValueError(f"Unsupported objects value in {json_path}: {type(objects)}")

    tumour_obj = None
    for obj in candidates:
        if str(obj.get("classTitle", "")).lower() == "tumor":
            tumour_obj = obj
            break
    if tumour_obj is None:
        tumour_obj = candidates[0]
    return np.asarray(tumour_obj["points"]["exterior"], dtype=np.float64).T


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
        denom = max(float(np.linalg.norm(next_point - current)), 1e-8)
        current = current + remain * (next_point - current) / denom
        dist_to_next = float(np.linalg.norm(next_point - current))
        sampled[:, i] = current
    return sampled


def load_profile_points(data_root, patient_id, frame_id):
    patient_tag = f"Patient{patient_no_from_id(patient_id)}"
    mat_path = data_root / "Dataset" / patient_tag / "LUS Calibration and Pose" / f"{frame_id}.mat"
    json_path = data_root / "Dataset" / patient_tag / "LUS Segmentation" / "json" / f"{frame_id}.json"
    mat = scipy.io.loadmat(mat_path)
    profile = load_tumour_exterior(json_path)

    slus = float(np.squeeze(mat["SLus"]))
    saus = float(np.squeeze(mat["SAus"]))
    rus = np.asarray(mat["Rus"], dtype=np.float64)
    tus = np.asarray(mat["Tus"], dtype=np.float64)
    rpr = np.asarray(mat["Rpr"], dtype=np.float64)
    tpr = np.asarray(mat["Tpr"], dtype=np.float64)

    us_profile = np.vstack([slus * profile[0, :], np.zeros(profile.shape[1]), saus * profile[1, :]])
    us_probe = rus @ us_profile + tus
    us_cam = rpr @ us_probe + tpr
    return sample_profile(us_cam, PROFILE_SAMPLES)


def read_stl_mesh(path):
    data = Path(path).read_bytes()
    vertices, faces = read_binary_stl(data)
    if vertices is None:
        vertices, faces = read_ascii_stl(data)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"No mesh triangles found in {path}")
    return vertices, faces


def read_binary_stl(data):
    if len(data) < 84:
        return None, None
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_len = 84 + tri_count * 50
    if expected_len != len(data):
        return None, None

    triangles = np.empty((tri_count, 3, 3), dtype=np.float64)
    offset = 84
    for idx in range(tri_count):
        offset += 12
        triangles[idx] = np.asarray(struct.unpack_from("<9f", data, offset), dtype=np.float64).reshape(3, 3)
        offset += 38
    return unique_triangle_vertices(triangles)


def read_ascii_stl(data):
    text = data.decode("utf-8", errors="ignore")
    coords = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(coords) % 3 != 0:
        raise ValueError("ASCII STL has incomplete vertex triplets")
    triangles = np.asarray(coords, dtype=np.float64).reshape((-1, 3, 3))
    return unique_triangle_vertices(triangles)


def unique_triangle_vertices(triangles):
    flat = triangles.reshape((-1, 3))
    rounded = np.round(flat, decimals=8)
    vertices, inverse = np.unique(rounded, axis=0, return_inverse=True)
    return vertices.astype(np.float64), inverse.reshape((-1, 3)).astype(np.int32)


def rigid_fit(source, target):
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    source0 = source - source_center
    target0 = target - target_center
    h = source0 @ target0.T
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def tre_like_matlab(profile_by_frame, vertices_by_frame, iterations=ICP_ITERATIONS):
    t0_parts = []
    for frame in profile_by_frame:
        profile = profile_by_frame[frame]
        vertices = vertices_by_frame[frame]
        t0_parts.append(vertices.T.mean(axis=1, keepdims=True) - profile.mean(axis=1, keepdims=True))
    rotation = np.eye(3, dtype=np.float64)
    translation = np.mean(np.concatenate(t0_parts, axis=1), axis=1, keepdims=True)

    closest_by_frame = {}
    trees = {frame: cKDTree(vertices) for frame, vertices in vertices_by_frame.items()}
    for _ in range(iterations):
        for frame, profile in profile_by_frame.items():
            query = (rotation @ profile + translation).T
            _, indices = trees[frame].query(query, k=1)
            closest_by_frame[frame] = vertices_by_frame[frame][indices].T
        all_profile = np.concatenate([profile_by_frame[frame] for frame in profile_by_frame], axis=1)
        all_closest = np.concatenate([closest_by_frame[frame] for frame in profile_by_frame], axis=1)
        rotation, translation = rigid_fit(all_profile, all_closest)

    all_profile = np.concatenate([profile_by_frame[frame] for frame in profile_by_frame], axis=1)
    all_closest = np.concatenate([closest_by_frame[frame] for frame in profile_by_frame], axis=1)
    return float(np.mean(vec_norm(all_profile - all_closest, axis=0)))


def expanded_vertices(vertices, margin_mm=ONCOLOGIC_MARGIN_MM):
    center = vertices.mean(axis=0, keepdims=True)
    direction = vertices - center
    scale = np.linalg.norm(direction, axis=1, keepdims=True)
    return vertices + margin_mm * direction / np.maximum(scale, 1e-8)


def point_inside_mesh(point, triangles):
    directions = np.asarray(
        [
            [1.0, 0.231, 0.119],
            [0.173, 1.0, 0.347],
            [0.271, 0.131, 1.0],
        ],
        dtype=np.float64,
    )
    votes = 0
    for direction in directions:
        direction = direction / np.linalg.norm(direction)
        hits = ray_hits(point, direction, triangles)
        if hits % 2 == 1:
            votes += 1
    return votes >= 2


def ray_hits(point, direction, triangles):
    eps = 1e-9
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    mask = np.abs(a) > eps
    if not np.any(mask):
        return 0
    f = np.zeros_like(a)
    f[mask] = 1.0 / a[mask]
    s = point - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = f * np.einsum("j,ij->i", direction, q)
    t = f * np.einsum("ij,ij->i", edge2, q)
    hit_mask = mask & (u >= -1e-8) & (v >= -1e-8) & ((u + v) <= 1.0 + 1e-8) & (t > 1e-8)
    ts = np.sort(t[hit_mask])
    if len(ts) == 0:
        return 0
    unique_hits = 1
    previous = ts[0]
    for current in ts[1:]:
        if abs(current - previous) > 1e-6:
            unique_hits += 1
            previous = current
    return unique_hits


def ic_like_matlab(profile_points, vertices, faces):
    vertices_aug = expanded_vertices(vertices)
    triangles = vertices_aug[faces]
    return int(all(point_inside_mesh(point, triangles) for point in profile_points.T))


def stl_path_for(result_root, patient_id, frame_id, mode):
    inference_dir = result_root / patient_id / frame_id / "inference"
    if mode.lower() == "final":
        return inference_dir / "tumour_pred_cam.stl"
    if mode.lower().startswith("iter_"):
        return inference_dir / "debug_iterations" / f"{mode}_tumour.stl"
    raise ValueError(f"Unsupported mode: {mode}")


def discover_frames(data_root, result_root, patient_id, frames, mode):
    if frames != DEFAULT_FRAMES:
        return [frame.strip() for frame in frames.split(",") if frame.strip()]
    patient_tag = f"Patient{patient_no_from_id(patient_id)}"
    mat_dir = data_root / "Dataset" / patient_tag / "LUS Calibration and Pose"
    discovered = []
    for mat_path in sorted(mat_dir.glob("*.mat")):
        frame = mat_path.stem
        if stl_path_for(result_root, patient_id, frame, mode).exists():
            discovered.append(frame)
    return discovered


def read_selection(result_root, patient_id, frame_id):
    selection_path = result_root / patient_id / frame_id / "inference" / "debug_iterations" / "best_selection.json"
    if not selection_path.exists():
        return {}
    return load_json(selection_path)


def evaluate_mode(data_root, result_root, patient_id, frames, mode):
    profiles = {}
    vertices = {}
    faces = {}
    rows = []
    for frame in frames:
        stl_path = stl_path_for(result_root, patient_id, frame, mode)
        if not stl_path.exists():
            continue
        profile = load_profile_points(data_root, patient_id, frame)
        mesh_vertices, mesh_faces = read_stl_mesh(stl_path)
        profiles[frame] = profile
        vertices[frame] = mesh_vertices
        faces[frame] = mesh_faces
        rows.append(
            {
                "patient": patient_id,
                "frame": frame,
                "mode": mode,
                "TRE_mm": tre_like_matlab({frame: profile}, {frame: mesh_vertices}),
                "IC": ic_like_matlab(profile, mesh_vertices, mesh_faces),
                "stl": str(stl_path),
            }
        )
    global_result = {
        "patient": patient_id,
        "mode": mode,
        "frames": list(profiles),
        "frame_count": len(profiles),
        "TRE_mm": tre_like_matlab(profiles, vertices) if profiles else None,
        "IC": int(all(ic_like_matlab(profiles[frame], vertices[frame], faces[frame]) for frame in profiles)) if profiles else 0,
    }
    return rows, global_result


def available_iter_modes(result_root, patient_id, frame_id):
    debug_dir = result_root / patient_id / frame_id / "inference" / "debug_iterations"
    modes = []
    if debug_dir.exists():
        for path in sorted(debug_dir.glob("iter_*_tumour.stl")):
            match = re.fullmatch(r"(iter_\d+)_tumour\.stl", path.name)
            if match:
                modes.append(match.group(1))
    return modes


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default="Patient1")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--result-root", default="results/run")
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--mode", default="final")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-debug", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    frames = discover_frames(data_root, result_root, args.patient_id, args.frames, args.mode)

    final_rows, global_result = evaluate_mode(data_root, result_root, args.patient_id, frames, args.mode)
    frame_summary_rows = []
    debug_rows = []
    for row in final_rows:
        selection = read_selection(result_root, args.patient_id, row["frame"])
        row_out = dict(row)
        row_out.update(
            {
                "selected_iter": selection.get("selected_iter"),
                "selected_reason": selection.get("selected_reason"),
                "stop_reason": selection.get("stop_reason"),
                "diagnostic_best_contour_iter": selection.get("diagnostic_best_contour_iter"),
            }
        )
        best_debug = None
        if args.include_debug:
            for mode in available_iter_modes(result_root, args.patient_id, row["frame"]):
                debug_eval_rows, _ = evaluate_mode(data_root, result_root, args.patient_id, [row["frame"]], mode)
                if not debug_eval_rows:
                    continue
                debug_row = debug_eval_rows[0]
                debug_row["iter"] = int(mode.split("_", 1)[1])
                debug_rows.append(debug_row)
                if best_debug is None or debug_row["TRE_mm"] < best_debug["TRE_mm"]:
                    best_debug = debug_row
        if best_debug is not None:
            row_out["posthoc_best_tre_iter"] = best_debug["iter"]
            row_out["posthoc_best_TRE_mm"] = best_debug["TRE_mm"]
            row_out["posthoc_best_IC"] = best_debug["IC"]
        frame_summary_rows.append(row_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_fields = [
        "patient",
        "frame",
        "mode",
        "selected_iter",
        "selected_reason",
        "stop_reason",
        "diagnostic_best_contour_iter",
        "TRE_mm",
        "IC",
        "posthoc_best_tre_iter",
        "posthoc_best_TRE_mm",
        "posthoc_best_IC",
        "stl",
    ]
    write_csv(output_dir / f"{args.patient_id}_{args.mode}_frame_summary.csv", frame_summary_rows, frame_fields)
    if args.include_debug:
        write_csv(
            output_dir / f"{args.patient_id}_debug_iterations.csv",
            debug_rows,
            ["patient", "frame", "mode", "iter", "TRE_mm", "IC", "stl"],
        )

    with open(output_dir / f"{args.patient_id}_{args.mode}_global_evaluation.json", "w", encoding="utf-8") as file:
        json.dump(global_result, file, indent=2)
    write_csv(
        output_dir / f"{args.patient_id}_{args.mode}_global_evaluation.csv",
        [
            {
                "Patient": args.patient_id,
                "ResultMode": args.mode,
                "Frames": global_result["frame_count"],
                "TRE": global_result["TRE_mm"],
                "IC": global_result["IC"],
            }
        ],
        ["Patient", "ResultMode", "Frames", "TRE", "IC"],
    )
    print(json.dumps(global_result, indent=2))


if __name__ == "__main__":
    main()
