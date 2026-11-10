"""按 prepare、annotate、run 阶段批量处理帧，并汇总 TRE/IC。"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def require_python_modules(stage, skip_existing_depth=False, skip_existing_infer=False):
    """检查当前阶段所需的 Python 模块。"""
    required = {"cv2", "numpy"}
    if stage in ("prepare", "all"):
        required.add("open3d")
    if stage in ("run", "all"):
        required.update({"open3d", "scipy"})
        if not skip_existing_depth or not skip_existing_infer:
            required.add("torch")
    missing = sorted(name for name in required if importlib.util.find_spec(name) is None)
    if missing:
        raise ModuleNotFoundError(
            f"Stage '{stage}' is missing Python modules: {', '.join(missing)}. "
            "Install requirements.txt or use the documented Docker/cloud environment."
        )


def frame_id(number):
    return f"{int(number):02d}"


def discover_frames(data_root, patient_id):
    """从 Lap Images 发现数字命名的帧。"""
    lap_dir = os.path.join(data_root, "Dataset", patient_id, "Lap Images")
    if not os.path.isdir(lap_dir):
        raise FileNotFoundError(f"Lap Images directory does not exist: {lap_dir}")
    frames = []
    for name in os.listdir(lap_dir):
        stem, extension = os.path.splitext(name)
        if extension.lower() == ".png" and stem.isdigit():
            frames.append(frame_id(stem))
    frames = sorted(set(frames), key=int)
    if not frames:
        raise ValueError(f"No numeric PNG frames found in: {lap_dir}")
    return frames


def parse_frames(text, data_root=None, patient_id=None):
    if text.strip().lower() == "all":
        if data_root is None or patient_id is None:
            raise ValueError("parse_frames('all') requires data_root and patient_id.")
        return discover_frames(data_root, patient_id)

    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(frame_id(value) for value in range(int(start), int(end) + 1))
        else:
            frames.append(frame_id(chunk))
    if not frames:
        raise ValueError("No frames were selected.")
    return list(dict.fromkeys(frames))


def frame_paths(patient_id, frame, case_root=None, result_root=None):
    case_dir = case_config.case_dir(case_root, patient_id, frame)
    result_dir = case_config.result_dir(result_root, patient_id, frame)
    return {
        "case": case_dir,
        "result": result_dir,
        "logs": os.path.join(result_dir, "logs"),
        "inference": os.path.join(result_dir, "inference"),
        "debug": os.path.join(result_dir, "inference", "debug_iterations"),
        "evaluation": os.path.join(result_dir, "evaluation"),
        "mask_qa": os.path.join(result_dir, "mask_qa"),
        "lap_image": os.path.join(case_dir, "image", "lap_undist.png"),
        "initial_pose": os.path.join(case_dir, "T_view.txt"),
        "depth": os.path.join(case_dir, "depth", "da2_intraop.npy"),
        "final_stl": os.path.join(result_dir, "inference", "tumour_pred_cam.stl"),
    }


def required_source_inputs(data_root, patient_id, frame):
    inputs = {
        "lap_image": os.path.join(data_root, "Dataset", patient_id, "Lap Images", f"{frame}.png"),
        "camera": os.path.join(data_root, "Dataset", patient_id, "Lap Images", "CameraParameters"),
        "calibration": os.path.join(
            data_root, "Dataset", patient_id, "LUS Calibration and Pose", f"{frame}.mat"
        ),
        "segmentation": os.path.join(
            data_root, "Dataset", patient_id, "LUS Segmentation", "json", f"{frame}.json"
        ),
        "contours": os.path.join(
            data_root,
            "contours_and_models",
            patient_id,
            "Annotations",
            frame,
            case_config.ANNOTATOR,
            "contours.xml",
        ),
        "liver": os.path.join(data_root, "contours_and_models", patient_id, "Liver.obj"),
        "tumour": os.path.join(data_root, "contours_and_models", patient_id, "Tumour.obj"),
    }
    if os.environ.get("AR_MODEL_SOURCE", "koo_obj").lower() == "dataset_hybrid":
        preop_dir = os.path.join(data_root, "Dataset", patient_id, "Preoperative Model")
        inputs.update(
            {
                "render_liver": os.path.join(preop_dir, "Liver.ply"),
                "render_tumour": os.path.join(preop_dir, "Tumour.ply"),
            }
        )
    return inputs


def missing_source_inputs(data_root, patient_id, frame):
    return {
        name: path
        for name, path in required_source_inputs(data_root, patient_id, frame).items()
        if not os.path.exists(path)
    }


def run_step(script_name, env, log_path):
    ensure_dir(os.path.dirname(log_path))
    command = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script_name)]
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def run_interactive_step(script_name, env, extra_args=None):
    command = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script_name)]
    command.extend(extra_args or [])
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    return process.returncode


def preflight_case_masks(case_dir, image_path=None):
    import cv2

    from shared.intraop_masks import quality_summary, validate_case_masks

    image_path = image_path or os.path.join(case_dir, "image", "lap_undist.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read prepared frame image: {image_path}")
    _, _, quality = validate_case_masks(case_dir, expected_shape=image.shape[:2])
    return quality, quality_summary(quality)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def write_rows(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_selection(debug_dir):
    path = os.path.join(debug_dir, "best_selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing inference selection metadata: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_debug(debug_dir, data_root, patient_id, frame, out_csv):
    from shared.evaluation import eval_debug_dir, write_csv

    rows = eval_debug_dir(
        debug_dir,
        data_root,
        patient_id,
        frame,
        sample_count=50,
        icp_iterations=10,
        oncologic_margin=10.0,
    )
    if not rows:
        raise ValueError(f"No iteration STL files found for evaluation: {debug_dir}")
    write_csv(out_csv, rows)
    return rows


def final_row_from_eval(frame, eval_rows, selection):
    selected_iter = int(selection["selected_iter"])
    selected_eval = next(
        (row for row in eval_rows if int(row["iter"]) == selected_iter),
        None,
    )
    if selected_eval is None:
        raise ValueError(f"Selected iteration {selected_iter} is missing from Python evaluation.")
    posthoc_best = min(eval_rows, key=lambda row: float(row["TRE_mm"]))
    return {
        "frame": frame,
        "status": "ok",
        "selected_iter": selected_iter,
        "stop_reason": selection.get("stop_reason", ""),
        "selected_TRE_mm": f"{float(selected_eval['TRE_mm']):.10f}",
        "selected_IC": int(selected_eval["IC"]),
        "posthoc_best_iter": int(posthoc_best["iter"]),
        "posthoc_best_TRE_mm": f"{float(posthoc_best['TRE_mm']):.10f}",
        "posthoc_best_IC": int(posthoc_best["IC"]),
        "selected_is_posthoc_best": int(selected_iter == int(posthoc_best["iter"])),
    }


def write_final_evaluation(path, patient_id, frame, stl_path, row):
    text = (
        "Evaluation: Python TRE/IC implementation\n"
        "Inference selection: last safe accepted pose; evaluation is post-hoc only\n"
        f"Patient: {patient_id}\n"
        f"Frame: {frame}\n"
        f"STL: {stl_path}\n"
        f"TRE_mm: {row['selected_TRE_mm']}\n"
        f"IC: {'Pass' if int(row['selected_IC']) == 1 else 'Failed'}\n"
        f"IC_numeric: {row['selected_IC']}\n"
        f"Selected_iter: {row['selected_iter']}\n"
        f"Stop_reason: {row['stop_reason']}\n"
        f"Posthoc_best_iter: {row['posthoc_best_iter']}\n"
        f"Posthoc_best_TRE_mm: {row['posthoc_best_TRE_mm']}\n"
    )
    write_text(path, text)


def evaluate_frame(paths, data_root, patient_id, frame):
    ensure_dir(paths["evaluation"])
    eval_csv = os.path.join(paths["evaluation"], "debug_iterations.csv")
    eval_rows = evaluate_debug(paths["debug"], data_root, patient_id, frame, eval_csv)
    selection = read_selection(paths["debug"])
    frame_row = final_row_from_eval(frame, eval_rows, selection)

    selection_report = {
        "inference_selection": selection,
        "posthoc_evaluation": frame_row,
        "warning": (
            "TRE/IC and posthoc_best are evaluation outputs only. "
            "They are not read by scripts/09_infer_pose_and_export_stl.py."
        ),
    }
    with open(os.path.join(paths["evaluation"], "selection.json"), "w", encoding="utf-8") as file:
        json.dump(selection_report, file, indent=2)
    write_final_evaluation(
        os.path.join(paths["evaluation"], "final_eval.txt"),
        patient_id,
        frame,
        paths["final_stl"],
        frame_row,
    )
    write_text(
        os.path.join(paths["logs"], "evaluation.log"),
        (
            f"[OK] selected_iter={frame_row['selected_iter']} "
            f"TRE_mm={frame_row['selected_TRE_mm']} IC={frame_row['selected_IC']}\n"
        ),
    )
    return frame_row, eval_rows


def make_environment(patient_id, frame, weight):
    env = os.environ.copy()
    env["PATIENT_ID"] = patient_id
    env["FRAME_ID"] = frame
    env["AR_WEIGHT_PATH"] = weight
    return env


def copy_shared_initial_pose(
    patient_id,
    source_frame,
    target_frame,
    case_root=None,
    result_root=None,
):
    """复制参考帧初始位姿并记录来源。"""
    source = frame_paths(patient_id, source_frame, case_root, result_root)
    target = frame_paths(patient_id, target_frame, case_root, result_root)
    if not os.path.exists(source["initial_pose"]):
        raise FileNotFoundError(f"Reference initial pose does not exist: {source['initial_pose']}")
    ensure_dir(target["case"])
    shutil.copy2(source["initial_pose"], target["initial_pose"])

    source_meta_path = os.path.join(source["case"], "T_view_meta.json")
    target_meta_path = os.path.join(target["case"], "T_view_meta.json")
    source_meta = {}
    if os.path.exists(source_meta_path):
        with open(source_meta_path, "r", encoding="utf-8") as file:
            source_meta = json.load(file)
    source_meta["patient_id"] = patient_id
    source_meta["frame_id"] = target_frame
    source_meta["shared_initial_pose"] = {
        "source_patient_id": patient_id,
        "source_frame_id": source_frame,
        "source_pose_path": source["initial_pose"],
    }
    with open(target_meta_path, "w", encoding="utf-8") as file:
        json.dump(source_meta, file, indent=2)


def prepare_frames(args, frames, initial_pose_frame):
    """预处理各帧，并将参考帧初始位姿共享给其余帧。"""
    rows = []
    prepared = []
    for frame in frames:
        paths = frame_paths(args.patient_id, frame)
        ensure_dir(paths["logs"])
        missing = missing_source_inputs(case_config.DATA_ROOT, args.patient_id, frame)
        if missing:
            rows.append(
                {
                    "frame": frame,
                    "status": "skipped_missing_source",
                    "missing": "; ".join(f"{name}={path}" for name, path in missing.items()),
                }
            )
            continue
        env = make_environment(args.patient_id, frame, args.weight)
        if not args.skip_existing_case or not os.path.exists(paths["lap_image"]):
            return_code = run_step(
                "01_prepare_frame_data_and_models.py",
                env,
                os.path.join(paths["logs"], "01_prepare_frame_data_and_models.log"),
            )
            if return_code:
                rows.append({"frame": frame, "status": "failed_01", "returncode": return_code})
                continue
        prepared.append(frame)

    if initial_pose_frame not in prepared:
        reference_failure = next(
            (row for row in rows if row.get("frame") == initial_pose_frame),
            None,
        )
        raise ValueError(
            f"Initial-pose reference frame {initial_pose_frame} was not prepared successfully: "
            f"{reference_failure}"
        )
    reference_paths = frame_paths(args.patient_id, initial_pose_frame)
    reference_env = make_environment(args.patient_id, initial_pose_frame, args.weight)
    if args.initial_pose_config:
        config_path = args.initial_pose_config
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Initial-pose config does not exist: {config_path}")
        reference_env["AR_TVIEW_CONFIG"] = config_path
    elif not any(
        name in os.environ
        for name in ("AR_TVIEW_TX", "AR_TVIEW_TY", "AR_TVIEW_TZ", "AR_TVIEW_RX", "AR_TVIEW_RY", "AR_TVIEW_RZ")
    ):
        print(
            "[WARN] No --initial-pose-config or AR_TVIEW_* override was provided. "
            "Script 03 will use its fallback pose; inspect the reference-frame debug image."
        )
    if not args.skip_existing_case or not os.path.exists(reference_paths["initial_pose"]):
        return_code = run_step(
            "03_save_initial_pose_tview.py",
            reference_env,
            os.path.join(reference_paths["logs"], "03_save_initial_pose_tview.log"),
        )
        if return_code:
            raise RuntimeError(
                f"Script 03 failed for initial-pose reference frame {initial_pose_frame}."
            )

    for frame in prepared:
        if frame != initial_pose_frame:
            copy_shared_initial_pose(args.patient_id, initial_pose_frame, frame)
        rows.append(
            {
                "frame": frame,
                "status": "prepared",
                "initial_pose_frame": initial_pose_frame,
            }
        )
    return rows


def annotate_frames(args, frames):
    """依次标注各帧；复用已有有效标注。"""
    rows = []
    for frame in frames:
        paths = frame_paths(args.patient_id, frame)
        ensure_dir(paths["logs"])
        if not os.path.exists(paths["lap_image"]):
            rows.append({"frame": frame, "status": "needs_prepare"})
            continue
        if not args.force_mask:
            try:
                _, summary = preflight_case_masks(paths["case"], paths["lap_image"])
                write_text(os.path.join(paths["logs"], "mask_preflight.log"), f"[OK] {summary}\n")
                rows.append({"frame": frame, "status": "mask_exists"})
                continue
            except Exception:
                pass
        env = make_environment(args.patient_id, frame, args.weight)
        return_code = run_interactive_step(
            "04_annotate_full_liver_outer_mask.py",
            env,
            ["--force"] if args.force_mask else None,
        )
        rows.append(
            {
                "frame": frame,
                "status": "mask_saved" if return_code == 0 else "failed_04",
                "returncode": return_code,
            }
        )
    return rows


def run_frame(args, frame):
    paths = frame_paths(args.patient_id, frame)
    for name in ("logs", "evaluation", "mask_qa"):
        ensure_dir(paths[name])

    missing = missing_source_inputs(case_config.DATA_ROOT, args.patient_id, frame)
    if missing:
        return {
            "frame": frame,
            "status": "skipped_missing_source",
            "missing": "; ".join(f"{name}={path}" for name, path in missing.items()),
        }, []

    env = make_environment(args.patient_id, frame, args.weight)
    missing_prepared = [
        path for path in (paths["lap_image"], paths["initial_pose"]) if not os.path.exists(path)
    ]
    if missing_prepared:
        return {
            "frame": frame,
            "status": "needs_prepare",
            "missing": "; ".join(missing_prepared),
        }, []

    try:
        quality, summary = preflight_case_masks(paths["case"], paths["lap_image"])
        write_text(os.path.join(paths["logs"], "mask_preflight.log"), f"[OK] {summary}\n")
        with open(os.path.join(paths["mask_qa"], "validated_quality.json"), "w", encoding="utf-8") as file:
            json.dump(quality, file, indent=2)
    except Exception as exc:
        write_text(os.path.join(paths["logs"], "mask_preflight.log"), f"[FAIL] {exc}\n")
        return {
            "frame": frame,
            "status": "needs_mask_annotation",
            "missing": str(exc),
        }, []

    if not args.skip_existing_depth or not os.path.exists(paths["depth"]):
        return_code = run_step(
            "07_generate_depth_for_real_frame.py",
            env,
            os.path.join(paths["logs"], "07_generate_depth_for_real_frame.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_05", "returncode": return_code}, []

    if not args.skip_existing_infer or not os.path.exists(paths["final_stl"]):
        return_code = run_step(
            "09_infer_pose_and_export_stl.py",
            env,
            os.path.join(paths["logs"], "09_infer_pose_and_export_stl_driver.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_07", "returncode": return_code}, []

    try:
        return evaluate_frame(paths, case_config.DATA_ROOT, args.patient_id, frame)
    except Exception as exc:
        write_text(os.path.join(paths["logs"], "evaluation.log"), f"[FAIL] {exc}\n")
        return {"frame": frame, "status": "failed_evaluation", "missing": str(exc)}, []


def main():
    parser = argparse.ArgumentParser(
        description="按阶段批量预处理、标注、推理并执行 Python TRE/IC 评估。"
    )
    parser.add_argument("--patient-id", default="Patient1", help="病例标识，例如 Patient1")
    parser.add_argument(
        "--frames",
        default="all",
        help="帧范围，例如 all、02-10 或 02,03,06",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "annotate", "run", "all"),
        default="run",
        help="prepare=批量预处理；annotate=依次标注；run=深度/推理/评估；all=依次执行全部阶段",
    )
    parser.add_argument(
        "--initial-pose-frame",
        help="共享初始位姿的参考帧；默认使用所选帧中的第一帧",
    )
    parser.add_argument(
        "--initial-pose-config",
        help="参考帧初始位姿 JSON；未指定时使用 AR_TVIEW_* 或 script 03 fallback",
    )
    parser.add_argument(
        "--weight",
        default=case_config.refinenet_weight_path(),
        help="用于推理的 RefineNet checkpoint 路径",
    )
    parser.add_argument("--skip-existing-depth", action="store_true", help="复用已有真实帧深度")
    parser.add_argument("--skip-existing-case", action="store_true", help="复用已有预处理病例数据")
    parser.add_argument("--skip-existing-infer", action="store_true", help="复用已有推理结果")
    parser.add_argument("--force-mask", action="store_true", help="重新标注并覆盖已有人工 mask")
    args = parser.parse_args()

    frames = parse_frames(args.frames, case_config.DATA_ROOT, args.patient_id)
    require_python_modules(
        args.stage,
        skip_existing_depth=args.skip_existing_depth,
        skip_existing_infer=args.skip_existing_infer,
    )
    initial_pose_frame = frame_id(args.initial_pose_frame) if args.initial_pose_frame else frames[0]
    if initial_pose_frame not in frames:
        raise ValueError(
            f"Initial-pose reference frame {initial_pose_frame} is not in selected frames: {frames}"
        )
    if args.stage in ("run", "all") and not args.skip_existing_infer and not os.path.exists(args.weight):
        raise FileNotFoundError(f"RefineNet weight does not exist: {args.weight}")

    batch_dir = os.path.join(
        case_config.RESULT_ROOT,
        args.patient_id,
        f"batch_{frames[0]}_{frames[-1]}",
    )
    ensure_dir(batch_dir)
    summary_rows = []
    all_debug_rows = []

    batch_config = {
        "patient_id": args.patient_id,
        "frames": frames,
        "stage": args.stage,
        "initial_pose_frame": initial_pose_frame,
        "initial_pose_config": args.initial_pose_config,
        "weight": args.weight,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(batch_dir, "batch_config.json"), "w", encoding="utf-8") as file:
        json.dump(batch_config, file, indent=2)

    if args.stage in ("prepare", "all"):
        print(f"=== Prepare {args.patient_id}: {','.join(frames)} ===")
        prepare_rows = prepare_frames(args, frames, initial_pose_frame)
        write_rows(
            os.path.join(batch_dir, "prepare_summary.csv"),
            prepare_rows,
            ["frame", "status", "initial_pose_frame", "returncode", "missing"],
        )
        summary_rows.extend(prepare_rows)

    if args.stage in ("annotate", "all"):
        print(f"=== Annotate masks {args.patient_id}: {','.join(frames)} ===")
        annotation_rows = annotate_frames(args, frames)
        write_rows(
            os.path.join(batch_dir, "annotation_summary.csv"),
            annotation_rows,
            ["frame", "status", "returncode"],
        )
        summary_rows.extend(annotation_rows)

    if args.stage not in ("run", "all"):
        write_rows(
            os.path.join(batch_dir, "stage_summary.csv"),
            summary_rows,
            ["frame", "status", "initial_pose_frame", "returncode", "missing"],
        )
        print(f"[OK] batch stage results: {batch_dir}")
        return

    if args.stage in ("run", "all"):
        summary_rows = []
        for frame in frames:
            print(f"=== Run frame {args.patient_id}/{frame} ===")
            row, eval_rows = run_frame(args, frame)
            summary_rows.append(row)
            for eval_row in eval_rows:
                all_debug_rows.append(
                    {
                        "frame": frame,
                        "iter": eval_row["iter"],
                        "TRE_mm": f"{float(eval_row['TRE_mm']):.10f}",
                        "IC": eval_row["IC"],
                        "file": eval_row["file"],
                        "script_selected": int(
                            row.get("status") == "ok"
                            and int(eval_row["iter"]) == int(row["selected_iter"])
                        ),
                    }
                )
            print(f"[{row['status']}] {frame}")

    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    tre_values = [float(row["selected_TRE_mm"]) for row in ok_rows]
    ic_values = [int(row["selected_IC"]) for row in ok_rows]
    average = {
        "num_requested_frames": len(frames),
        "num_ok_frames": len(ok_rows),
        "mean_selected_TRE_mm": float(sum(tre_values) / len(tre_values)) if tre_values else None,
        "ic_pass_count": int(sum(ic_values)),
        "ic_pass_rate": float(sum(ic_values) / len(ic_values)) if ic_values else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "initial_pose_frame": initial_pose_frame,
        "stage": args.stage,
        "note": (
            "The average uses only script-selected final outputs. "
            "Post-hoc best iterations are reported but never used for inference selection."
        ),
    }
    summary_fields = [
        "frame",
        "status",
        "selected_iter",
        "stop_reason",
        "selected_TRE_mm",
        "selected_IC",
        "posthoc_best_iter",
        "posthoc_best_TRE_mm",
        "posthoc_best_IC",
        "selected_is_posthoc_best",
        "returncode",
        "missing",
    ]
    write_rows(os.path.join(batch_dir, "summary.csv"), summary_rows, summary_fields)
    write_rows(
        os.path.join(batch_dir, "debug_all_frames.csv"),
        all_debug_rows,
        ["frame", "iter", "TRE_mm", "IC", "file", "script_selected"],
    )
    write_rows(
        os.path.join(batch_dir, "incomplete_frames.csv"),
        [row for row in summary_rows if row.get("status") != "ok"],
        ["frame", "status", "returncode", "missing"],
    )
    with open(os.path.join(batch_dir, "average.json"), "w", encoding="utf-8") as file:
        json.dump(average, file, indent=2)
    print(f"[OK] batch results: {batch_dir}")


if __name__ == "__main__":
    main()
