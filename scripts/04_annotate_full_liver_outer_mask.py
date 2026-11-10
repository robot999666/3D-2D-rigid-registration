"""交互标注真实帧的肝脏外轮廓，并保存掩码与质量记录。"""

import argparse
import json
import os
import sys
import textwrap

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.intraop_masks import MaskValidationError, mask_paths, quality_summary, save_mask_bundle


WINDOW_NAME = "Intra-op full liver outer mask"
INFO_PANEL_WIDTH = 360
DISPLAY_SCALE = max(0.05, float(os.environ.get("AR_MASK_DISPLAY_SCALE", "1.0")))


def polygon_mask(shape, polygon):
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


class MaskAnnotator:
    def __init__(self, image):
        self.image = image
        self.shape = image.shape[:2]
        self.display_scale = DISPLAY_SCALE
        self.stage = "full_liver"
        self.full_polygon = []
        self.cursor = None
        self.message = "Draw the complete outer liver boundary; Enter closes it."
        self.saved = False

    def on_mouse(self, event, x, y, _flags, _param):
        display_w = max(1, int(round(self.shape[1] * self.display_scale)))
        display_h = max(1, int(round(self.shape[0] * self.display_scale)))
        if x < 0 or y < 0 or x >= display_w or y >= display_h:
            return
        x = int(np.clip(round(x / self.display_scale), 0, self.shape[1] - 1))
        y = int(np.clip(round(y / self.display_scale), 0, self.shape[0] - 1))
        self.cursor = (x, y)
        if self.stage == "review":
            self.message = "Press B to edit the polygon, or S to save."
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.full_polygon.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.full_polygon:
            self.full_polygon.pop()

    def close_current(self):
        if len(self.full_polygon) < 3:
            self.message = "Full liver outer polygon needs at least 3 points."
            return
        self.stage = "review"
        self.message = "Outer liver polygon closed. Press S to save, B to edit, or R to restart."

    def reset_current(self):
        self.stage = "full_liver"
        self.full_polygon.clear()
        self.message = "Polygon reset. Draw the complete outer liver boundary."

    def edit_polygon(self):
        self.stage = "full_liver"
        self.message = "Editing polygon. Left click adds points; right click removes last point."

    def build_masks(self):
        full = polygon_mask(self.shape, self.full_polygon)
        # 深度有效区使用同一人工外轮廓。
        return full, full.copy()

    def render(self):
        image_canvas = self.image.copy()
        full = polygon_mask(self.shape, self.full_polygon)

        tint = image_canvas.copy()
        tint[full > 0] = (60, 150, 60)
        image_canvas = cv2.addWeighted(image_canvas, 0.72, tint, 0.28, 0.0)

        self._draw_polyline(image_canvas, self.full_polygon, (0, 255, 0), self.stage == "review")

        if self.display_scale != 1.0:
            display_size = (
                max(1, int(round(self.shape[1] * self.display_scale))),
                max(1, int(round(self.shape[0] * self.display_scale))),
            )
            image_canvas = cv2.resize(image_canvas, display_size, interpolation=cv2.INTER_AREA)

        panel = np.full((image_canvas.shape[0], INFO_PANEL_WIDTH, 3), 28, dtype=np.uint8)
        lines = [
            f"Stage: {self.stage}",
            "",
            "Left click: add point",
            "Right click: undo point",
            "Enter: close polygon",
            "B: edit closed polygon",
            "R: restart polygon",
            "S: validate and save",
            "Esc/Q: cancel",
            "",
            "Mask policy:",
            "One outer liver mask only.",
            "No separate instrument removal.",
            "",
        ]
        lines.extend(textwrap.wrap(self.message, width=42) or [""])
        y = 20
        for line in lines:
            cv2.putText(
                panel,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 18
        return np.hstack([image_canvas, panel])

    @staticmethod
    def _draw_polyline(canvas, points, color, closed):
        if not points:
            return
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], closed, color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)


def load_existing_annotation(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively annotate one complete outer liver mask for a frame."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing annotation")
    args = parser.parse_args()

    case_dir = case_config.case_dir()
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Run scripts/01_prepare_frame_data_and_models.py first. Missing: {image_path}")

    paths = mask_paths(case_dir)
    existing = load_existing_annotation(paths["annotation"])
    if existing and not args.force:
        raise FileExistsError(
            f"Annotation already exists: {paths['annotation']}. "
            "Use --force only after reviewing the existing overlay."
        )

    annotator = MaskAnnotator(image)
    if existing:
        annotator.full_polygon = [tuple(x) for x in existing.get("full_liver_polygon_xy", [])]
        annotator.stage = "review" if len(annotator.full_polygon) >= 3 else "full_liver"
        annotator.message = "Loaded existing annotation. Review and press S to save."

    print("=== Interactive full liver outer mask annotation ===")
    print(f"Frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")
    print(f"Display scale: {DISPLAY_SCALE:.3f}; saved masks keep full prepared resolution.")
    print("Draw one COMPLETE outer liver boundary.")
    print("Do not separately remove instrument occlusions; depth_valid_mask will equal full_liver_mask.")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable in this environment. "
            "This usually means the Docker image uses opencv-python-headless. "
            "Run annotation in a GUI-enabled Python/OpenCV environment, or rebuild the container with HighGUI support."
        ) from exc

    cv2.setMouseCallback(WINDOW_NAME, annotator.on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, annotator.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):
                annotator.close_current()
            elif key in (ord("r"), ord("R")):
                annotator.reset_current()
            elif key in (ord("b"), ord("B")):
                annotator.edit_polygon()
            elif key in (ord("s"), ord("S")):
                if annotator.stage != "review":
                    annotator.close_current()
                    if annotator.stage != "review":
                        continue
                full, depth_valid = annotator.build_masks()
                annotation = {
                    "patient_id": case_config.PATIENT_ID,
                    "frame_id": case_config.FRAME_ID,
                    "image_path": image_path,
                    "mask_policy": "single_outer_liver_mask_no_separate_occlusion",
                    "full_liver_polygon_xy": [list(x) for x in annotator.full_polygon],
                    "depth_valid_equals_full_liver_mask": True,
                    "occlusion_polygons_xy": [],
                }
                try:
                    saved_paths, quality = save_mask_bundle(case_dir, image, full, depth_valid, annotation)
                except MaskValidationError as exc:
                    annotator.message = str(exc)
                    print(f"[INVALID] {exc}")
                    continue
                print(f"[OK] {quality_summary(quality)}")
                for name, path in saved_paths.items():
                    if name != "root":
                        print(f"[OK] {name}: {path}")
                annotator.saved = True
                break
            elif key in (27, ord("q"), ord("Q")):
                break
    finally:
        cv2.destroyAllWindows()

    if not annotator.saved:
        raise RuntimeError("Annotation cancelled; no masks were written.")


if __name__ == "__main__":
    main()
