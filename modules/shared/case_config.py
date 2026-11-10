"""统一管理病例、路径、图像尺寸、随机种子和模型权重。

环境变量覆盖默认值；相对路径以进程工作目录为基准。"""

import os


def _project_default():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


# 编号脚本与统一入口共用交付配置；显式环境变量优先。
_profile = os.path.join(_project_default(), "configs", "patient1_fullres.env")
with open(_profile, encoding="utf-8") as _config:
    for _line in _config:
        if _line.strip() and not _line.lstrip().startswith("#"):
            _key, _value = _line.strip().split("=", 1)
            os.environ.setdefault(_key.strip(), _value.strip())

# 病例和帧选择
PATIENT_ID = os.environ.get("PATIENT_ID", "Patient1")
FRAME_ID = os.environ.get("FRAME_ID", "02")


# 标准根目录
PROJECT_ROOT = os.environ.get("AR_PROJECT_ROOT", _project_default())
CASE_ROOT = os.environ.get("CASE_ROOT", os.path.join(PROJECT_ROOT, "data/processed"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "data/raw"))
RESULT_ROOT = os.environ.get("RESULT_ROOT", os.path.join(PROJECT_ROOT, "results", "run"))
WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", os.path.join(PROJECT_ROOT, "weights"))

# 公开数据的轮廓标注者子目录。
ANNOTATOR = os.environ.get("AR_ANNOTATOR", "koo")


# 共享图像和模型输入契约
# 所有阶段共用工作分辨率。
WORK_WIDTH = int(os.environ.get("AR_WORK_WIDTH", "1920"))
WORK_HEIGHT = int(os.environ.get("AR_WORK_HEIGHT", "1080"))
IMG_SIZE = (WORK_WIDTH, WORK_HEIGHT)

# 真实轮廓与渲染轮廓共用线宽。
CONTOUR_THICKNESS = int(os.environ.get("AR_CONTOUR_THICKNESS", "2"))

# 用于样本生成、数据划分、训练及 Open3D 采样。
SEED = int(os.environ.get("AR_SEED", "42"))


# Depth Anything V2
DA2_PROJECT_DIR = os.environ.get("DA2_PROJECT_DIR", os.path.join(PROJECT_ROOT, "modules"))
DA2_ENCODER = os.environ.get("DA2_ENCODER", "vitl")
DA2_INPUT_SIZE = int(os.environ.get("DA2_INPUT_SIZE", "518"))
DA2_CHECKPOINT_PATH = os.environ.get(
    "DA2_CHECKPOINT_PATH",
    os.path.join(WEIGHTS_ROOT, f"depth_anything_v2_{DA2_ENCODER}.pth"),
)


# RefineNet checkpoint 回退路径
# 权重查找顺序：AR_WEIGHT_PATH → 当前帧 checkpoint → 固定权重。
REFINENET_PRETRAINED_PATH = os.environ.get(
    "AR_PRETRAINED_WEIGHT_PATH",
    os.path.join(WEIGHTS_ROOT, "refinenet_patient1_fullres_best.pth"),
)


def case_dir(case_root=None, patient_id=None, frame_id=None):
    """返回病例帧目录。"""
    return os.path.join(
        case_root or CASE_ROOT,
        patient_id or PATIENT_ID,
        frame_id or FRAME_ID,
    )


def depth_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "depth")


def image_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "image")


def contour_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "contours")


def mask_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "masks")


def result_dir(result_root=None, patient_id=None, frame_id=None):
    """返回病例帧结果目录。"""
    return os.path.join(
        result_root or RESULT_ROOT,
        patient_id or PATIENT_ID,
        frame_id or FRAME_ID,
    )


def logs_dir(result_root=None, patient_id=None, frame_id=None):
    """返回日志目录。"""
    return os.path.join(result_dir(result_root, patient_id, frame_id), "logs")


def checkpoints_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "checkpoints")


def training_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "training")


def inference_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "inference")


def evaluation_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "evaluation")


def mask_qa_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "mask_qa")


def refinenet_weight_path(result_root=None, patient_id=None, frame_id=None):
    """按显式路径、当前帧 checkpoint、预训练权重的顺序查找。"""
    explicit = os.environ.get("AR_WEIGHT_PATH")
    if explicit:
        return explicit
    trained = os.path.join(
        checkpoints_dir(result_root, patient_id, frame_id),
        "best.pth",
    )
    return trained if os.path.exists(trained) else REFINENET_PRETRAINED_PATH
