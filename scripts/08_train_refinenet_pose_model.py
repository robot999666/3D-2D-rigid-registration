"""使用脚本 05、06 生成的配对与深度训练 RefineNet。

训练集应用随机增强，验证集使用确定性预处理。"""

import os
import sys
import json
import logging
import random
import cv2
import open3d as o3d
import numpy as np

import torch
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "modules")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from learning.models.refine_network import RefineNet

from shared import case_config as _cc
from shared.depth_augment import augment_depth, depth_for_network_eval


class Config:
    """训练配置；AR_RESUME=1 时从 last.pth 恢复训练。"""

    PATIENT_ID = _cc.PATIENT_ID
    FRAME_ID = _cc.FRAME_ID
    CASE_ROOT = _cc.CASE_ROOT

    LOG_DIR = _cc.logs_dir()
    CHECKPOINT_DIR = _cc.checkpoints_dir()
    TRAINING_DIR = _cc.training_dir()

    BATCH_SIZE = int(os.environ.get("AR_BATCH_SIZE", "32"))
    ACCUMULATION_STEPS = 1
    LR = 1e-4
    NUM_EPOCHS = int(os.environ.get("AR_NUM_EPOCHS", "50"))
    NUM_WORKERS = int(os.environ.get("AR_NUM_WORKERS", "4"))
    SEED = _cc.SEED
    RESUME = os.environ.get("AR_RESUME", "0").lower() in ("1", "true", "yes")

    # 平移缩放与推理保持一致。
    ROT_REP = "6d"
    TRANS_SCALE = 50.0
    NUM_SAMPLED_PTS = 2000
    IMG_SIZE = _cc.IMG_SIZE

    # 训练用轮廓和掩码增强。
    CONTOUR_ELASTIC_ALPHA = float(os.environ.get("AR_CONTOUR_ELASTIC_ALPHA", "10"))
    CONTOUR_ELASTIC_SIGMA = float(os.environ.get("AR_CONTOUR_ELASTIC_SIGMA", "4"))
    MASK_AUG_PROB = float(os.environ.get("AR_MASK_AUG_PROB", "0.5"))


class DummyCfg:
    use_BN = True
    rot_rep = Config.ROT_REP


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("PoseOffsetTrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(o3d, "utility") and hasattr(o3d.utility, "random"):
        try:
            o3d.utility.random.seed(seed)
        except Exception:
            pass


def seed_worker(worker_id):
    worker_seed = Config.SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def plot_loss_curves(train_losses, val_losses, result_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Surface MSE', marker='o')
    plt.plot(val_losses, label='Val Surface MSE', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Surface MSE (mm^2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()


def load_3d_model_points(ply_path, num_points):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.has_vertices():
        raise ValueError(f"无法读取 3D 模型: {ply_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return torch.from_numpy(np.asarray(pcd.points).astype(np.float32))


def zhang_suen_thinning(binary):
    if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'thinning'):
        return zhang_suen_thinning_numpy(binary)
    binary = (binary > 0).astype(np.uint8) * 255
    return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def zhang_suen_thinning_numpy(binary):
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(neighbors)
            transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & cond
            if np.any(remove):
                img[remove] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def elastic_transform(image, alpha, sigma):
    random_state = np.random
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def augment_contour_channel(channel):
    img = (channel > 0).astype(np.uint8) * 255
    img = zhang_suen_thinning(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    iterations = int(np.random.choice([1, 2, 3]))
    img = cv2.dilate(img, kernel, iterations=iterations)

    h, w = img.shape
    num_blocks = np.random.randint(0, 4)
    for _ in range(num_blocks):
        bw, bh = int(0.2 * w), int(0.2 * h)
        x = np.random.randint(0, max(1, w - bw))
        y = np.random.randint(0, max(1, h - bh))
        img[y:y + bh, x:x + bw] = 0

    img = elastic_transform(img, Config.CONTOUR_ELASTIC_ALPHA, Config.CONTOUR_ELASTIC_SIGMA)
    return (img > 0).astype(np.float32)


def augment_contours(contours):
    return np.stack([augment_contour_channel(contours[i]) for i in range(contours.shape[0])], axis=0)


def augment_mask(mask):
    if np.random.rand() > Config.MASK_AUG_PROB:
        return mask.astype(np.float32)
    img = (mask > 0).astype(np.uint8) * 255
    op = np.random.choice([cv2.MORPH_ERODE, cv2.MORPH_DILATE])
    k = np.random.randint(2, 7)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img = cv2.morphologyEx(img, op, kernel, iterations=1)
    return (img > 0).astype(np.float32)


class PoseDataset(Dataset):
    def __init__(self, data_dir, is_train):
        self.img_dir = os.path.join(data_dir, "images")
        with open(os.path.join(data_dir, "labels.json"), "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.keys = sorted(self.labels.keys())
        self.is_train = is_train
        self.pairs = []

        pairs_path = os.path.join(data_dir, "pairs.json")
        if os.path.exists(pairs_path):
            with open(pairs_path, "r", encoding="utf-8") as f:
                raw_pairs = json.load(f)
            if isinstance(raw_pairs, dict):
                self.pairs = [raw_pairs[k] for k in sorted(raw_pairs.keys())]
            elif isinstance(raw_pairs, list):
                self.pairs = raw_pairs
            else:
                raise ValueError(f"Invalid pairs.json format: {pairs_path}")

        self.uses_explicit_pairs = len(self.pairs) > 0
        if self.uses_explicit_pairs:
            self.pair_type_counts = {}
            for pair in self.pairs:
                pair_type = pair.get("pair_type", "unknown")
                self.pair_type_counts[pair_type] = self.pair_type_counts.get(pair_type, 0) + 1
        else:
            self.pair_type_counts = {"random_pairs_legacy": len(self.keys)}

    def __len__(self):
        return len(self.pairs) if self.uses_explicit_pairs else len(self.keys)

    def _load_state(self, prefix):
        contours = load_contour_array(os.path.join(self.img_dir, f"{prefix}_contours.npy"))
        depth_path = os.path.join(self.img_dir, f"{prefix}_depth.npy")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(
                f"Missing {depth_path}. Run scripts/06_generate_depth_for_training_samples.py after scripts/05_render_training_pose_pairs.py."
            )
        depth = np.load(depth_path).astype(np.float32)
        mask = cv2.imread(os.path.join(self.img_dir, f"{prefix}_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        depth[mask <= 0] = 0.0

        target_size = Config.IMG_SIZE
        contours_rs = np.stack([
            cv2.resize(contours[i], target_size, interpolation=cv2.INTER_NEAREST) for i in range(3)
        ], axis=0)
        depth_rs = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        mask_rs = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            contours_rs = augment_contours(contours_rs)
            mask_rs = augment_mask(mask_rs)
            depth_rs = augment_depth(depth_rs, mask_rs)
        else:
            depth_rs = depth_for_network_eval(depth_rs, mask_rs)

        tensor = np.concatenate([contours_rs, depth_rs[None, ...], mask_rs[None, ...]], axis=0)
        return torch.from_numpy(tensor.astype(np.float32))

    def __getitem__(self, idx):
        if self.uses_explicit_pairs:
            pair = self.pairs[idx]
            prefix_A = pair["A"]
            prefix_B = pair["B"]
            pair_type = pair.get("pair_type", "unknown")
            if prefix_A not in self.labels or prefix_B not in self.labels:
                raise KeyError(f"Pair references missing state: A={prefix_A}, B={prefix_B}")
        else:
            prefix_B = self.keys[idx]
            pair_type = "random_pairs_legacy"
            prefix_A = self.keys[np.random.randint(0, len(self.keys))]
            while prefix_A == prefix_B and len(self.keys) > 1:
                prefix_A = self.keys[np.random.randint(0, len(self.keys))]

        tensor_A = self._load_state(prefix_A)
        tensor_B = self._load_state(prefix_B)
        gt_T_A = np.array(self.labels[prefix_A]["Delta_T"], dtype=np.float32)
        gt_T_B = np.array(self.labels[prefix_B]["Delta_T"], dtype=np.float32)
        return torch.cat([tensor_A, tensor_B], dim=0), torch.from_numpy(gt_T_A), torch.from_numpy(gt_T_B), pair_type


def load_contour_array(path):
    contours = np.load(path)
    if contours.shape[0] != 3:
        raise ValueError(f"Expected contour array with 3 channels: {path}, got {contours.shape}.")
    contours = contours.astype(np.float32)
    if (float(np.max(contours)) if contours.size else 0.0) > 1.0:
        contours /= 255.0
    return contours


def compute_rotation_matrix_from_6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = F.normalize(x_raw, p=2, dim=-1)
    z_dot_x = (y_raw * x).sum(dim=-1, keepdim=True)
    y = F.normalize(y_raw - z_dot_x * x, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def compute_surface_mse_loss(
    pred_trans,
    pred_rot_6d,
    gt_T_A,
    gt_T_B,
    model_points,
    trans_scale=50.0,
    reduction="mean",
):
    B = pred_trans.shape[0]
    R_rel_pred = compute_rotation_matrix_from_6d(pred_rot_6d)
    t_rel_pred = (pred_trans * trans_scale).unsqueeze(2)

    R_A = gt_T_A[:, :3, :3]
    t_A = gt_T_A[:, :3, 3:4]
    R_B = gt_T_B[:, :3, :3]
    t_B = gt_T_B[:, :3, 3:4]

    R_est = torch.bmm(R_A, R_rel_pred)
    t_est = torch.bmm(R_A, t_rel_pred) + t_A

    pts_obj = model_points.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)
    pts_est = torch.bmm(R_est, pts_obj) + t_est
    pts_gt = torch.bmm(R_B, pts_obj) + t_B

    sq_dist = torch.sum((pts_est - pts_gt) ** 2, dim=1)
    per_sample = sq_dist.mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction != "mean":
        raise ValueError(f"Unknown reduction: {reduction}")
    return per_sample.mean()


def main():
    cfg = Config()
    set_global_seed(cfg.SEED)
    case_dir = os.path.join(cfg.CASE_ROOT, cfg.PATIENT_ID, cfg.FRAME_ID)
    train_dir = os.path.join(case_dir, "sample", "train")
    val_dir = os.path.join(case_dir, "sample", "val")
    liver_path = os.path.join(case_dir, "models", "Liver_obj.ply")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.TRAINING_DIR, exist_ok=True)
    logger = setup_logger(cfg.LOG_DIR)
    logger.info("=== 08. Train rigid pose offset network ===")
    logger.info("Seed: %d", cfg.SEED)
    logger.info("Checkpoints: %s", cfg.CHECKPOINT_DIR)
    logger.info("Training artifacts: %s", cfg.TRAINING_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PoseDataset(train_dir, is_train=True)
    val_dataset = PoseDataset(val_dir, is_train=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            "Train and validation datasets must both be non-empty. "
            "Regenerate samples with scripts/05_render_training_pose_pairs.py."
        )
    if train_dataset.uses_explicit_pairs != val_dataset.uses_explicit_pairs:
        raise RuntimeError("Train/val pair mode mismatch. Regenerate both splits with 05_render_training_pose_pairs.py.")

    dataset_mode = "explicit_pairs" if train_dataset.uses_explicit_pairs else "random_pairs_legacy"
    logger.info(
        "Dataset mode: %s | train states=%d pairs=%d | val states=%d pairs=%d",
        dataset_mode,
        len(train_dataset.labels),
        len(train_dataset),
        len(val_dataset.labels),
        len(val_dataset),
    )
    logger.info("Train pair types: %s", train_dataset.pair_type_counts)
    logger.info("Val pair types: %s", val_dataset.pair_type_counts)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    model_points = load_3d_model_points(liver_path, cfg.NUM_SAMPLED_PTS).to(device)
    model = RefineNet(cfg=DummyCfg(), c_in=5).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.LR)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    last_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "last.pth")
    logger.info("Resume requested: %s", cfg.RESUME)
    if cfg.RESUME and os.path.exists(last_ckpt_path):
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        ckpt_mode = checkpoint.get('config', {}).get('dataset_mode')
        legacy_random_ckpt = ckpt_mode is None and dataset_mode == "random_pairs_legacy"
        if ckpt_mode == dataset_mode or legacy_random_ckpt:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint.get('scaler_state', scaler.state_dict()))
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            logger.info("[Resume] loaded checkpoint: %s", last_ckpt_path)
        else:
            logger.warning(
                "[Resume] skipped checkpoint because dataset mode changed: checkpoint=%s current=%s",
                ckpt_mode or "legacy_unknown",
                dataset_mode,
            )
    elif cfg.RESUME:
        logger.warning("[Resume] requested but checkpoint does not exist: %s", last_ckpt_path)

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        logger.info(f"--- Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] LR={optimizer.param_groups[0]['lr']:.2e} ---")
        model.train()
        epoch_train_loss = 0.0
        epoch_train_count = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, gt_T_A, gt_T_B, pair_types) in enumerate(pbar):
            inputs, gt_T_A, gt_T_B = inputs.to(device), gt_T_A.to(device), gt_T_B.to(device)
            A_inputs = inputs[:, :5]
            B_inputs = inputs[:, 5:]
            with torch.cuda.amp.autocast():
                out = model(A_inputs, B_inputs)
                loss = compute_surface_mse_loss(out['trans'], out['rot'], gt_T_A, gt_T_B, model_points, cfg.TRANS_SCALE)
                scaled_loss = loss / cfg.ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()
            if ((batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            batch_size = inputs.shape[0]
            epoch_train_loss += loss.item() * batch_size
            epoch_train_count += batch_size
            pbar.set_postfix({'SurfaceMSE': f"{loss.item():.2f}"})
        avg_train_loss = epoch_train_loss / max(1, epoch_train_count)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_count = 0
        val_loss_by_type = {}
        val_count_by_type = {}
        with torch.no_grad():
            for inputs, gt_T_A, gt_T_B, pair_types in tqdm(val_loader, desc="Validation"):
                inputs, gt_T_A, gt_T_B = inputs.to(device), gt_T_A.to(device), gt_T_B.to(device)
                with torch.cuda.amp.autocast():
                    out = model(inputs[:, :5], inputs[:, 5:])
                    per_sample_loss = compute_surface_mse_loss(
                        out['trans'],
                        out['rot'],
                        gt_T_A,
                        gt_T_B,
                        model_points,
                        cfg.TRANS_SCALE,
                        reduction="none",
                    )
                epoch_val_loss += float(per_sample_loss.sum().item())
                epoch_val_count += inputs.shape[0]
                for pair_type, loss_value in zip(pair_types, per_sample_loss.detach().cpu().tolist()):
                    val_loss_by_type[pair_type] = val_loss_by_type.get(pair_type, 0.0) + float(loss_value)
                    val_count_by_type[pair_type] = val_count_by_type.get(pair_type, 0) + 1
        avg_val_loss = epoch_val_loss / max(1, epoch_val_count)
        logger.info(f"[Result] Train SurfaceMSE: {avg_train_loss:.4f} | Val SurfaceMSE: {avg_val_loss:.4f}")
        for pair_type in sorted(val_loss_by_type.keys()):
            type_avg = val_loss_by_type[pair_type] / max(1, val_count_by_type[pair_type])
            logger.info(
                "[Val:%s] SurfaceMSE: %.4f | count=%d",
                pair_type,
                type_avg,
                val_count_by_type[pair_type],
            )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        plot_loss_curves(train_losses, val_losses, cfg.TRAINING_DIR)
        scheduler.step(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        checkpoint_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'batch_size': cfg.BATCH_SIZE,
                'accumulation_steps': cfg.ACCUMULATION_STEPS,
                'lr': cfg.LR,
                'num_epochs': cfg.NUM_EPOCHS,
                'optimizer': 'Adam',
                'img_size': cfg.IMG_SIZE,
                'dataset_mode': dataset_mode,
                'train_pairs': len(train_dataset),
                'val_pairs': len(val_dataset),
                'feature_token_grid': os.environ.get("AR_FEATURE_TOKEN_GRID", ""),
                'contour_elastic_alpha': cfg.CONTOUR_ELASTIC_ALPHA,
                'contour_elastic_sigma': cfg.CONTOUR_ELASTIC_SIGMA,
                'mask_aug_prob': cfg.MASK_AUG_PROB,
            },
        }
        torch.save(checkpoint_state, last_ckpt_path)
        logger.info(f"[Checkpoint] saved/resumable: {last_ckpt_path}")
        if is_best:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best.pth"))
            torch.save(checkpoint_state, os.path.join(cfg.CHECKPOINT_DIR, "best_full.pth"))
            logger.info(f"[*] Best checkpoint saved. Val SurfaceMSE={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
