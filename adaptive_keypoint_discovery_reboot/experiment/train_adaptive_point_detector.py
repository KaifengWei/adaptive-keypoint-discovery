#!/usr/bin/env python
"""Train a variable-count DINOv2 heatmap head from automatic pseudo labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_dinov2_feasibility as g1  # noqa: E402
from adaptive_point_model import AdaptivePointDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gaussian_heatmap(
    points: list[tuple[float, float, float]], output_hw: tuple[int, int], input_hw: tuple[int, int], sigma: float
) -> np.ndarray:
    output_h, output_w = output_hw
    input_h, input_w = input_hw
    heatmap = np.zeros((output_h, output_w), dtype=np.float32)
    radius = max(2, math.ceil(3.0 * sigma))
    for x_input, y_input, confidence in points:
        x = x_input * (output_w - 1) / max(input_w - 1, 1)
        y = y_input * (output_h - 1) / max(input_h - 1, 1)
        x0, x1 = max(0, math.floor(x) - radius), min(output_w, math.floor(x) + radius + 1)
        y0, y1 = max(0, math.floor(y) - radius), min(output_h, math.floor(y) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        patch = confidence * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch.astype(np.float32))
    return heatmap


class PseudoPointDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(self, rows: list[dict[str, Any]], config: dict[str, Any], augment: bool) -> None:
        self.rows = rows
        self.config = config
        self.augment = augment
        self.size = int(config["image_size"])
        self.output_stride = int(config.get("output_stride", 4))
        self.sigma = float(config.get("gaussian_sigma_output", 1.6))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        if row.get("image_relative_path"):
            portable_relative = str(row["image_relative_path"]).replace("\\", "/")
            image_path = Path(str(self.config["dataset_root"])) / Path(portable_relative)
        else:
            image_path = Path(str(row.get("image_path", row.get("image_path_at_generation", ""))))
        image, mapping = g1.letterbox_rgb(image_path, self.size)
        points: list[tuple[float, float, float]] = []
        for point in row["points"]:
            x = float(point["x_source"]) * mapping["scale"] + mapping["pad_x"]
            y = float(point["y_source"]) * mapping["scale"] + mapping["pad_y"]
            points.append((x, y, float(point["consensus_confidence"])))

        if self.augment and random.random() < float(self.config.get("horizontal_flip_probability", 0.5)):
            image = np.ascontiguousarray(image[:, ::-1])
            points = [(self.size - 1.0 - x, y, confidence) for x, y, confidence in points]
        if self.augment:
            low, high = self.config.get("brightness_range", [0.85, 1.15])
            factor = random.uniform(float(low), float(high))
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        normalized = image.astype(np.float32) / 255.0
        normalized = (normalized - g1.IMAGENET_MEAN) / g1.IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        output_hw = (self.size // self.output_stride, self.size // self.output_stride)
        target = gaussian_heatmap(points, output_hw, (self.size, self.size), self.sigma)
        return tensor, torch.from_numpy(target[None]), torch.tensor(float(len(points))), str(row["dataset_id"])


def deterministic_split(rows: list[dict[str, Any]], fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows, validation_rows = [], []
    threshold = round(10000 * fraction)
    for row in rows:
        digest = hashlib.sha256(f"{seed}|{row['dataset_id']}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) % 10000
        (validation_rows if value < threshold else train_rows).append(row)
    if len(rows) >= 3 and not validation_rows:
        validation_rows.append(train_rows.pop())
    if len(rows) >= 2 and not train_rows:
        train_rows.append(validation_rows.pop())
    return train_rows, validation_rows


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, point_count: torch.Tensor, sigma: float) -> tuple[torch.Tensor, dict[str, float]]:
    weights = 1.0 + 80.0 * target
    heatmap = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).mean()
    probability = logits.sigmoid()
    gaussian_mass = max(1e-6, 2.0 * math.pi * sigma**2)
    predicted_count = probability.sum(dim=(1, 2, 3)) / gaussian_mass
    count = F.smooth_l1_loss(predicted_count, point_count)
    total = heatmap + 0.02 * count
    return total, {"heatmap_loss": float(heatmap.detach()), "count_loss": float(count.detach())}


def match_f1(predicted: list[Any], target_heatmap: torch.Tensor, radius: float = 4.0) -> float:
    pooled = F.max_pool2d(target_heatmap[None], 5, stride=1, padding=2)[0]
    mask = (target_heatmap >= pooled) & (target_heatmap >= 0.30)
    target_y, target_x = torch.where(mask[0])
    target = np.column_stack([target_x.cpu().numpy(), target_y.cpu().numpy()]).astype(np.float64)
    if not predicted and not len(target):
        return 1.0
    if not predicted or not len(target):
        return 0.0
    output_h, output_w = target_heatmap.shape[-2:]
    input_h = input_w = 518
    prediction = np.asarray(
        [
            [point.x * (output_w - 1) / (input_w - 1), point.y * (output_h - 1) / (input_h - 1)]
            for point in predicted
        ]
    )
    distances = np.linalg.norm(prediction[:, None] - target[None, :], axis=2)
    matches = 0
    used_prediction: set[int] = set()
    used_target: set[int] = set()
    for flat_index in np.argsort(distances, axis=None):
        i, j = np.unravel_index(flat_index, distances.shape)
        if distances[i, j] > radius:
            break
        if i not in used_prediction and j not in used_target:
            used_prediction.add(i)
            used_target.add(j)
            matches += 1
    precision = matches / max(len(prediction), 1)
    recall = matches / max(len(target), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


@torch.no_grad()
def evaluate(
    model: AdaptivePointDetector,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    losses, f1_values = [], []
    for images, targets, counts, _ in loader:
        images, targets, counts = images.to(device), targets.to(device), counts.to(device)
        logits = model(images)
        loss, _ = heatmap_loss(logits, targets, counts, float(config.get("gaussian_sigma_output", 1.6)))
        losses.append(float(loss))
        decoded = model.decode(
            logits,
            (int(config["image_size"]), int(config["image_size"])),
            threshold=float(config.get("inference_threshold", 0.35)),
            max_points=int(config.get("inference_safety_cap", 64)),
            fixed_k=int(config.get("fixed_k_eval", 0)),
        )
        for batch_index in range(len(decoded)):
            f1_values.append(match_f1(decoded[batch_index], targets[batch_index].cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "pseudo_label_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
    }


def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))

    def resolved_config_path(key: str) -> Path:
        path = Path(str(config[key]))
        return path if path.is_absolute() else (args.config.parent / path).resolve()

    for path_key in ("dataset_root", "pseudo_labels_jsonl", "dinov2_local_repo", "dinov2_weights", "output_dir"):
        if path_key in config:
            config[path_key] = str(resolved_config_path(path_key))
    if args.seed >= 0:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir.resolve())
    seed = int(config.get("seed", 20260717))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is false")
    device = torch.device(requested_device)

    pseudo_path = Path(str(config["pseudo_labels_jsonl"]))
    rows = [row for row in load_jsonl(pseudo_path) if int(row.get("training_usable", 0)) == 1]
    if not rows:
        raise RuntimeError(f"No training-usable pseudo labels in {pseudo_path}")
    train_rows, validation_rows = deterministic_split(rows, float(config.get("validation_fraction", 0.15)), seed)
    train_dataset = PseudoPointDataset(train_rows, config, augment=True)
    validation_dataset = PseudoPointDataset(validation_rows, config, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model_args = argparse.Namespace(
        local_repo=Path(str(config["dinov2_local_repo"])),
        model=str(config.get("dinov2_model", "dinov2_vits14_reg")),
        weights=Path(str(config["dinov2_weights"])),
    )
    backbone = g1.load_official_model(model_args, device)
    model = AdaptivePointDetector(
        backbone,
        patch_size=14,
        decoder_dim=int(config.get("decoder_dim", 192)),
        output_stride=int(config.get("output_stride", 4)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        unfreeze_last_blocks=int(config.get("unfreeze_last_blocks", 0)),
    ).to(device)
    decoder_parameters = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    parameter_groups = [{"params": decoder_parameters, "lr": float(config.get("head_learning_rate", 1e-3))}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": float(config.get("backbone_learning_rate", 1e-5))})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(config.get("weight_decay", 0.01)))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        images, targets, counts, ids = next(iter(train_loader))
        with torch.no_grad():
            logits = model(images.to(device))
        result = {
            "status": "dry_run_ok",
            "device": str(device),
            "batch_ids": list(ids),
            "image_shape": list(images.shape),
            "target_shape": list(targets.shape),
            "logit_shape": list(logits.shape),
            "train_images": len(train_rows),
            "validation_images": len(validation_rows),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
        (output / "dry_run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    global_step = 0
    started = time.time()
    for epoch in range(1, int(config.get("epochs", 80)) + 1):
        model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        accumulation = int(config.get("gradient_accumulation_steps", 1))
        for batch_index, (images, targets, counts, _) in enumerate(train_loader, start=1):
            images, targets, counts = images.to(device), targets.to(device), counts.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss, parts = heatmap_loss(logits, targets, counts, float(config.get("gaussian_sigma_output", 1.6)))
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach()))
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        validation = evaluate(model, validation_loader, device, config)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation["loss"],
            "validation_pseudo_label_f1": validation["pseudo_label_f1"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, encoding="utf-8-sig")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation": validation,
                },
                output / "best.pt",
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    summary = {
        "status": "complete",
        "epochs_completed": len(history),
        "global_steps": global_step,
        "best_validation_loss": best_loss,
        "training_images": len(train_rows),
        "validation_images": len(validation_rows),
        "manual_keypoint_labels_used": False,
        "target_source": "cross-transform G1-prime automatic consensus",
        "elapsed_seconds": time.time() - started,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
