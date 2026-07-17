#!/usr/bin/env python3
"""G1: training-free feasibility gate with a frozen official DINOv2 backbone.

This stage does not train a landmark detector and does not use keypoint labels.
It asks whether image-dependent candidate points emerge from DINOv2 features and
whether those points survive known geometric and photometric transformations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.decomposition import PCA


DEFAULT_G0 = Path(r"D:\kp\adaptive_keypoint_discovery_reboot\experiment\data_g0")
DEFAULT_OUTPUT = Path(r"D:\kp\adaptive_keypoint_discovery_reboot\experiment\outputs_g1")
SEED = 20260716
PATCH_SIZE = 14
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
METHOD_TAU_GRID = {
    "feature_local_contrast": [6.0, 7.0, 8.0, 10.0],
    # Attention is extremely sparse and heavy-tailed, so its parameter is a
    # quantile rather than a MAD multiplier. The CSV keeps the generic `tau`
    # column for a single machine-readable schema.
    "cls_to_patch_attention": [0.985, 0.990, 0.993, 0.996],
}
PEAK_METHODS = ("feature_local_contrast", "cls_to_patch_attention")
FEATURE_METHODS = ("feature_local_contrast", "feature_hdbscan_medoid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-root", type=Path, default=DEFAULT_G0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("verify", "smoke", "calibrate", "pilot"), default="verify")
    parser.add_argument("--sizes", type=int, nargs="+", default=[518, 728])
    parser.add_argument("--layers", choices=("last", "last4avg"), nargs="+", default=["last", "last4avg"])
    parser.add_argument("--model", default="dinov2_vits14_reg")
    parser.add_argument("--local-repo", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for debugging.")
    parser.add_argument("--visual-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--full-transforms", action="store_true", help="Smoke phase normally uses four transforms.")
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def partition_pilot(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    def key(row: dict[str, str]) -> str:
        return hashlib.sha256(f"{seed}|{row['image_id']}|{row['sha256']}".encode("utf-8")).hexdigest()

    ordered = sorted(rows, key=key)
    if len(ordered) < 80:
        raise RuntimeError(f"G1 requires the 80-image G0 pilot, found {len(ordered)} rows.")
    return ordered[:20], ordered[20:80]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def configure_python_certificates() -> str | None:
    if os.environ.get("SSL_CERT_FILE"):
        return os.environ["SSL_CERT_FILE"]
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        return certifi.where()
    except Exception:
        return None


def checkpoint_integrity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        signature = handle.read(4)
        digest.update(signature)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "zip_crc_ok": None,
    }
    if signature.startswith(b"PK"):
        with zipfile.ZipFile(path) as archive:
            failed = archive.testzip()
            result["zip_entries"] = len(archive.infolist())
            result["zip_crc_ok"] = failed is None
            result["zip_crc_failure"] = failed
        if failed is not None:
            raise RuntimeError(f"Checkpoint CRC failed at {failed}: {path}")
    return result


def load_official_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    configure_python_certificates()
    if args.local_repo:
        repo = args.local_repo.resolve()
        if not (repo / "hubconf.py").exists():
            raise FileNotFoundError(f"No hubconf.py under local DINOv2 repository: {repo}")
        model = torch.hub.load(str(repo), args.model, source="local", pretrained=args.weights is None)
    else:
        model = torch.hub.load("facebookresearch/dinov2", args.model, trust_repo=True, pretrained=args.weights is None)
    if args.weights:
        checkpoint_integrity(args.weights)
        checkpoint = torch.load(args.weights.resolve(), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("model", "teacher", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        cleaned = {}
        for key, value in checkpoint.items():
            new_key = key
            for prefix in ("module.", "backbone.", "teacher."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            cleaned[new_key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            print(f"WEIGHT_KEYS missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def letterbox_rgb(path: Path, size: int) -> tuple[np.ndarray, dict[str, float]]:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    border = np.concatenate(
        [rgb[: max(1, h // 25)].reshape(-1, 3), rgb[-max(1, h // 25) :].reshape(-1, 3)], axis=0
    )
    fill = tuple(int(x) for x in np.median(border, axis=0))
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "source_width": float(w),
        "source_height": float(h),
    }


def square_plant_mask(row: dict[str, str], size: int, mapping: dict[str, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    mask_path = Path(row["crop_path"]).parent.parent / "audit_masks" / f"{row['image_id']}.png"
    mask_source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_source is None:
        raise FileNotFoundError(mask_path)
    cx1, cy1, cx2, cy2 = (int(row[name]) for name in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
    crop = mask_source[cy1:cy2, cx1:cx2]
    scale, pad_x, pad_y = mapping["scale"], int(mapping["pad_x"]), int(mapping["pad_y"])
    new_w, new_h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0:
        bbox = (0.0, 0.0, float(size - 1), float(size - 1))
    else:
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    return canvas, bbox


def normalize_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    array = image_rgb.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def matrix3(affine2: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2] = affine2
    return matrix


def make_transforms(base: np.ndarray, full: bool) -> list[dict[str, Any]]:
    size = base.shape[0]
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    transforms: list[dict[str, Any]] = [
        {"name": "identity", "family": "identity", "image": base, "matrix": np.eye(3)},
    ]
    flip = np.asarray([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    transforms.append(
        {"name": "flip_horizontal", "family": "geometric", "image": cv2.warpAffine(base, flip, (size, size), borderMode=cv2.BORDER_REFLECT_101), "matrix": matrix3(flip)}
    )
    rot10 = cv2.getRotationMatrix2D(center, 10.0, 1.0)
    transforms.append(
        {"name": "rotate_10", "family": "geometric", "image": cv2.warpAffine(base, rot10, (size, size), borderMode=cv2.BORDER_REFLECT_101), "matrix": matrix3(rot10)}
    )
    dark = np.clip(base.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    transforms.append({"name": "brightness_075", "family": "photometric", "image": dark, "matrix": np.eye(3)})
    if not full:
        return transforms
    rotm10 = cv2.getRotationMatrix2D(center, -10.0, 1.0)
    scale09 = cv2.getRotationMatrix2D(center, 0.0, 0.9)
    scale11 = cv2.getRotationMatrix2D(center, 0.0, 1.1)
    translate = np.asarray([[1.0, 0.0, 0.05 * size], [0.0, 1.0, -0.04 * size]], dtype=np.float64)
    for name, affine in (
        ("rotate_minus10", rotm10),
        ("scale_090", scale09),
        ("scale_110", scale11),
        ("translate", translate),
    ):
        transformed = cv2.warpAffine(base, affine, (size, size), borderMode=cv2.BORDER_REFLECT_101)
        transforms.append({"name": name, "family": "geometric", "image": transformed, "matrix": matrix3(affine)})
    mean = base.mean(axis=(0, 1), keepdims=True)
    contrast = np.clip((base.astype(np.float32) - mean) * 1.25 + mean, 0, 255).astype(np.uint8)
    transforms.append({"name": "contrast_125", "family": "photometric", "image": contrast, "matrix": np.eye(3)})
    return transforms


def last_block_attention_module(model: torch.nn.Module) -> torch.nn.Module | None:
    try:
        block = model.blocks[-1]
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            return block.attn
        # Some configurations wrap blocks in chunks.
        if hasattr(block, "blocks"):
            inner = block.blocks[-1]
            if hasattr(inner, "attn") and hasattr(inner.attn, "qkv"):
                return inner.attn
    except Exception:
        return None
    return None


@torch.inference_mode()
def extract_representations(
    model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    tensor = normalize_tensor(image_rgb, device)
    captured: dict[str, torch.Tensor] = {}
    attention_module = last_block_attention_module(model)
    hook = None
    if attention_module is not None:
        hook = attention_module.qkv.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("qkv", output.detach())
        )
    try:
        try:
            layers = model.get_intermediate_layers(tensor, n=4, reshape=True, return_class_token=True)
        except TypeError:
            layers = model.get_intermediate_layers(tensor, n=4, reshape=True)
    finally:
        if hook is not None:
            hook.remove()

    maps: list[torch.Tensor] = []
    cls_token: torch.Tensor | None = None
    for item in layers:
        if isinstance(item, (tuple, list)):
            patch_map, cls = item[0], item[1]
            cls_token = cls
        else:
            patch_map = item
        if patch_map.ndim != 4:
            raise RuntimeError(f"Unexpected DINOv2 intermediate shape: {tuple(patch_map.shape)}")
        maps.append(F.normalize(patch_map.float(), dim=1))
    last = maps[-1]
    last4avg = F.normalize(torch.stack(maps, dim=0).mean(dim=0), dim=1)

    attention_mode = "exact_last_block_cls_attention"
    if "qkv" in captured and attention_module is not None:
        qkv = captured["qkv"].float()
        batch, tokens, triple_dim = qkv.shape
        heads = int(attention_module.num_heads)
        head_dim = triple_dim // (3 * heads)
        qkv = qkv.reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = float(getattr(attention_module, "scale", head_dim**-0.5))
        weights = (q[:, :, 0:1] * scale) @ k.transpose(-2, -1)
        weights = weights.softmax(dim=-1).mean(dim=1)[0, 0]
        registers = int(getattr(model, "num_register_tokens", 0))
        patch_weights = weights[1 + registers :]
        grid_h, grid_w = last.shape[-2:]
        attention = patch_weights.reshape(grid_h, grid_w)
    else:
        # Honest fallback: this is not called attention in the output metadata.
        attention_mode = "fallback_cls_patch_cosine_similarity"
        if cls_token is None:
            features = model.forward_features(tensor)
            cls_token = features["x_norm_clstoken"]
        cls = F.normalize(cls_token.float(), dim=-1)
        attention = torch.einsum("bc,bchw->bhw", cls, last)[0]
    representations = {
        "last": last[0].cpu().numpy(),
        "last4avg": last4avg[0].cpu().numpy(),
    }
    return representations, attention.cpu().numpy(), attention_mode


def feature_local_contrast(feature_map: np.ndarray) -> np.ndarray:
    channels, height, width = feature_map.shape
    flattened = feature_map.reshape(channels, -1)
    norm = np.linalg.norm(flattened, axis=0, keepdims=True) + 1e-9
    normalized = (flattened / norm).reshape(channels, height, width)
    local_mean = np.stack([cv2.blur(channel, (3, 3)) for channel in normalized], axis=0)
    local_mean /= np.linalg.norm(local_mean, axis=0, keepdims=True) + 1e-9
    return 1.0 - np.sum(normalized * local_mean, axis=0)


def robust_peaks(score_map: np.ndarray, tau: float, max_points: int = 30) -> list[tuple[float, float, float]]:
    score = np.asarray(score_map, dtype=np.float64)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    sigma = max(1.4826 * mad, 1e-8)
    threshold = median + tau * sigma
    local_max = maximum_filter(score, size=3, mode="nearest")
    valid = (score >= threshold) & (score >= local_max - 1e-12)
    if min(score.shape) >= 5:
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
    ys, xs = np.where(valid)
    ranked = sorted(((float(score[y, x]), int(x), int(y)) for x, y in zip(xs, ys)), reverse=True)
    selected: list[tuple[float, float, float]] = []
    for value, x, y in ranked:
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= 4.0 for old_x, old_y, _ in selected):
            selected.append((float(x), float(y), value))
        if len(selected) >= max_points:
            break
    return selected


def quantile_peaks(score_map: np.ndarray, quantile: float, max_points: int = 30) -> list[tuple[float, float, float]]:
    score = np.asarray(score_map, dtype=np.float64)
    threshold = float(np.quantile(score, quantile))
    local_max = maximum_filter(score, size=3, mode="nearest")
    valid = (score >= threshold) & (score >= local_max - 1e-12)
    if min(score.shape) >= 5:
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
    ys, xs = np.where(valid)
    ranked = sorted(((float(score[y, x]), int(x), int(y)) for x, y in zip(xs, ys)), reverse=True)
    selected: list[tuple[float, float, float]] = []
    for value, x, y in ranked:
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= 4.0 for old_x, old_y, _ in selected):
            selected.append((float(x), float(y), value))
        if len(selected) >= max_points:
            break
    return selected


def grid_to_pixels(points: Sequence[tuple[float, float, float]], size: int, grid_shape: tuple[int, int]) -> np.ndarray:
    height, width = grid_shape
    scale_x, scale_y = size / width, size / height
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([((x + 0.5) * scale_x, (y + 0.5) * scale_y) for x, y, _ in points], dtype=np.float64)


def hdbscan_medoids(feature_map: np.ndarray, size: int, seed: int) -> np.ndarray:
    channels, height, width = feature_map.shape
    features = feature_map.reshape(channels, -1).T.astype(np.float64)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    components = min(16, features.shape[0] - 1, features.shape[1])
    if components >= 2:
        features = PCA(n_components=components, random_state=seed).fit_transform(features)
        features /= np.std(features, axis=0, keepdims=True) + 1e-6
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    spatial = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1) * 0.30
    data = np.concatenate([features, spatial], axis=1)
    min_cluster = max(8, round(0.012 * len(data)))
    labels = SklearnHDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=max(3, min_cluster // 3),
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        copy=True,
    ).fit_predict(data)
    points = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        indices = np.where(labels == label)[0]
        cluster = data[indices]
        centroid = cluster.mean(axis=0, keepdims=True)
        medoid_index = indices[int(np.argmin(np.square(cluster - centroid).sum(axis=1)))]
        y, x = divmod(int(medoid_index), width)
        points.append(((x + 0.5) * size / width, (y + 0.5) * size / height))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def apply_inverse(points: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(-1, 2)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    mapped = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :2]
    valid = (mapped[:, 0] >= 0) & (mapped[:, 0] < size) & (mapped[:, 1] >= 0) & (mapped[:, 1] < size)
    return mapped[valid]


def match_points(reference: np.ndarray, predicted: np.ndarray, tolerance: float) -> dict[str, float]:
    if len(reference) == 0 and len(predicted) == 0:
        return {"matches": 0.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "median_error": 0.0}
    if len(reference) == 0 or len(predicted) == 0:
        return {"matches": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "median_error": float("nan")}
    distances = np.linalg.norm(reference[:, None, :] - predicted[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distances)
    accepted = [float(distances[row, col]) for row, col in zip(rows, cols) if distances[row, col] <= tolerance]
    matches = len(accepted)
    precision = matches / len(predicted)
    recall = matches / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": float(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_error": float(np.median(accepted)) if accepted else float("nan"),
    }


def plant_hit_ratio(points: np.ndarray, mask: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    height, width = mask.shape
    hits = 0
    radius = max(2, round(min(height, width) * 0.01))
    dilated = cv2.dilate((mask > 0).astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    for x, y in points:
        xi, yi = int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))
        hits += int(dilated[yi, xi] > 0)
    return hits / len(points)


def finite_median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else float("nan")


def candidates_for(
    method: str,
    feature_map: np.ndarray,
    attention_map: np.ndarray,
    tau: float,
    size: int,
    seed: int,
) -> np.ndarray:
    if method == "feature_local_contrast":
        score = feature_local_contrast(feature_map)
        return grid_to_pixels(robust_peaks(score, tau), size, score.shape)
    if method == "cls_to_patch_attention":
        return grid_to_pixels(quantile_peaks(attention_map, tau), size, attention_map.shape)
    if method == "feature_hdbscan_medoid":
        return hdbscan_medoids(feature_map, size, seed)
    raise KeyError(method)


def evaluate_image_config(
    row: dict[str, str],
    size: int,
    layer_mode: str,
    transformed: list[dict[str, Any]],
    representations: list[dict[str, np.ndarray]],
    attention_maps: list[np.ndarray],
    plant_mask: np.ndarray,
    plant_bbox: tuple[float, float, float, float],
    method_taus: dict[str, Sequence[float]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, float], np.ndarray]]:
    methods = list(PEAK_METHODS)
    methods.append("feature_hdbscan_medoid")
    candidate_cache: dict[tuple[int, str, float], np.ndarray] = {}
    base_points: dict[tuple[str, float], np.ndarray] = {}
    bbox_diag = max(1.0, math.hypot(plant_bbox[2] - plant_bbox[0], plant_bbox[3] - plant_bbox[1]))
    tolerance = 0.05 * bbox_diag
    detail_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for method in methods:
        taus = method_taus.get(method, [0.0] if method == "feature_hdbscan_medoid" else [])
        for tau in taus:
            for index in range(len(transformed)):
                candidate_cache[(index, method, tau)] = candidates_for(
                    method,
                    representations[index][layer_mode],
                    attention_maps[index],
                    tau,
                    size,
                    seed,
                )
            reference = candidate_cache[(0, method, tau)]
            base_points[(method, tau)] = reference
            hit_ratio = plant_hit_ratio(reference, plant_mask)
            f1_values, error_values, photo_count_diff = [], [], []
            for index, transform in enumerate(transformed[1:], start=1):
                mapped = apply_inverse(candidate_cache[(index, method, tau)], transform["matrix"], size)
                matched = match_points(reference, mapped, tolerance)
                error_norm = matched["median_error"] / bbox_diag if np.isfinite(matched["median_error"]) else float("nan")
                f1_values.append(matched["f1"])
                error_values.append(error_norm)
                if transform["family"] == "photometric":
                    photo_count_diff.append(abs(len(mapped) - len(reference)))
                detail_rows.append(
                    {
                        "image_id": row["image_id"],
                        "size": size,
                        "layer_mode": layer_mode,
                        "method": method,
                        "tau": tau,
                        "transform": transform["name"],
                        "transform_family": transform["family"],
                        "base_count": len(reference),
                        "transformed_count_after_inverse": len(mapped),
                        "matches": int(matched["matches"]),
                        "precision": matched["precision"],
                        "recall": matched["recall"],
                        "f1": matched["f1"],
                        "localization_error_bbox_diag": error_norm,
                    }
                )
            aggregate_rows.append(
                {
                    "image_id": row["image_id"],
                    "source_name": row["source_name"],
                    "structural_complexity_proxy": row["structural_complexity_proxy"],
                    "size": size,
                    "layer_mode": layer_mode,
                    "method": method,
                    "tau": tau,
                    "candidate_count": len(reference),
                    "safety_cap_hit": int(len(reference) >= 30),
                    "plant_hit_ratio": hit_ratio,
                    "collapse_background_or_single": int(len(reference) <= 1 or hit_ratio == 0.0),
                    "mean_repeatability_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
                    "median_localization_error_bbox_diag": finite_median(error_values),
                    "median_photometric_count_difference": finite_median(photo_count_diff),
                }
            )
    return aggregate_rows, detail_rows, base_points


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["size"], row["layer_mode"], row["method"], float(row["tau"]))
        grouped[key].append(row)
    summaries = []
    for (size, layer_mode, method, tau), group in sorted(grouped.items()):
        summaries.append(
            {
                "size": size,
                "layer_mode": layer_mode,
                "method": method,
                "tau": tau,
                "image_count": len(group),
                "noncollapse_rate": 1.0 - float(np.mean([row["collapse_background_or_single"] for row in group])),
                "median_candidate_count": finite_median(row["candidate_count"] for row in group),
                "candidate_count_iqr": float(
                    np.quantile([row["candidate_count"] for row in group], 0.75)
                    - np.quantile([row["candidate_count"] for row in group], 0.25)
                ),
                "safety_cap_hit_rate": float(np.mean([row["safety_cap_hit"] for row in group])),
                "median_plant_hit_ratio": finite_median(row["plant_hit_ratio"] for row in group),
                "median_repeatability_f1": finite_median(row["mean_repeatability_f1"] for row in group),
                "median_localization_error_bbox_diag": finite_median(
                    row["median_localization_error_bbox_diag"] for row in group
                ),
                "median_photometric_count_difference": finite_median(
                    row["median_photometric_count_difference"] for row in group
                ),
            }
        )
    for row in summaries:
        localization = row["median_localization_error_bbox_diag"]
        row["calibration_utility"] = (
            row["median_repeatability_f1"]
            + 0.25 * row["median_plant_hit_ratio"]
            + 0.15 * row["noncollapse_rate"]
            - 0.75 * (localization if np.isfinite(localization) else 1.0)
            - 0.25 * row["safety_cap_hit_rate"]
        )
    return summaries


def choose_thresholds(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["size"], row["layer_mode"], row["method"])].append(row)
    frozen = {}
    for key, rows in grouped.items():
        best = max(rows, key=lambda row: (row["calibration_utility"], -float(row["tau"])))
        frozen["|".join(map(str, key))] = {
            "tau": float(best["tau"]),
            "calibration_metrics": best,
        }
    return frozen


def selected_taus_for_phase(
    phase: str, output: Path, sizes: Sequence[int], layers: Sequence[str]
) -> dict[tuple[int, str, str], list[float]]:
    selected: dict[tuple[int, str, str], list[float]] = {}
    if phase in ("smoke", "calibrate"):
        for size in sizes:
            for layer in layers:
                for method, grid in METHOD_TAU_GRID.items():
                    selected[(size, layer, method)] = list(grid)
                selected[(size, layer, "feature_hdbscan_medoid")] = [0.0]
        return selected
    frozen_path = output / "calibration" / "frozen_thresholds.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"Pilot phase requires {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    candidates: dict[str, list[tuple[tuple[int, str, str], dict[str, Any]]]] = defaultdict(list)
    for key, payload in frozen.items():
        size, layer, method = key.split("|")
        if int(size) in sizes and layer in layers:
            candidates[method].append(((int(size), layer, method), payload))
    # The pilot evaluates one frozen winner per method. This prevents the held
    # out 60 images from becoming another model-selection set and avoids
    # redundant CPU/GPU computation for configurations already rejected on 20.
    for method, options in candidates.items():
        config, payload = max(
            options,
            key=lambda item: (
                float(item[1]["calibration_metrics"]["calibration_utility"]),
                int(item[0][1] == "last4avg"),
                -item[0][0],
            ),
        )
        selected[config] = [float(payload["tau"])]
    return selected


def overlay_points(image_rgb: np.ndarray, panels: list[tuple[str, np.ndarray]], output: Path) -> None:
    size = image_rgb.shape[0]
    panel_images = []
    colors = [(0, 190, 255), (70, 220, 80), (255, 80, 75)]
    for panel_index, (title, points) in enumerate(panels):
        pil = Image.fromarray(image_rgb.copy())
        draw = ImageDraw.Draw(pil)
        radius = max(4, size // 90)
        for point_index, (x, y) in enumerate(points, start=1):
            color = colors[panel_index % len(colors)]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
            draw.text((x + radius + 1, y - radius), str(point_index), fill="white", stroke_width=2, stroke_fill="black")
        draw.rectangle((0, 0, size, 28), fill=(0, 0, 0, 180))
        draw.text((8, 7), f"{title} | n={len(points)}", fill="white")
        panel_images.append(pil)
    canvas = Image.new("RGB", (size * len(panel_images), size), "white")
    for index, panel in enumerate(panel_images):
        canvas.paste(panel, (index * size, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def gate_decision(summary_rows: list[dict[str, Any]], hdbscan_available: bool) -> dict[str, Any]:
    feature_rows = [row for row in summary_rows if row["method"] in FEATURE_METHODS]
    baseline_rows = [row for row in summary_rows if row["method"] == "cls_to_patch_attention"]
    if not feature_rows or not baseline_rows:
        return {"status": "not_evaluable", "reason": "Feature or attention baseline rows are missing."}
    best_feature = max(feature_rows, key=lambda row: row["median_repeatability_f1"])
    best_baseline = max(baseline_rows, key=lambda row: row["median_repeatability_f1"])
    checks = {
        "noncollapse_rate_at_least_0.80": best_feature["noncollapse_rate"] >= 0.80,
        "median_plant_hit_ratio_at_least_0.50": best_feature["median_plant_hit_ratio"] >= 0.50,
        "safety_cap_hit_rate_at_most_0.10": best_feature["safety_cap_hit_rate"] <= 0.10,
        "candidate_count_iqr_at_least_1": best_feature["candidate_count_iqr"] >= 1.0,
        "feature_f1_advantage_at_least_0.10": (
            best_feature["median_repeatability_f1"] - best_baseline["median_repeatability_f1"] >= 0.10
        ),
        "localization_error_at_most_0.05_bbox_diag": (
            np.isfinite(best_feature["median_localization_error_bbox_diag"])
            and best_feature["median_localization_error_bbox_diag"] <= 0.05
        ),
        "photometric_count_difference_at_most_1": (
            np.isfinite(best_feature["median_photometric_count_difference"])
            and best_feature["median_photometric_count_difference"] <= 1.0
        ),
        "two_reviewer_structure_check": None,
    }
    automatic = all(value is True for key, value in checks.items() if key != "two_reviewer_structure_check")
    return {
        "status": "automatic_checks_pass_human_review_pending" if automatic else "stop_or_revise_g1",
        "best_feature_configuration": best_feature,
        "best_attention_baseline": best_baseline,
        "automatic_checks": checks,
        "hdbscan_available": hdbscan_available,
        "interpretation_boundary": "Passing automatic checks would show stable structure-related candidates, not biological landmark validity.",
    }


def verify_environment(args: argparse.Namespace, calibration: list[dict[str, str]], pilot: list[dict[str, str]]) -> dict[str, Any]:
    missing_crops = [row["image_id"] for row in calibration + pilot if not Path(row["crop_path"]).exists()]
    sizes_valid = all(size > 0 and size % PATCH_SIZE == 0 for size in args.sizes)
    local_repo_valid = args.local_repo is None or (args.local_repo / "hubconf.py").exists()
    weights_valid = args.weights is None or args.weights.exists()
    integrity: dict[str, Any] | None = None
    integrity_error: str | None = None
    if weights_valid and args.weights is not None:
        try:
            integrity = checkpoint_integrity(args.weights)
        except Exception as exc:
            weights_valid = False
            integrity_error = str(exc)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "requested_sizes": args.sizes,
        "sizes_divisible_by_patch14": sizes_valid,
        "calibration_images": len(calibration),
        "pilot_images": len(pilot),
        "missing_crops": missing_crops,
        "hdbscan_available": True,
        "local_repo_valid": local_repo_valid,
        "weights_valid": weights_valid,
        "weights_integrity": integrity,
        "weights_integrity_error": integrity_error,
        "ready_for_model_run": sizes_valid and not missing_crops and local_repo_valid and weights_valid,
        "note": "CPU can verify logic; the full 80-image multi-transform run is intended for CUDA.",
    }


def run(args: argparse.Namespace) -> None:
    set_deterministic(args.seed)
    g0_root, output = args.g0_root.resolve(), args.output.resolve()
    pilot_rows = read_csv(g0_root / "pilot80_manifest.csv")
    calibration_rows, held_rows = partition_pilot(pilot_rows, args.seed)
    role_rows = [dict(row, g1_role="calibration20") for row in calibration_rows] + [
        dict(row, g1_role="pilot60") for row in held_rows
    ]
    write_csv(output / "g1_partition_manifest.csv", role_rows)
    environment = verify_environment(args, calibration_rows, held_rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment_check.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    if args.phase == "verify":
        return
    if not environment["ready_for_model_run"]:
        raise RuntimeError("Environment verification failed; see environment_check.json")

    rows = calibration_rows if args.phase in ("smoke", "calibrate") else held_rows
    if args.limit:
        rows = rows[: args.limit]
    phase_dir = output / ("calibration" if args.phase == "calibrate" else args.phase)
    phase_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"LOADING_MODEL model={args.model} device={device}", flush=True)
    model = load_official_model(args, device)
    selected_taus = selected_taus_for_phase(args.phase, output, args.sizes, args.layers)
    if args.phase == "pilot":
        chosen = {
            "|".join(map(str, key)): values[0]
            for key, values in sorted(selected_taus.items())
        }
        (phase_dir / "selected_configurations.json").write_text(
            json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    full_transforms = args.full_transforms or args.phase in ("calibrate", "pilot")
    all_aggregate: list[dict[str, Any]] = []
    all_detail: list[dict[str, Any]] = []
    attention_modes = Counter()
    visualization_cache: dict[tuple[int, str], list[tuple[dict[str, str], np.ndarray, dict[tuple[str, float], np.ndarray]]]] = defaultdict(list)
    start = time.time()
    for size in args.sizes:
        if size % PATCH_SIZE:
            raise ValueError(f"Input size {size} is not divisible by patch size {PATCH_SIZE}.")
        for row_index, row in enumerate(rows, start=1):
            base, mapping = letterbox_rgb(Path(row["crop_path"]), size)
            mask, bbox = square_plant_mask(row, size, mapping)
            transforms = make_transforms(base, full_transforms)
            reps, attn = [], []
            for transform in transforms:
                layer_reps, attention_map, attention_mode = extract_representations(model, transform["image"], device)
                reps.append(layer_reps)
                attn.append(attention_map)
                attention_modes[attention_mode] += 1
            for layer in args.layers:
                if args.phase == "pilot" and not any(
                    key[0] == size and key[1] == layer for key in selected_taus
                ):
                    continue
                if args.phase == "pilot":
                    method_taus = {
                        method: selected_taus[(size, layer, method)]
                        for method in (*PEAK_METHODS, "feature_hdbscan_medoid")
                        if (size, layer, method) in selected_taus
                    }
                else:
                    method_taus = {
                        method: selected_taus.get((size, layer, method), grid)
                        for method, grid in METHOD_TAU_GRID.items()
                    }
                    method_taus["feature_hdbscan_medoid"] = selected_taus.get(
                        (size, layer, "feature_hdbscan_medoid"), [0.0]
                    )
                aggregate, details, base_points = evaluate_image_config(
                    row, size, layer, transforms, reps, attn, mask, bbox, method_taus, args.seed
                )
                # Remove threshold variants that were not frozen for this method in pilot phase.
                if args.phase == "pilot":
                    aggregate = [
                        item
                        for item in aggregate
                        if float(item["tau"]) in selected_taus.get((size, layer, item["method"]), [])
                    ]
                    details = [
                        item
                        for item in details
                        if float(item["tau"]) in selected_taus.get((size, layer, item["method"]), [])
                    ]
                all_aggregate.extend(aggregate)
                all_detail.extend(details)
                if len(visualization_cache[(size, layer)]) < args.visual_limit:
                    visualization_cache[(size, layer)].append((row, base, base_points))
            elapsed = time.time() - start
            print(f"G1 size={size} image={row_index}/{len(rows)} elapsed={elapsed:.1f}s", flush=True)

    summary_rows = summarize(all_aggregate)
    write_csv(phase_dir / "per_image_metrics.csv", all_aggregate)
    write_csv(phase_dir / "per_transform_metrics.csv", all_detail)
    write_csv(phase_dir / "summary_metrics.csv", summary_rows)
    metadata = {
        "phase": args.phase,
        "model": args.model,
        "device": str(device),
        "sizes": args.sizes,
        "layers": args.layers,
        "transforms": "full" if full_transforms else "smoke_core",
        "attention_extraction_modes": dict(attention_modes),
        "hdbscan_available": True,
        "weights_integrity": checkpoint_integrity(args.weights) if args.weights else None,
        "elapsed_seconds": time.time() - start,
    }
    if args.phase in ("smoke", "calibrate"):
        frozen = choose_thresholds(summary_rows)
        (phase_dir / "frozen_thresholds.json").write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        decision = gate_decision(summary_rows, True)
        (phase_dir / "g1_gate_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata["gate_status"] = decision["status"]
    (phase_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen_file = phase_dir / "frozen_thresholds.json" if args.phase != "pilot" else output / "calibration" / "frozen_thresholds.json"
    frozen = json.loads(frozen_file.read_text(encoding="utf-8")) if frozen_file.exists() else {}
    for (size, layer), cached in visualization_cache.items():
        panels_config = []
        for method in ("feature_local_contrast", "feature_hdbscan_medoid", "cls_to_patch_attention"):
            key = f"{size}|{layer}|{method}"
            if key in frozen and (
                args.phase != "pilot" or (size, layer, method) in selected_taus
            ):
                panels_config.append((method, float(frozen[key]["tau"])))
        for row, base, base_points in cached:
            panels = []
            for method, tau in panels_config:
                points = base_points.get((method, tau), np.empty((0, 2)))
                panels.append((method, points))
            if panels:
                overlay_points(base, panels, phase_dir / "visualizations" / f"{row['image_id']}_{size}_{layer}.jpg")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
