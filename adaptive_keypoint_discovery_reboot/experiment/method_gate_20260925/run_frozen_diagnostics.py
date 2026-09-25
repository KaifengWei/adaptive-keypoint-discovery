"""Val-only, read-only runtime and perturbation audit of the frozen methods.

Run each method in its own process. This script never reads GT or the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "phenotype_pilot_protocol"))

import evaluate_adaptive_point_detector as student_eval  # noqa: E402
import evaluate_point_conditioned_graph_v1 as graph_eval  # noqa: E402
import g1_dinov2_feasibility as g1  # noqa: E402
import g1_prime_phenotype_bridge as bridge  # noqa: E402
import g1_prime_structural_support as gp  # noqa: E402
import generate_g1prime_pseudolabels as teacher_eval  # noqa: E402
from adaptive_point_model import AdaptivePointDetector  # noqa: E402
from phenotype_gt_geometry import (  # noqa: E402
    divergence_angle, match_cross_session, select_reference_main, trace_metrics,
)
from phenotype_roi_basal_anchor import load_phenotype_input  # noqa: E402
from point_conditioned_graph import build_point_conditioned_graph  # noqa: E402
from point_conditioned_organ_paths import decode_candidate_organ_paths  # noqa: E402

CHECKPOINT_SHA = "bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2"
SEED = 20260925
WARMUP_PASSES = 2
MEASURED_PASSES = 10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_bytes() -> int:
    with Path("/proc/self/status").open() as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Linux /proc/self/status has no VmRSS")


def total_ram_bytes() -> int:
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Linux /proc/meminfo has no MemTotal")


def device_info() -> dict:
    import subprocess
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    return {
        "host": platform.node(), "gpu": gpu, "cuda": torch.version.cuda,
        "torch": torch.__version__, "cpu": platform.processor(),
        "system_ram_bytes": total_ram_bytes(),
        "python": sys.version.split()[0],
    }


def load_inputs() -> list[dict]:
    dataset = HERE / "data_stage_clean_v4_fullplant_candidate"
    selection = pd.read_csv(HERE / "phenotype_pilot_protocol/phenotype_pilot_selection.csv")
    if len(selection) != 16 or selection.pilot_group.value_counts().to_dict() != {"core": 12, "diagnostic": 4}:
        raise RuntimeError("The frozen pilot set changed")
    manifest = pd.read_csv(dataset / "manifests/val.csv").set_index("dataset_id")
    if not set(selection.dataset_id) <= set(manifest.index):
        raise RuntimeError("Pilot IDs are not all in the locked V4 val manifest")
    inputs = []
    for entry in selection.to_dict("records"):
        ident = str(entry["dataset_id"])
        record = manifest.loc[ident].to_dict()
        record["dataset_id"] = ident
        image, mapping, _, roi = load_phenotype_input(dataset, record, 518)
        masks = {
            name: graph_eval.load_mask_canvas(
                dataset / Path(str(record[key]).replace("\\", "/")), mapping, 518
            )
            for name, key in (
                ("shoot", "shoot_mask_relative_path"),
                ("seed_base_root", "seed_base_root_mask_relative_path"),
                ("full_plant", "full_plant_mask_relative_path"),
            )
        }
        masks["phenotype_roi"] = roi["phenotype_roi_model"]
        masks["basal_transition"] = roi["basal_transition_model"]
        inputs.append({"dataset_id": ident, "group": entry["pilot_group"], "image": image, "mapping": mapping, "masks": masks})
    return inputs


class Method:
    def __init__(self, name: str):
        self.name = name
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("This protocol requires the same CUDA RTX 3090")
        config = json.loads((HERE / "training_outputs/core_dinov2_v4_phenotype_roi/resolved_config.json").read_text())
        model_args = argparse.Namespace(
            local_repo=Path(config["dinov2_local_repo"]),
            weights=Path(config["dinov2_weights"]), model=config["dinov2_model"],
        )
        g1.set_deterministic(SEED)
        backbone = g1.load_official_model(model_args, self.device)
        if name == "Teacher-direct":
            self.model = backbone.eval()
            self.consensus_args = argparse.Namespace(size=518, no_consistency_filter=False, min_presence=0.75, max_localization_error=0.025)
        elif name == "Student-B":
            checkpoint_file = HERE / "training_outputs/core_dinov2_v4_phenotype_roi/best.pt"
            if sha(checkpoint_file) != CHECKPOINT_SHA:
                raise RuntimeError("Student checkpoint hash changed")
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.model = AdaptivePointDetector(
                backbone, patch_size=14, decoder_dim=int(config["decoder_dim"]),
                output_stride=int(config["output_stride"]), freeze_backbone=True,
                unfreeze_last_blocks=0,
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            self.threshold = float(config["inference_threshold"])
            self.safety_cap = int(config["inference_safety_cap"])
            self.fixed_k = int(config["fixed_k_eval"])
        else:
            raise ValueError(name)

    @torch.inference_mode()
    def points(self, image: np.ndarray) -> list[dict]:
        if self.name == "Student-B":
            _, records = student_eval.predict(self.model, image, self.device, self.threshold, self.safety_cap, self.fixed_k)
            return [
                {"point_id": f"p{i:02d}", "x": float(row["x"]), "y": float(row["y"]), "score": float(row["score"]), "kind": row["kind"]}
                for i, row in enumerate(records, 1)
            ]
        transforms = g1.make_transforms(image, True)
        outputs = []
        for transform in transforms:
            representations, attention, _ = g1.extract_representations(self.model, transform["image"], self.device)
            points, records, support, skeleton, diagnostics = gp.structural_candidates(
                transform["image"], representations["last4avg"], attention, 30,
                evidence_mode="full", structure_coverage=False, basal_transition_mask=None,
            )
            outputs.append({"points": points, "records": records, "support": support, "skeleton": skeleton, "diagnostics": diagnostics})
        bbox = gp.bbox_from_mask(outputs[0]["support"])
        diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        accepted, _ = teacher_eval.consensus(outputs, transforms, diag, self.consensus_args)
        return [
            {"point_id": f"p{i:02d}", "x": float(row["x_model"]), "y": float(row["y_model"]), "score": float(row["consensus_confidence"]), "kind": row["kind"]}
            for i, row in enumerate(accepted, 1)
        ]


def graph_decoder(image: np.ndarray, masks: dict[str, np.ndarray], points: list[dict]) -> tuple[list[dict], dict, float]:
    support, raw_skeleton, _ = gp.automatic_structural_support(image)
    bbox = gp.bbox_from_mask(support)
    diag = max(1.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    skeleton, _ = bridge.prune_short_terminal_spurs(raw_skeleton, max(4.0, 0.012 * diag))
    graph = build_point_conditioned_graph(skeleton, points, diag, 0.025)
    radius = max(2, round(image.shape[0] * 0.01))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    tolerant = {name: cv2.dilate(mask.astype(np.uint8), kernel) > 0 for name, mask in masks.items()}
    graph_eval.annotate_regions(graph, masks, tolerant)
    paths, decision = decode_candidate_organ_paths(
        graph, masks["shoot"], masks["seed_base_root"], diag,
        phenotype_roi_mask=masks["phenotype_roi"], basal_transition_mask=masks["basal_transition"],
        branch_pruning_mode="local_learned_support",
    )
    return paths, decision, diag


def phenotype(paths: list[dict], model_diag: float, mapping: dict, matrix: np.ndarray | None = None) -> list[dict]:
    source_diag = model_diag / float(mapping["scale"])
    traces = []
    for path in paths:
        curve = np.asarray(path["full_base_to_tip_path"], dtype=np.float64)
        if matrix is not None:
            curve = g1.apply_inverse(curve, matrix, 518)
        curve = (curve - np.asarray([mapping["pad_x"], mapping["pad_y"]])) / float(mapping["scale"])
        curve[:, 0] = np.clip(curve[:, 0], 0.0, float(mapping["source_width"]) - 1.0)
        curve[:, 1] = np.clip(curve[:, 1], 0.0, float(mapping["source_height"]) - 1.0)
        metrics = trace_metrics(curve, source_diag)
        resampled = metrics["resampled_curve"]
        traces.append({
            "trace_uuid": str(path["path_id"]), "visibility_status": "measurable",
            "tip_xy": resampled[-1].tolist(), "tip_clockwise_angle_deg": metrics["tip_clockwise_angle_deg"],
            "resampled_curve": resampled, "structural_path_length_px": metrics["structural_path_length_px"],
            "chord_length_px": metrics["chord_length_px"], "divergence_angle_deg": None,
        })
    if traces:
        main = select_reference_main(traces, source_diag)
        main_curve = next(t["resampled_curve"] for t in traces if t["trace_uuid"] == main)
        for trace in traces:
            if trace["trace_uuid"] != main:
                trace["divergence_angle_deg"] = divergence_angle(main_curve, trace["resampled_curve"], source_diag).get("divergence_angle_deg")
    return traces


def saved_points(method: str, inputs: list[dict], teacher_path: Path) -> dict[str, list[dict]]:
    path = teacher_path if method == "Teacher-direct" else HERE / "evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv"
    if method == "Teacher-direct" and sha(path) != "1ed017a6d6ba6f360bf216c44b01ffedaef9c2a14f18d46d75b1f640763b1814":
        raise RuntimeError("Frozen Teacher points changed")
    if method == "Student-B" and sha(path) != "9a11bbcc7283333fca34c0e82a3931edbfa96cef8347ec952f450daecb1aa756":
        raise RuntimeError("Frozen Student points changed")
    frame = pd.read_csv(path)
    result = {}
    for sample in inputs:
        mapping = sample["mapping"]
        rows = frame[frame.dataset_id == sample["dataset_id"]].sort_values("point_id")
        result[sample["dataset_id"]] = [
            {"point_id": str(row.point_id), "x": float(row.x_source) * mapping["scale"] + mapping["pad_x"],
             "y": float(row.y_source) * mapping["scale"] + mapping["pad_y"],
             "score": float(row.confidence), "kind": "learned_heatmap"}
            for row in rows.itertuples()
        ]
    return result


class RssSampler:
    def __init__(self):
        self.peak = rss_bytes()
        self.active = False

    def start(self):
        self.peak = rss_bytes()
        self.active = True
        def poll():
            while self.active:
                self.peak = max(self.peak, rss_bytes())
                time.sleep(0.005)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        self.thread.join()
        self.peak = max(self.peak, rss_bytes())
        return self.peak


def timed(fn) -> tuple[float, object]:
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    value = fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / 1e6, value


def benchmark(method: Method, inputs: list[dict], points_by_id: dict, output: Path) -> None:
    rows = []
    stage_peaks = {}
    for stage in ("point_generation_only", "shared_graph_decoder", "end_to_end_phenotype"):
        def run_one(sample):
            image, masks, ident = sample["image"], sample["masks"], sample["dataset_id"]
            if stage == "point_generation_only":
                return method.points(image)
            if stage == "shared_graph_decoder":
                return graph_decoder(image, masks, points_by_id[ident])
            pts = method.points(image)
            paths, _, diag = graph_decoder(image, masks, pts)
            return phenotype(paths, diag, sample["mapping"])
        for _ in range(WARMUP_PASSES):
            for sample in inputs:
                run_one(sample)
        torch.cuda.synchronize()
        rss = RssSampler()
        baseline_ram = rss_bytes()
        baseline_gpu = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rss.start()
        for repeat in range(MEASURED_PASSES):
            for sample in inputs:
                milliseconds, _ = timed(lambda sample=sample: run_one(sample))
                rows.append({"stage": stage, "repeat": repeat, "dataset_id": sample["dataset_id"], "group": sample["group"], "milliseconds": milliseconds})
            print(f"{method.name} {stage} pass {repeat + 1}/{MEASURED_PASSES}", flush=True)
        peak_ram = rss.stop()
        peak_gpu = torch.cuda.max_memory_allocated()
        stage_peaks[stage] = {"baseline_ram_bytes": baseline_ram, "peak_ram_bytes": peak_ram, "baseline_gpu_bytes": baseline_gpu, "peak_gpu_bytes": peak_gpu}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"benchmark_{method.name}.csv", index=False)
    metadata = {"method": method.name, "hardware": device_info(), "seed": SEED, "warmup_passes": WARMUP_PASSES, "measured_passes": MEASURED_PASSES,
                "N_per_stage": len(inputs) * MEASURED_PASSES, "stages": stage_peaks, "preprocessing_and_file_IO_excluded": True,
                "no_GT_or_test_read": True, "model_training": False}
    (output / f"benchmark_{method.name}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def transform_sample(sample: dict, name: str) -> tuple[np.ndarray, dict, np.ndarray]:
    image = sample["image"]
    if name == "brightness_085":
        return np.clip(image.astype(np.float32) * 0.85, 0, 255).astype(np.uint8), sample["masks"], np.eye(3)
    view = next(t for t in g1.make_transforms(image, True) if t["name"] == name)
    masks = {key: teacher_eval.transformed_mask(mask, view["matrix"], 518) for key, mask in sample["masks"].items()}
    return view["image"], masks, view["matrix"]


def robustness(method: Method, inputs: list[dict], output: Path) -> None:
    rows = []
    for sample in inputs:
        identity_points = method.points(sample["image"])
        identity_paths, _, diag = graph_decoder(sample["image"], sample["masks"], identity_points)
        identity_traces = phenotype(identity_paths, diag, sample["mapping"])
        point_xy = np.asarray([[p["x"], p["y"]] for p in identity_points], dtype=float).reshape(-1, 2)
        for name in ("flip_horizontal", "rotate_10", "brightness_085"):
            image, masks, matrix = transform_sample(sample, name)
            perturbed_points = method.points(image)
            paths, _, _ = graph_decoder(image, masks, perturbed_points)
            mapped_points = g1.apply_inverse(np.asarray([[p["x"], p["y"]] for p in perturbed_points], dtype=float).reshape(-1, 2), matrix, 518)
            point_match = g1.match_points(point_xy, mapped_points, 0.05 * diag)
            perturbed_traces = phenotype(paths, diag, sample["mapping"], matrix)
            matches = match_cross_session(identity_traces, perturbed_traces, diag / float(sample["mapping"]["scale"]))
            by_a = {t["trace_uuid"]: t for t in identity_traces}
            by_b = {t["trace_uuid"]: t for t in perturbed_traces}
            length_diff = []
            angle_diff = []
            matched = 0
            for pair in matches:
                if pair.status != "matched":
                    continue
                matched += 1
                a, b = by_a[pair.trace_uuid_session_1], by_b[pair.trace_uuid_session_2]
                la, lb = a["structural_path_length_px"], b["structural_path_length_px"]
                length_diff.append(100.0 * abs(la - lb) / max((la + lb) / 2.0, 1e-12))
                if a["divergence_angle_deg"] is not None and b["divergence_angle_deg"] is not None:
                    angle_diff.append(abs(a["divergence_angle_deg"] - b["divergence_angle_deg"]))
            rows.append({"dataset_id": sample["dataset_id"], "group": sample["group"], "transform": name,
                         "identity_points": len(identity_points), "perturbed_points": len(perturbed_points), "point_count_abs_change": abs(len(identity_points) - len(perturbed_points)),
                         "point_f1": float(point_match["f1"]), "identity_paths": len(identity_paths), "perturbed_paths": len(paths),
                         "path_count_exact": int(len(identity_paths) == len(paths)), "matched_identity_paths": matched,
                         "path_coverage": None if not identity_paths else matched / len(identity_paths), "length_symmetric_diff_pct": length_diff,
                         "branch_angle_abs_diff_deg": angle_diff})
        print(f"{method.name} robustness {len(rows) // 3}/{len(inputs)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"robustness_{method.name}.json").write_text(json.dumps({"method": method.name, "hardware": device_info(), "rows": rows,
        "transform_set": ["flip_horizontal", "rotate_10", "brightness_085"], "no_GT_or_test_read": True, "model_training": False}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "robustness", "preflight"], required=True)
    parser.add_argument("--method", choices=["Teacher-direct", "Student-B"], required=True)
    parser.add_argument("--teacher-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = load_inputs()
    method = Method(args.method)
    if args.mode == "preflight":
        for sample in inputs:
            points = method.points(sample["image"])
            print(sample["dataset_id"], len(points), flush=True)
        return
    if args.mode == "benchmark":
        frozen_points = saved_points(args.method, inputs, args.teacher_points)
        benchmark(method, inputs, frozen_points, args.output)
    else:
        robustness(method, inputs, args.output)


if __name__ == "__main__":
    main()
