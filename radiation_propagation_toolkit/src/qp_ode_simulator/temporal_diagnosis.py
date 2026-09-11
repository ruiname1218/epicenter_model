"""Predeclared temporal ablations and explicitly privileged oracle diagnostics.

REI adaptation follows arXiv:2506.16834v1 Algorithm 1, not its inconsistent
prose thresholds. Input is interior detector events at hosted check sites;
this is NOT a reproduction of the original noise model or online benchmark.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import _hash_file, _new_directory, load_syndromes
from .temporal_localization import TemporalCNN, _prepare, _json_spec, _mean_interval, detector_series, summarize

ARMS = ("original", "original_joint", "multiscale", "multiscale_joint",
        "oracle_onset", "oracle_parameters")
ORACLE_FIELDS = ("event_onset_ms", "axis_ratio", "angle_degrees", "apparent_speed_m_per_s",
                 "diffusion_coefficient_mm2_per_ms", "initial_lambda_mm", "maximum_lambda_mm",
                 "maximum_distance_mm", "front_width_ms", "qp_source_lifetime_ms",
                 "qp_generation_scale_per_us")


class DiagnosisCNN(nn.Module):
    """Matched 129-feature/site encoders; optional auxiliary time or oracle head."""

    def __init__(self, sites, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown arm")
        self.arm = arm
        self.joint = arm.endswith("_joint")
        self.oracle_size = 1 if arm == "oracle_onset" else len(ORACLE_FIELDS) if arm == "oracle_parameters" else 0
        # Construct original first so the original arm has exactly the same
        # initialization and architecture as the preceding experiment.
        original = TemporalCNN(sites)
        self.temporal, self.head = original.temporal, original.head
        if arm.startswith("multiscale"):
            del self.temporal
            self.short = nn.Sequential(nn.Conv1d(1, 8, 9, stride=4, padding=4), nn.ReLU(),
                                       nn.Conv1d(8, 4, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(16))
            self.long = nn.Sequential(nn.AvgPool1d(16, ceil_mode=True),
                                      nn.Conv1d(1, 8, 9, padding=8, dilation=2), nn.ReLU(),
                                      nn.Conv1d(8, 8, 5, padding=8, dilation=4), nn.ReLU(),
                                      nn.AdaptiveAvgPool1d(8))
        if self.oracle_size:
            self.head[0] = nn.Linear(sites * 129 + self.oracle_size, 64)
        # Independent auxiliary head preserves the position head initialization.
        if self.joint:
            self.time_head = nn.Linear(sites * 129, 1)

    def forward(self, values, privileged=None):
        batch, sites, ticks = values.shape
        signal = values.reshape(batch * sites, 1, ticks) * 10
        if self.arm.startswith("multiscale"):
            encoded = torch.cat((self.short(signal).flatten(1), self.long(signal).flatten(1)), 1)
        else:
            encoded = self.temporal(signal).flatten(1)
        encoded = encoded.reshape(batch, sites, 128)
        feature = torch.cat((encoded, values.mean(2, keepdim=True) * 10), 2).flatten(1)
        if self.oracle_size:
            if privileged is None or privileged.shape != (batch, self.oracle_size):
                raise ValueError("diagnostic oracle requires privileged parameters")
            feature = torch.cat((feature, privileged), 1)
        elif privileged is not None:
            raise ValueError("ordinary inference must not receive privileged labels")
        position = self.head(feature)
        return torch.cat((position, self.time_head(feature)), 1) if self.joint else position


def oracle_values(rows, arm):
    fields = ORACLE_FIELDS[:1] if arm == "oracle_onset" else ORACLE_FIELDS if arm == "oracle_parameters" else ()
    values = rows[list(fields)].to_numpy(dtype=np.float32).copy()
    if arm == "oracle_parameters":
        values[:, -1] = np.log10(values[:, -1])
    # NaNs denote an inapplicable ballistic/diffusive coefficient, not a label.
    return np.nan_to_num(values, nan=0.)


def rei_center(values, sites, device, *, correlation_multiplier=2.,
               circuit_repetitions=None, history_length=None):
    """One FIFO endpoint, K=whole observed prefix. NaN means no detection.

    Algorithm 1: alpha=1/((rounds+1)*K), >2 retained sites, min-max
    normalization, nearest-neighbour gate, squared weights. rounds=K+1
    because the initial boundary measurement is excluded in this dataset.
    That legacy correspondence is an unverified adaptation, not a paper fact.
    New comparisons should explicitly set circuit_repetitions independently
    of history_length. The latter retains the most recent observed rows.
    """
    sites, device = np.asarray(sites), np.asarray(device)
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1] != len(sites) or values.shape[-1] == 0:
        raise ValueError("expected nonempty sample/site/time input")
    if history_length is not None:
        if not isinstance(history_length, (int, np.integer)) or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        values = values[..., -history_length:]
    k = values.shape[-1]
    repetitions = k + 1 if circuit_repetitions is None else circuit_repetitions
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("circuit_repetitions must be a positive integer")
    rate = values.mean(-1)
    dist = np.linalg.norm(device[:, None] - device[None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    threshold = correlation_multiplier * dist.min(1).mean()
    result = np.full((len(values), 2), np.nan)
    for i, frequency in enumerate(rate):
        keep = frequency > 1 / ((repetitions + 1) * k)
        if keep.sum() <= 2:
            continue
        xy, weight = sites[keep], frequency[keep].copy()
        distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        if distance.min(1).mean() > threshold:
            continue
        if np.ptp(weight) > 0:
            weight = (weight - weight.min()) / np.ptp(weight)
        weight **= 2
        if weight.sum() > 0:
            result[i] = np.average(xy, axis=0, weights=weight)
    return result


def predict_arrays(model, raw, indices, center, scale, privileged, time_center, time_scale, batch_size=64):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            extra = torch.tensor(privileged[idx]) if model.oracle_size else None
            pred = model(torch.tensor(np.array(raw[idx], dtype=np.float32)), extra).numpy()
            pred[:, :2] = pred[:, :2] * scale + center
            if model.joint:
                pred[:, 2] = pred[:, 2] * time_scale + time_center
            predictions.append(pred)
    return np.concatenate(predictions)


def fit_arm(raw, rows, spec, train, val, arm, seed, output, *, epochs=40, patience=8, alpha=.1):
    output.mkdir()
    torch.manual_seed(seed)
    model = DiagnosisCNN(raw.shape[1], arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    onset = rows.event_onset_ms.to_numpy()
    center = truth[train].mean(0)
    scale = max(float(np.ptp(spec["geometry"]["circuit_physical_coords_mm"], axis=0).max()/2), 1e-3)
    time_center, time_scale = float(onset[train].mean()), max(float(onset[train].std()), 1e-6)
    privileged = oracle_values(rows, arm)
    p_center = privileged[train].mean(0)
    p_scale = np.maximum(privileged[train].std(0), 1e-6)
    privileged = ((privileged - p_center) / p_scale).astype(np.float32)
    rng, history, best, selected, best_error = np.random.default_rng(seed), [], None, 0, np.inf
    begin = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), 64):
            idx = order[start:start+64]
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.tensor(np.array(raw[idx], dtype=np.float32)),
                         torch.tensor(privileged[idx]) if model.oracle_size else None)
            loss = nn.functional.mse_loss(pred[:, :2], torch.tensor((truth[idx]-center)/scale, dtype=torch.float32))
            if model.joint:
                loss = loss + alpha * nn.functional.mse_loss(pred[:, 2], torch.tensor((onset[idx]-time_center)/time_scale, dtype=torch.float32))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        prediction = predict_arrays(model, raw, val, center, scale, privileged, time_center, time_scale)
        error = float(np.linalg.norm(prediction[:, :2]-truth[val], axis=1).mean())
        history.append({"epoch": epoch, "validation_mean_error_mm": error})
        if error < best_error:
            best, selected, best_error = copy.deepcopy(model.state_dict()), epoch, error
        if epoch-selected >= patience:
            break
    if best is None:
        raise ValueError("training has no finite checkpoint")
    model.load_state_dict(best)
    torch.save(best, output / "weights.pt")
    metadata = {"format": "temporal_diagnosis_v1", "arm": arm, "seed": seed,
                "alpha": alpha, "selected_epoch": selected, "history": history,
                "validation_mean_error_mm": best_error, "training_seconds": time.perf_counter()-begin,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "target_center_mm": center.tolist(), "target_scale_mm": scale,
                "time_center_ms": time_center, "time_scale_ms": time_scale,
                "privileged_center": p_center.tolist(), "privileged_scale": p_scale.tolist(),
                "diagnostic_only": bool(model.oracle_size), "specification": _json_spec(spec),
                "weights_sha256": _hash_file(output / "weights.pt")}
    _write_json(output / "model.json", metadata)
    return model, metadata, privileged


def predict_model(directory, input_path):
    """Deployable arms take observations only; diagnostic oracles are rejected."""
    root = Path(directory)
    info = json.loads((root / "model.json").read_text())
    if info.get("format") != "temporal_diagnosis_v1" or info["diagnostic_only"]:
        raise ValueError("unsupported or privileged diagnostic model")
    if _hash_file(root / "weights.pt") != info["weights_sha256"]:
        raise ValueError("weights checksum mismatch")
    raw = detector_series(load_syndromes(input_path), info["specification"])
    model = DiagnosisCNN(raw.shape[1], info["arm"])
    model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
    return predict_arrays(model, raw, np.arange(len(raw)), np.asarray(info["target_center_mm"]),
                          info["target_scale_mm"], None, info["time_center_ms"], info["time_scale_ms"])


def run(dataset, plan_path, output, *, seeds=(41, 42, 43), epochs=40, patience=8, alpha=.1):
    root, destination = Path(dataset), Path(output)
    if epochs < 1 or patience < 1 or alpha < 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid experiment settings")
    manifest = dataset_manifest(root)
    plan = json.loads(Path(plan_path).read_text())
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(plan, labels, manifest, root)
    _new_directory(destination)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    _write_json(destination / "experiment.json", {
        "arms": ARMS, "seeds": seeds, "epochs": epochs, "patience": patience, "alpha": alpha,
        "source_sha256": _hash_file(Path(__file__)), "torch_version": torch.__version__,
        "dataset_sha256": _hash_file(root / "dataset_manifest.json"), "plan_sha256": _hash_file(Path(plan_path)),
        "selection": "validation seed-mean position error; exclude privileged diagnostics; first seed artifact",
        "test_policy": "freeze every model and selected arm before scoring; reuse of old test is exploratory",
        "rei_scope": "arxiv v1 Algorithm 1 adapted to detector events; whole-prefix endpoint, not original online/noise reproduction"})
    _write_json(destination / "split_plan.json", plan)
    raw, _, rows, spec, _ = _prepare(root, destination / "cache", min(16, manifest["stim_config"]["rounds"]-1))
    train = np.flatnonzero(rows.event.isin(plan["training_sets"][str(max(map(int, plan["training_sets"])))]))
    val = np.flatnonzero(rows.event.isin(plan["validation"]))
    models, validation = [], []
    for arm in ARMS:
        for seed in seeds:
            folder = destination / f"{arm}_seed{seed}"
            model, info, privileged = fit_arm(raw, rows, spec, train, val, arm, seed, folder,
                                              epochs=epochs, patience=patience, alpha=alpha)
            models.append((model, info, privileged))
            validation.append({k: info[k] for k in ("arm", "seed", "validation_mean_error_mm", "selected_epoch", "parameter_count", "training_seconds")})
            print(f"trained {arm} seed={seed} validation={info['validation_mean_error_mm']:.4f}", flush=True)
    validation = pd.DataFrame(validation)
    validation.to_csv(destination / "validation_runs.csv", index=False)
    ordinary = validation[~validation.arm.str.startswith("oracle")]
    selected = ordinary.groupby("arm").validation_mean_error_mm.mean().idxmin()
    _write_json(destination / "selected_model.json", {"arm": selected, "seed": seeds[0], "directory": f"{selected}_seed{seeds[0]}"})
    # No test errors used before all fitted models and their selection are frozen.
    frames = []
    truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    duration = float(np.median(np.diff(spec["geometry"]["round_time_ms"])))
    def record(name, seed, idx, prediction, split):
        frame = rows.iloc[idx][["event", "shot", "strength_band", "geometry", "propagation_law", "epicenter_region", "event_onset_ms"]].copy()
        frame["arm"], frame["seed"], frame["split"] = name, seed, split
        frame["true_x_mm"], frame["true_y_mm"] = truth[idx, 0], truth[idx, 1]
        frame["predicted_x_mm"], frame["predicted_y_mm"] = prediction[:, 0], prediction[:, 1]
        frame["error_mm"] = np.linalg.norm(prediction[:, :2]-truth[idx], axis=1)
        if prediction.shape[1] == 3:
            frame["predicted_onset_ms"] = prediction[:, 2]
            frame["onset_error_rounds"] = np.abs(prediction[:, 2]-rows.event_onset_ms.to_numpy()[idx])/duration
        frames.append(frame)
    for split in ("test_id", "test_ood"):
        idx = np.flatnonzero(rows.event.isin(plan[split]))
        for model, info, privileged in models:
            pred = predict_arrays(model, raw, idx, np.asarray(info["target_center_mm"]), info["target_scale_mm"],
                                  privileged, info["time_center_ms"], info["time_scale_ms"])
            record(info["arm"], info["seed"], idx, pred, split)
        center = truth[train].mean(0)
        constant = np.broadcast_to(center, (len(idx), 2)).copy()
        record("constant", -1, idx, constant, split)
        rei = rei_center(np.asarray(raw[idx]), spec["sites_mm"], spec["geometry"]["circuit_physical_coords_mm"])
        record("rei_v1_adapted", -1, idx, rei, split)
        fallback = np.where(np.isfinite(rei), rei, constant)
        record("rei_v1_fallback", -1, idx, fallback, split)
        from .window_calibration import _centroid
        centroid = _centroid(raw, idx, raw.shape[2]+1, spec, duration)
        record("centroid", -1, idx, centroid, split)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(destination / "test_predictions.csv", index=False)
    metrics, comparisons = [], []
    for (arm, split), group in predictions.groupby(["arm", "split"]):
        for band in ("all", 0, 1, 2):
            subset = group if band == "all" else group[group.strength_band == band]
            if not len(subset):
                continue
            valid = subset.dropna(subset=["error_mm"])
            item = {"arm": arm, "split": split, "strength_band": band, "coverage": len(valid)/len(subset)}
            if len(valid):
                item.update(summarize(valid))
                if "onset_error_rounds" in valid and valid.onset_error_rounds.notna().any():
                    item["onset_mae_rounds"] = float(valid.onset_error_rounds.mean())
                    item["constant_onset_mae_rounds"] = float(np.abs(valid.event_onset_ms-rows.iloc[train].event_onset_ms.mean()).mean()/duration)
            metrics.append(item)
        if arm not in ("original", "rei_v1_adapted"):
            base = predictions[(predictions.arm == "original") & (predictions.split == split)].groupby(["event", "shot"]).error_mm.mean()
            ours = group.groupby(["event", "shot"]).error_mm.mean()
            delta = (ours-base).rename("error_mm").reset_index()
            comparisons.append({"arm": arm, "split": split, "minus_original_mm": float(delta.error_mm.mean()),
                                "paired_event_95pct_mm": _mean_interval(delta)})
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "paired_comparisons.json", comparisons)
    return metrics


def evaluate_frozen(experiment, dataset, training_dataset, output):
    """New events only; no training, threshold tuning, or model selection here."""
    experiment, root, old, destination = map(Path, (experiment, dataset, training_dataset, output))
    settings = json.loads((experiment / "experiment.json").read_text())
    old_manifest, manifest = dataset_manifest(old), dataset_manifest(root)
    if _hash_file(old / "dataset_manifest.json") != settings["dataset_sha256"]:
        raise ValueError("training dataset identity mismatch")
    if old_manifest["circuit_id"] != manifest["circuit_id"] or old_manifest["generation"]["device_seed"] != manifest["generation"]["device_seed"]:
        raise ValueError("fresh data must use the same circuit and device")
    old_rows = pd.concat([p for _, p in iter_shards(old)], ignore_index=True)
    new_rows = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    for field in ("event_uid", "generation_seed", "syndrome_seed"):
        if set(old_rows[field]) & set(new_rows[field]):
            raise ValueError(f"fresh evaluation overlaps {field}")
    if new_rows.is_control.astype(bool).any() or (new_rows.source_count != 1).any():
        raise ValueError("this evaluation is conditional single-event localization")
    checkpoints = []
    for arm in settings["arms"]:
        for seed in settings["seeds"]:
            folder = experiment / f"{arm}_seed{seed}"
            info = json.loads((folder / "model.json").read_text())
            if _hash_file(folder / "weights.pt") != info["weights_sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            first, _ = next(iter_shards(root))
            detector_series(first, info["specification"])
            checkpoints.append((folder, info))
    _new_directory(destination)
    torch.set_num_threads(2)
    _write_json(destination / "evaluation.json", {
        "source_sha256": _hash_file(Path(__file__)), "dataset_sha256": _hash_file(root / "dataset_manifest.json"),
        "frozen_experiment_sha256": _hash_file(experiment / "experiment.json"),
        "selected_model": json.loads((experiment / "selected_model.json").read_text()),
        "checkpoint_hashes": {folder.name: info["weights_sha256"] for folder, info in checkpoints},
        "independent_events": len(new_rows), "seed_overlap_checks": "event_uid, generation_seed, syndrome_seed disjoint",
        "policy": "all fresh events are test only; no tuning or model reselection"})
    raw, _, rows, spec, _ = _prepare(root, destination / "cache", min(16, manifest["stim_config"]["rounds"]-1))
    truth = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    duration = float(np.median(np.diff(spec["geometry"]["round_time_ms"])))
    plan = json.loads((experiment / "split_plan.json").read_text())
    held = np.ones(len(rows), dtype=bool)
    for key, value in plan["held_out_conjunction"].items():
        held &= rows[key].to_numpy() == value
    frame_list = []
    def record(arm, seed, pred):
        frame = rows[["event", "shot", "event_uid", "strength_band", "geometry", "propagation_law", "epicenter_region", "event_onset_ms"]].copy()
        frame["arm"], frame["seed"] = arm, seed
        frame["split"] = np.where(held, "test_ood", "test_id")
        frame["true_x_mm"], frame["true_y_mm"] = truth[:, 0], truth[:, 1]
        frame["predicted_x_mm"], frame["predicted_y_mm"] = pred[:, 0], pred[:, 1]
        frame["error_mm"] = np.linalg.norm(pred[:, :2]-truth, axis=1)
        if pred.shape[1] == 3:
            frame["predicted_onset_ms"] = pred[:, 2]
            frame["onset_error_rounds"] = np.abs(pred[:, 2]-rows.event_onset_ms.to_numpy())/duration
        frame_list.append(frame)
    for folder, info in checkpoints:
        model = DiagnosisCNN(raw.shape[1], info["arm"])
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        privileged = oracle_values(rows, info["arm"])
        privileged = ((privileged-np.asarray(info["privileged_center"])) / np.asarray(info["privileged_scale"])).astype(np.float32)
        pred = predict_arrays(model, raw, np.arange(len(rows)), np.asarray(info["target_center_mm"]),
                              info["target_scale_mm"], privileged, info["time_center_ms"], info["time_scale_ms"])
        record(info["arm"], info["seed"], pred)
        print(f"fresh evaluated {folder.name}", flush=True)
    constant = np.broadcast_to(checkpoints[0][1]["target_center_mm"], (len(rows), 2)).copy()
    record("constant", -1, constant)
    rei = rei_center(np.asarray(raw), spec["sites_mm"], spec["geometry"]["circuit_physical_coords_mm"])
    record("rei_v1_adapted", -1, rei)
    record("rei_v1_fallback", -1, np.where(np.isfinite(rei), rei, constant))
    from .window_calibration import _centroid
    record("centroid", -1, _centroid(raw, np.arange(len(rows)), raw.shape[2]+1, spec, duration))
    frame = pd.concat(frame_list, ignore_index=True)
    frame.to_csv(destination / "test_predictions.csv", index=False)
    metrics, paired = [], []
    for (arm, split), group in frame.groupby(["arm", "split"]):
        for band in ("all", 0, 1, 2):
            subset = group if band == "all" else group[group.strength_band == band]
            if not len(subset):
                continue
            valid = subset.dropna(subset=["error_mm"])
            item = {"arm": arm, "split": split, "strength_band": band, "coverage": len(valid)/len(subset)}
            if len(valid):
                item.update(summarize(valid))
            if valid.onset_error_rounds.notna().any():
                item["onset_mae_rounds"] = float(valid.onset_error_rounds.mean())
                item["constant_onset_mae_rounds"] = float(np.abs(valid.event_onset_ms-checkpoints[0][1]["time_center_ms"]).mean()/duration)
            metrics.append(item)
            if arm not in ("original", "rei_v1_adapted"):
                base = frame[(frame.arm == "original") & (frame.split == split)]
                if band != "all":
                    base = base[base.strength_band == band]
                delta = (subset.groupby(["event", "shot"]).error_mm.mean()-base.groupby(["event", "shot"]).error_mm.mean()).rename("error_mm").reset_index()
                paired.append({"arm": arm, "split": split, "strength_band": band,
                               "minus_original_mm": float(delta.error_mm.mean()), "paired_event_95pct_mm": _mean_interval(delta)})
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "paired_comparisons.json", paired)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--frozen-experiment", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--alpha", type=float, default=.1)
    args = parser.parse_args()
    if args.frozen_experiment:
        if args.training_dataset is None:
            parser.error("--training-dataset is required for frozen evaluation")
        evaluate_frozen(args.frozen_experiment, args.dataset, args.training_dataset, args.output)
    else:
        if args.plan is None:
            parser.error("--plan is required for training")
        run(args.dataset, args.plan, args.output, seeds=args.seeds, epochs=args.epochs, patience=args.patience, alpha=args.alpha)


if __name__ == "__main__":
    main()
