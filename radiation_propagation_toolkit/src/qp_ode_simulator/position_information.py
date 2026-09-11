"""Diagnostic only: classify each coordinate's side, separately by true strength.

This easier task is not an information-theoretic bound. True strength is used
to route samples, and the onset-aware features require privileged timing.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import _hash_file, _new_directory
from .temporal_localization import _prepare, _json_spec, detector_series


def timing_features(raw, onset, spec):
    """Mean rates before/after true physical onset; missing sides are rejected."""
    times = np.asarray(spec["geometry"]["round_time_ms"])[1:]
    after = times[None, :] >= np.asarray(onset)[:, None]
    if np.any(after.sum(1) == 0) or np.any((~after).sum(1) == 0):
        raise ValueError("timing diagnostic needs observed pre- and post-onset rounds")
    pre = np.einsum("nst,nt->ns", raw, (~after).astype(float)) / (~after).sum(1)[:, None]
    post = np.einsum("nst,nt->ns", raw, after.astype(float)) / after.sum(1)[:, None]
    return np.concatenate((pre, post, post-pre), axis=1)


def train(dataset, plan_path, output):
    root, output = Path(dataset), Path(output)
    manifest = dataset_manifest(root)
    plan = json.loads(Path(plan_path).read_text())
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(plan, labels, manifest, root)
    _new_directory(output)
    raw, flat, rows, spec, _ = _prepare(root, output / "cache", min(16, manifest["stim_config"]["rounds"]-1))
    known = timing_features(raw, rows.event_onset_ms.to_numpy(), spec)
    sets = {"ordinary_features": flat, "oracle_timing": known}
    train_ids = plan["training_sets"][str(max(map(int, plan["training_sets"])))]
    midpoint = np.asarray(spec["layout_center_mm"])
    fitted, choices = [], []
    # Small declared regularization grid; no test predictions during training.
    for name, features in sets.items():
        for band in (0, 1, 2):
            tr = np.flatnonzero(rows.event.isin(train_ids) & rows.strength_band.eq(band))
            va = np.flatnonzero(rows.event.isin(plan["validation"]) & rows.strength_band.eq(band))
            for axis, column in enumerate(("epicenter_row", "epicenter_col")):
                target = (rows[column].to_numpy() >= midpoint[axis]).astype(int)
                if len(np.unique(target[tr])) < 2 or len(np.unique(target[va])) < 2:
                    raise ValueError("both coordinate sides required in train and validation")
                best, score, selected = None, -np.inf, None
                for regularization in (.001, .01, .1, 1.):
                    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=2000, random_state=41))
                    estimator.fit(features[tr], target[tr])
                    auc = roc_auc_score(target[va], estimator.predict_proba(features[va])[:, 1])
                    if auc > score:
                        best, score, selected = estimator, float(auc), regularization
                fitted.append({"feature_kind": name, "band": band, "axis": axis, "estimator": best})
                choices.append({"feature_kind": name, "band": band, "axis": axis, "C": selected, "validation_auc": score})
    joblib.dump(fitted, output / "diagnostic.joblib")
    _write_json(output / "diagnostic.json", {
        "diagnostic_only": True, "source_sha256": _hash_file(Path(__file__)), "dataset_sha256": _hash_file(root / "dataset_manifest.json"),
        "artifact_sha256": _hash_file(output / "diagnostic.joblib"), "plan": plan,
        "specification": _json_spec(spec), "midpoint_mm": midpoint.tolist(), "validation_choices": choices,
        "scope": "true strength routed classifiers, not deployable localization or impossibility proof"})
    print(pd.DataFrame(choices).to_string(index=False), flush=True)


def evaluate(model, dataset, training_dataset, output):
    model, root, old, output = map(Path, (model, dataset, training_dataset, output))
    info = json.loads((model / "diagnostic.json").read_text())
    if _hash_file(model / "diagnostic.joblib") != info["artifact_sha256"]:
        raise ValueError("diagnostic artifact checksum mismatch")
    if _hash_file(old / "dataset_manifest.json") != info["dataset_sha256"]:
        raise ValueError("training dataset mismatch")
    old_labels = pd.concat([p for _, p in iter_shards(old)], ignore_index=True)
    fresh_labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    for key in ("event_uid", "generation_seed", "syndrome_seed"):
        if set(old_labels[key]) & set(fresh_labels[key]):
            raise ValueError("diagnostic test overlaps training source")
    first, _ = next(iter_shards(root))
    detector_series(first, info["specification"])
    _new_directory(output)
    raw, flat, rows, spec, _ = _prepare(root, output / "cache", info["specification"]["bins"])
    known = timing_features(raw, rows.event_onset_ms.to_numpy(), spec)
    features = {"ordinary_features": flat, "oracle_timing": known}
    predictions, metrics = [], []
    held = np.ones(len(rows), dtype=bool)
    for key, value in info["plan"]["held_out_conjunction"].items():
        held &= rows[key].to_numpy() == value
    for item in joblib.load(model / "diagnostic.joblib"):
        axis, band, name = item["axis"], item["band"], item["feature_kind"]
        for split, mask in (("test_id", ~held), ("test_ood", held)):
            idx = np.flatnonzero(rows.strength_band.eq(band) & mask)
            if len(idx) == 0:
                continue
            column = ("epicenter_row", "epicenter_col")[axis]
            target = (rows.iloc[idx][column].to_numpy() >= info["midpoint_mm"][axis]).astype(int)
            probability = item["estimator"].predict_proba(features[name][idx])[:, 1]
            frame = rows.iloc[idx][["event", "shot"]].copy()
            frame["feature_kind"], frame["band"], frame["axis"], frame["split"] = name, band, axis, split
            frame["target"], frame["probability"] = target, probability
            predictions.append(frame)
            # Resample whole source events (including both shots), not individual bits/shots.
            ids = frame.event.unique()
            by_event = [frame[frame.event == event][["target", "probability"]].to_numpy() for event in ids]
            rng, boot = np.random.default_rng(2718), []
            for _ in range(500):
                sample = np.concatenate([by_event[j] for j in rng.integers(len(ids), size=len(ids))])
                if len(np.unique(sample[:, 0])) == 2:
                    boot.append(roc_auc_score(sample[:, 0], sample[:, 1]))
            metrics.append({"feature_kind": name, "band": band, "axis": axis, "split": split,
                            "events": len(ids), "auc": float(roc_auc_score(target, probability)),
                            "balanced_accuracy": float(balanced_accuracy_score(target, probability >= .5)),
                            "auc_event_95pct": np.quantile(boot, [.025, .975]).tolist()})
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "evaluation.json", {"source_sha256": _hash_file(Path(__file__)),
                "model_sha256": info["artifact_sha256"], "dataset_sha256": _hash_file(root / "dataset_manifest.json"),
                "policy": "frozen diagnostic models, fresh events only; no retuning"})
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--model")
    parser.add_argument("--training-dataset")
    args = parser.parse_args()
    if args.model:
        if not args.training_dataset:
            parser.error("--training-dataset required")
        evaluate(args.model, args.dataset, args.training_dataset, args.output)
    else:
        if not args.plan:
            parser.error("--plan required")
        train(args.dataset, args.plan, args.output)


if __name__ == "__main__":
    main()
