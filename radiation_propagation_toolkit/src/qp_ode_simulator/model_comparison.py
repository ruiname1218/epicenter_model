"""Model-only comparison on the frozen 57-feature representation.

External event-group validation selects all hyperparameters and MLP epochs.
No internal shot-level validation split and no test-based reselection.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .adaptive_localization import load_model, segmented_features
from .dataset import _write_json, dataset_manifest, iter_shards
from .learning_plan import validate_plan
from .localization import _hash_file, _new_directory
from .temporal_localization import _prepare, detector_series


class FeatureMLP:
    """Serializable MLP with train-only input/target normalization."""
    def __init__(self, scaler, estimator, center, scale):
        self.scaler, self.estimator, self.center, self.scale = scaler, estimator, center, scale

    def predict(self, values):
        return self.estimator.predict(self.scaler.transform(values))*self.scale+self.center


def fit_mlp(x, y, train, val, *, seed, hidden, alpha, epochs=150, patience=15):
    scaler = StandardScaler().fit(x[train])
    center, scale = y[train].mean(0), max(float(y[train].std()), 1e-6)
    normalized = scaler.transform(x)
    estimator = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=.001,
                             batch_size=min(64, len(train)), random_state=seed, early_stopping=False)
    best, error, chosen, history = None, np.inf, 0, []
    for epoch in range(1, epochs+1):
        estimator.partial_fit(normalized[train], (y[train]-center)/scale)
        prediction = estimator.predict(normalized[val])*scale+center
        current = float(np.linalg.norm(prediction-y[val], axis=1).mean())
        history.append({"epoch": epoch, "validation_error_mm": current})
        if current < error:
            best, error, chosen = copy.deepcopy(estimator), current, epoch
        if epoch-chosen >= patience:
            break
    if best is None:
        raise ValueError("MLP produced no finite checkpoint")
    return FeatureMLP(scaler, best, center, scale), {"seed": seed, "selected_epoch": chosen, "history": history}


def configurations():
    """Small fixed grids; not an exhaustive best-possible comparison."""
    settings = []
    for alpha in (10., 100., 1000.):
        settings.append({"family": "ridge", "alpha": alpha})
    for hidden, alpha in (((32,), 1.), ((64, 32), .1), ((64, 32), 10.)):
        settings.append({"family": "mlp", "hidden": hidden, "alpha": alpha})
    for family in ("random_forest", "extra_trees"):
        for leaf in (4, 12):
            settings.append({"family": family, "leaf": leaf})
    for leaf in (8, 16):
        settings.append({"family": "hist_gradient_boosting", "leaf": leaf})
    for family in ("kernel_ridge", "svr"):
        for gamma in (.001, .01):
            for regularization in (1., 10.):
                settings.append({"family": family, "gamma": gamma, "regularization": regularization})
    return settings


def fit(dataset, parent_experiment, output):
    root, parent, destination = map(Path, (dataset, parent_experiment, output))
    info, previous = load_model(parent)
    if info.get("inference_only") or info["dataset_sha256"] != _hash_file(root / "dataset_manifest.json"):
        raise ValueError("use the full parent experiment with matching training data")
    manifest = dataset_manifest(root)
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    validate_plan(info["plan"], labels, manifest, root)
    detector_series(next(iter_shards(root))[0], info["specification"])
    _new_directory(destination)
    _write_json(destination / "grid.json", {"configurations": configurations(), "seeds": [41,42,43],
                "mlp_max_epochs": 150, "mlp_patience": 15, "selection": "external validation mean Euclidean error",
                "source_sha256": _hash_file(Path(__file__)), "parent_sha256": info["artifact_sha256"]})
    raw, _, rows, _, _ = _prepare(root, destination / "cache", 16)
    x = segmented_features(raw, np.full(len(raw), info["fixed_cut"]), info["quiet"])
    y = rows[["epicenter_row", "epicenter_col"]].to_numpy()
    plan = info["plan"]
    train = np.flatnonzero(rows.event.isin(plan["training_sets"][str(max(map(int, plan["training_sets"])))]))
    val = np.flatnonzero(rows.event.isin(plan["validation"]))
    candidates = [c for c in previous["candidates"] if c["name"] in ("cnn_single", "cnn_ensemble", "constant")]
    validation, histories = [], []
    for index, config in enumerate(configurations()):
        family, begin, models = config["family"], time.perf_counter(), []
        name = "fixed_ridge_100.0" if family == "ridge" and config["alpha"] == 100 else f"{family}_{index:02d}"
        if family == "ridge":
            models = [make_pipeline(StandardScaler(), Ridge(alpha=config["alpha"])).fit(x[train], y[train])]
        elif family == "mlp":
            for seed in (41,42,43):
                model, history = fit_mlp(x, y, train, val, seed=seed, hidden=config["hidden"], alpha=config["alpha"])
                models.append(model)
                histories.append({"candidate": name, **history})
        elif family in ("random_forest", "extra_trees"):
            cls = RandomForestRegressor if family == "random_forest" else ExtraTreesRegressor
            models = [cls(n_estimators=128, min_samples_leaf=config["leaf"], n_jobs=2, random_state=seed).fit(x[train], y[train])
                      for seed in (41,42,43)]
        elif family == "hist_gradient_boosting":
            models = [MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=config["leaf"],
                       learning_rate=.05, l2_regularization=10., early_stopping=False, random_state=41)).fit(x[train], y[train])]
        elif family == "kernel_ridge":
            models = [make_pipeline(StandardScaler(), KernelRidge(alpha=config["regularization"], kernel="rbf", gamma=config["gamma"])).fit(x[train], y[train])]
        elif family == "svr":
            models = [make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=config["regularization"], epsilon=.1,
                       gamma=config["gamma"], kernel="rbf"))).fit(x[train], y[train])]
        prediction = np.mean([model.predict(x[val]) for model in models], axis=0)
        error = float(np.linalg.norm(prediction-y[val], axis=1).mean())
        candidates.append({"name": name, "kind": "regressor", "representation": "fixed", "models": models, "family": family, "configuration": config})
        validation.append({"name": name, "family": family, "validation_error_mm": error, "training_seconds": time.perf_counter()-begin})
        print(f"{name}: validation={error:.5f} mm", flush=True)
    table = pd.DataFrame(validation)
    table.to_csv(destination / "validation.csv", index=False)
    # Family winners and overall winner are locked before any test evaluation.
    winners = table.loc[table.groupby("family").validation_error_mm.idxmin()]
    selected = table.sort_values("validation_error_mm", kind="stable").iloc[0]["name"]
    info = {**info, "selected": selected, "parent_artifact_sha256": info["artifact_sha256"],
            "comparison_source_sha256": _hash_file(Path(__file__)),
            "selection": "minimum validation error among 20 configurations with identical 57 features; CNNs external references",
            "family_winners": dict(zip(winners.family, winners.name))}
    joblib.dump({"candidates": candidates, "states": previous["states"]}, destination / "models.joblib", compress=3)
    info["artifact_sha256"] = _hash_file(destination / "models.joblib")
    _write_json(destination / "model.json", info)
    _write_json(destination / "mlp_histories.json", histories)
    print("FROZEN SELECTION:", selected, flush=True)
    print(winners.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent-experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fit(args.dataset, args.parent_experiment, args.output)


if __name__ == "__main__":
    # Keep serialized FeatureMLP references importable outside this CLI process.
    from .model_comparison import main as entrypoint
    entrypoint()
