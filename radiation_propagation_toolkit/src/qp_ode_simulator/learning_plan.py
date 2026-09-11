"""Immutable event-group splits and nested training sets for learning curves."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import _identity, _write_json, dataset_manifest, iter_shards
from .localization import _hash_file


STRATA = ["geometry", "propagation_law", "epicenter_region", "strength_band"]


def _observation_identity(manifest):
    generation = dict(manifest["generation"])
    generation.pop("n_events")
    return _identity({"generation": generation, "stim_config": manifest["stim_config"]})


def create_plan(dataset, output, *, sizes=(180, 360, 720), seed=90210,
                held_out=None):
    """Freeze disjoint splits; hold out a conjunction, never individual shots.

    This planner accepts independent events, not paired-variant datasets. A
    family_id column, if present, must be unique to avoid silently splitting it.
    """
    root, path = Path(dataset), Path(output)
    if path.exists():
        raise ValueError("plan already exists; reuse it, do not overwrite evaluation splits")
    manifest = dataset_manifest(root)
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    if labels.is_control.astype(bool).any() or (labels.source_count != 1).any():
        raise ValueError("plan requires single-source non-control events")
    for key in ("event", "event_uid", "family_id"):
        if key in labels and labels[key].duplicated().any():
            raise ValueError(f"duplicate {key}; paired variants require family-level planning")
    held = held_out if held_out is not None else {"geometry": "elliptical", "strength_band": 2}
    if not held or not set(held) <= set(STRATA):
        raise ValueError("held_out must be a nonempty conjunction of stratum values")
    sizes = list(sizes)
    if not sizes or any(type(n) is not int or n < 1 for n in sizes) or sizes != sorted(set(sizes)):
        raise ValueError("training sizes must be distinct increasing positive integers")
    rng = np.random.default_rng(seed)
    train_groups, validation, test_id, test_ood, excluded = [], [], [], [], []
    for _, group in labels.groupby(STRATA, sort=True):
        if len(group) < 5:
            raise ValueError("each stratum needs at least five independent events")
        ids = rng.permutation(group.event.to_numpy(dtype=int)).tolist()
        n = max(1, len(ids) // 5)
        is_held = all(group.iloc[0][k] == v for k, v in held.items())
        if is_held:
            test_ood.extend(ids[:n])
            excluded.extend(ids[n:])
        else:
            test_id.extend(ids[:n])
            validation.extend(ids[n:2*n])
            train_groups.append(ids[2*n:])
    if not test_id or not test_ood or not validation:
        raise ValueError("held-out conjunction must leave both seen and unseen conditions")
    # Interleave strata, so small prefixes cover all eligible conditions.
    order = rng.permutation(len(train_groups))
    ranked = [int(v) for row in itertools.zip_longest(*(train_groups[i] for i in order))
              for v in row if v is not None]
    if sizes[-1] > len(ranked):
        raise ValueError(f"largest training size exceeds eligible pool ({len(ranked)})")
    plan = {
        "format": "epicenter_learning_plan_v1", "seed": seed,
        "dataset_manifest_sha256": _hash_file(root / "dataset_manifest.json"),
        "source_id": manifest["source_id"], "circuit_id": manifest["circuit_id"],
        "observation_identity": _observation_identity(manifest),
        "held_out_conjunction": held,
        "train_pool": ranked, "validation": sorted(validation),
        "test_id": sorted(test_id), "test_ood": sorted(test_ood), "excluded": sorted(excluded),
        "training_sets": {str(n): ranked[:n] for n in sizes},
        "event_uids": dict(zip(labels.event.astype(str), labels.event_uid)),
        "split_unit": "independent source event; all shots kept together",
    }
    validate_plan(plan, labels, manifest, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, plan)
    return plan


def extend_plan(previous_path, dataset, output, *, sizes):
    """Expand only the training pool while preserving evaluation events/bits.

    The new dataset must use identical source/circuit/noise seeds and code/runtime
    but a larger declared event count. This does not reopen test-based tuning.
    """
    import copy
    import json
    previous_path, root, path = Path(previous_path), Path(dataset), Path(output)
    if path.exists():
        raise ValueError("plan output already exists")
    previous = json.loads(previous_path.read_text())
    manifest = dataset_manifest(root)
    if (previous["source_id"] != manifest["source_id"] or previous["circuit_id"] != manifest["circuit_id"]
            or previous["observation_identity"] != _observation_identity(manifest)):
        raise ValueError("source or observation generation changed; cannot preserve evaluation")
    labels = pd.concat([p for _, p in iter_shards(root)], ignore_index=True)
    uids = dict(zip(labels.event.astype(str), labels.event_uid))
    if len(uids) <= len(previous["event_uids"]) or any(uids.get(k) != v for k, v in previous["event_uids"].items()):
        raise ValueError("extended dataset must retain all prior events and add new ones")
    sizes = list(sizes)
    if (not sizes or any(type(n) is not int or n < 1 for n in sizes)
            or sizes != sorted(set(sizes)) or not set(previous["training_sets"]) <= {str(n) for n in sizes}):
        raise ValueError("sizes must be increasing and retain every previous learning-curve size")
    fresh = labels[~labels.event.astype(str).isin(previous["event_uids"])]
    rng = np.random.default_rng(previous["seed"])
    groups, excluded = [], []
    for _, group in fresh.groupby(STRATA, sort=True):
        ids = rng.permutation(group.event.to_numpy(dtype=int)).tolist()
        if all(group.iloc[0][k] == v for k, v in previous["held_out_conjunction"].items()):
            excluded.extend(ids)
        else:
            groups.append(ids)
    order = rng.permutation(len(groups))
    addition = [int(v) for row in itertools.zip_longest(*(groups[i] for i in order))
                for v in row if v is not None]
    result = copy.deepcopy(previous)
    result["train_pool"] += addition
    if sizes[-1] > len(result["train_pool"]):
        raise ValueError("largest training size exceeds expanded pool")
    result["excluded"] += sorted(excluded)
    result["training_sets"] = {str(n): result["train_pool"][:n] for n in sizes}
    result["event_uids"] = uids
    result["dataset_manifest_sha256"] = _hash_file(root / "dataset_manifest.json")
    result["parent_plan_sha256"] = _hash_file(previous_path)
    validate_plan(result, labels, manifest, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, result)
    return result


def validate_plan(plan, labels, manifest, root):
    if plan.get("format") != "epicenter_learning_plan_v1":
        raise ValueError("unsupported split plan")
    if labels.is_control.astype(bool).any() or (labels.source_count != 1).any():
        raise ValueError("plan requires single-source non-control events")
    if any(labels[k].duplicated().any() for k in ("event", "event_uid", "family_id") if k in labels):
        raise ValueError("duplicate event/family identity")
    if (plan["dataset_manifest_sha256"] != _hash_file(Path(root) / "dataset_manifest.json")
            or plan["source_id"] != manifest["source_id"] or plan["circuit_id"] != manifest["circuit_id"]
            or plan["observation_identity"] != _observation_identity(manifest)):
        raise ValueError("split plan belongs to another dataset or changed manifest")
    groups = [plan[k] for k in ("train_pool", "validation", "test_id", "test_ood", "excluded")]
    if any(not g for g in groups[:4]) or not plan["training_sets"]:
        raise ValueError("training, validation and both tests must be nonempty")
    flat = sum(groups, [])
    if len(set(flat)) != len(flat) or set(flat) != set(labels.event):
        raise ValueError("split groups overlap or fail to cover events")
    if plan["event_uids"] != dict(zip(labels.event.astype(str), labels.event_uid)):
        raise ValueError("source event identities changed")
    indexed = labels.set_index("event")
    held = pd.Series(True, index=indexed.index)
    for key, value in plan["held_out_conjunction"].items():
        held &= indexed[key] == value
    if held.loc[plan["train_pool"] + plan["validation"] + plan["test_id"]].any():
        raise ValueError("held-out conditions leaked into training/validation/ID test")
    if not held.loc[plan["test_ood"]].all():
        raise ValueError("OOD test contains seen conditions")
    for size, ids in plan["training_sets"].items():
        if len(ids) != int(size) or ids != plan["train_pool"][:int(size)]:
            raise ValueError("training sets must be nested prefixes of frozen training pool")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[180, 360, 720])
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--held-out", type=Path,
                        help="JSON conjunction of geometry/propagation_law/epicenter_region/strength_band values")
    parser.add_argument("--extend", type=Path, help="prior plan; preserve its evaluation events in a larger dataset")
    args = parser.parse_args()
    import json
    if args.extend:
        if args.held_out:
            parser.error("cannot change held-out rule while extending")
        result = extend_plan(args.extend, args.dataset, args.output, sizes=args.sizes)
    else:
        held = json.loads(args.held_out.read_text()) if args.held_out else None
        result = create_plan(args.dataset, args.output, sizes=args.sizes, seed=args.seed, held_out=held)
    print({k: len(result[k]) for k in ("train_pool", "validation", "test_id", "test_ood", "excluded")})


if __name__ == "__main__":
    main()
