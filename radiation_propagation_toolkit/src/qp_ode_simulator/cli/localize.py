"""Train or apply a single-shot syndrome-to-epicenter regressor."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..localization import predict_epicenters, train_localizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate on event-separated splits")
    train.add_argument("--dataset", type=Path, required=True, help="directory saved by qp-ode-stim")
    train.add_argument("--output", type=Path, required=True, help="new or empty model directory")
    train.add_argument("--bins", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--trees", type=int, default=256)
    train.add_argument("--jobs", type=int, default=2)
    train.add_argument("--epochs", type=int, default=20, help="MLP epochs for a sharded dataset")
    train.add_argument("--batch-size", type=int, default=256, help="MLP mini-batch size")
    predict = commands.add_parser("predict", help="infer coordinates without simulator truth")
    predict.add_argument("--model", type=Path, required=True, help="trusted model.joblib")
    predict.add_argument("--input", type=Path, required=True, help="Stim NPZ or output directory")
    predict.add_argument("--output", type=Path, required=True, help="new predictions CSV")
    args = parser.parse_args()
    if args.command == "train":
        if (args.dataset / "dataset_manifest.json").is_file():
            from ..streaming_localization import train_sharded_localizer
            report = train_sharded_localizer(
                args.dataset, args.output, bins=args.bins, seed=args.seed,
                epochs=args.epochs, batch_size=args.batch_size,
            )
        else:
            report = train_localizer(
                args.dataset, args.output, bins=args.bins, seed=args.seed,
                trees=args.trees, jobs=args.jobs,
            )
        result = report["evaluation"]["test"]["model"]
        print(f"test events: {report['independent_events']['test']}; single-shot predictions: {result['samples']}")
        print(f"mean / median / p90 error: {result['mean_error_mm']:.3f} / {result['median_error_mm']:.3f} / {result['p90_error_mm']:.3f} mm")
        print(f"saved model and evaluation to {args.output}")
    else:
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.input.is_dir() and (args.input / "dataset_manifest.json").is_file():
            from ..dataset import dataset_manifest
            manifest = dataset_manifest(args.input)
            # Validate shards and checksums before predicting, without reading labels as features.
            from ..dataset import iter_shards
            for _ in iter_shards(args.input):
                pass
            count = 0
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            for shard in manifest["shards"]:
                frame = predict_epicenters(args.model, args.input / shard["npz"])
                frame.to_csv(temporary, index=False, mode="w" if count == 0 else "a", header=count == 0)
                count += len(frame)
            temporary.replace(args.output)
        else:
            frame = predict_epicenters(args.model, args.input)
            frame.to_csv(args.output, index=False)
            count = len(frame)
        print(f"saved {count} single-shot (x_mm, y_mm) predictions to {args.output}")


if __name__ == "__main__":
    main()
