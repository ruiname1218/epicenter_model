"""CLI for the synthetic propagation ground-truth benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..configuration import config_path, load_json
from ..propagation_validation import run_validation, save_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate propagation parameter recovery on synthetic ground truth"
    )
    parser.add_argument(
        "--config", type=Path, default=config_path("propagation_validation")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("propagation_validation_output")
    )
    args = parser.parse_args()

    frame, summary = run_validation(load_json(args.config))
    save_validation(frame, summary, args.output)
    ballistic = summary["by_law"]["ballistic"]
    diffusive = summary["by_law"]["diffusive"]
    print(f"validated {len(frame)} synthetic propagation fields")
    print(
        "mean absolute error: "
        f"speed {ballistic['propagation_mean_absolute_percent_error']:.1f}%, "
        f"diffusion {diffusive['propagation_mean_absolute_percent_error']:.1f}%"
    )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
