"""CLI for exact-GAD versus PTGAD validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..channel_validation import make_figure, run_validation, write_report
from ..configuration import config_path, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Stim PTGAD approximation")
    parser.add_argument("--config", type=Path, default=config_path("channel_validation"))
    parser.add_argument("--output", type=Path, default=Path("channel_validation_output"))
    args = parser.parse_args()

    config = load_json(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    frame, summary = run_validation(config)
    frame.to_csv(args.output / "qec_channel_validation.csv", index=False)
    (args.output / "qec_channel_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    make_figure(frame, args.output, config["screening_thresholds"])
    write_report(summary, args.output)
    print(f"validated {len(frame)} exact/PTGAD circuit cases")
    print(f"screening pass fraction: {summary['overall_pass_fraction']:.1%}")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
