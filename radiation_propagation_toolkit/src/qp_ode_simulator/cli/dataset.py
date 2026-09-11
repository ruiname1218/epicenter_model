"""Generate, resume and audit compact labelled syndrome shards."""

import argparse
from pathlib import Path

from ..configuration import config_path, load_json
from ..dataset import audit_dataset, generate_dataset
from ..simulator import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, default=config_path("simulator_base"))
    generate.add_argument("--profile", type=Path, default=config_path("qp_ode_generic"))
    generate.add_argument("--stim-config", type=Path, default=config_path("stim_surface_code_d3"))
    generate.add_argument("--sampling", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--n-events", type=int, required=True)
    generate.add_argument("--batch-size", type=int, default=16)
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device-seed", type=int, default=123)
    generate.add_argument("--syndrome-seed", type=int, default=456)
    generate.add_argument("--resume", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--windows-ms", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate_dataset(
            load_config(args.config, args.profile), load_json(args.stim_config),
            load_json(args.sampling), args.output, n_events=args.n_events,
            batch_size=args.batch_size, seed=args.seed, device_seed=args.device_seed,
            syndrome_seed=args.syndrome_seed, resume=args.resume, workers=args.workers,
        )
    else:
        report = audit_dataset(args.dataset, args.output, args.windows_ms)
        print(f"audited {report['events']} events at {len(report['windows_ms'])} window lengths")


if __name__ == "__main__":
    main()
