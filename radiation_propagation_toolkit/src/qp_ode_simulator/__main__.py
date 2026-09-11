"""Show the installed toolkit and its commands."""

from . import __version__


def main() -> None:
    print(f"qp-ode-simulator {__version__}")
    print("Commands: qp-ode-simulate, qp-ode-stim, qp-ode-syndrome, qp-ode-validate-channel")


if __name__ == "__main__":
    main()
