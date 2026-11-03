"""Command-line entry point: `subway <stage>` (or `subway all`)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    from .pipeline import STAGES, run

    p = argparse.ArgumentParser(prog="subway", description="NYC subway per-trip occupancy pipeline")
    p.add_argument("stage", choices=STAGES + ["all"])
    run(p.parse_args(argv).stage)


if __name__ == "__main__":
    main()
