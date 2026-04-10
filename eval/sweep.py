#!/usr/bin/env python3
from __future__ import annotations

"""Run the primary verbose multi-model eval sweep."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import DEFAULT_SWEEP_RESULTS_ROOT, build_sweep_arg_parser, run_sweep_from_namespace


def main() -> int:
    parser = build_sweep_arg_parser(
        description=(
            "Run the local KDictate eval sweep with full runtime stats, "
            "token-level reference/hypothesis diffs, and per-run reports."
        ),
        default_results_root=DEFAULT_SWEEP_RESULTS_ROOT,
        default_preset="accuracy-bakeoff",
    )
    return run_sweep_from_namespace(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
