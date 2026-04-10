#!/usr/bin/env python3
from __future__ import annotations

"""Run a single-model verbose evaluation over the local manifest."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import DEFAULT_SINGLE_RESULTS_ROOT, build_single_arg_parser, run_single_from_namespace


def main() -> int:
    parser = build_single_arg_parser(
        description=(
            "Run a single-model KDictate eval with full runtime stats, "
            "side-by-side transcript diffs, and a saved markdown report."
        ),
        default_results_root=DEFAULT_SINGLE_RESULTS_ROOT,
    )
    return run_single_from_namespace(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
