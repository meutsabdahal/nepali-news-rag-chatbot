from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run smoke checks for the project")
    parser.add_argument(
        "--with-benchmark",
        action="store_true",
        help="Include benchmark run (requires ready vector store and model availability)",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=2,
        help="Number of benchmark questions to run in smoke mode",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parent.parent

    _run(
        [sys.executable, "-m", "compileall", "nepali_news_rag", "app", "scripts"], root
    )
    _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        root,
    )

    if args.with_benchmark:
        _run(
            [
                sys.executable,
                "scripts/evaluate_benchmark.py",
                "--limit",
                str(args.benchmark_limit),
            ],
            root,
        )

    print("Smoke checks passed.")


if __name__ == "__main__":
    main()
