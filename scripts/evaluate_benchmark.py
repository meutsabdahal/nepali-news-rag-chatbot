from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, UTC

from nepali_news_rag.pipeline import NepaliNewsPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark questions against pipeline"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only first N benchmark items (0 means all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="latest_results.json",
        help="Output filename inside evaluation/results",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    root = Path(__file__).resolve().parent.parent
    benchmark_path = root / "evaluation" / "benchmark_questions.json"
    results_dir = root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        bench = bench[: args.limit]

    pipeline = NepaliNewsPipeline()

    results = []
    matched_routes = 0
    for item in bench:
        try:
            output = pipeline.run(item["question"])
            actual_route = output["route"]
            answer = output["answer"]
            sources = output.get("sources", [])
            error = None
        except Exception as exc:
            actual_route = "ERROR"
            answer = ""
            sources = []
            error = str(exc)

        if actual_route == item["expected_route"]:
            matched_routes += 1

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_route": item["expected_route"],
                "actual_route": actual_route,
                "answer": answer,
                "sources": sources,
                "error": error,
            }
        )

    summary = {
        "run_at_utc": datetime.now(UTC).isoformat(),
        "total": len(results),
        "route_match_count": matched_routes,
        "route_match_rate": (matched_routes / len(results)) if results else 0.0,
    }
    payload = {"summary": summary, "results": results}

    out_file = results_dir / args.output
    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved benchmark results to {out_file}")
    print(
        f"Route match: {summary['route_match_count']}/{summary['total']} "
        f"({summary['route_match_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
