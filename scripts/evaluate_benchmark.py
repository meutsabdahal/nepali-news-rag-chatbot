from __future__ import annotations

import json
from pathlib import Path

from nepali_news_rag.pipeline import NepaliNewsPipeline


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    benchmark_path = root / "evaluation" / "benchmark_questions.json"
    results_dir = root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
    pipeline = NepaliNewsPipeline()

    results = []
    for item in bench:
        output = pipeline.run(item["question"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_route": item["expected_route"],
                "actual_route": output["route"],
                "answer": output["answer"],
                "sources": output.get("sources", []),
            }
        )

    out_file = results_dir / "latest_results.json"
    out_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
