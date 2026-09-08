"""Run seal-oriented retrieval ablation, benchmark, and fault aggregation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from evaluation.benchmark import evaluate_benchmark, write_benchmark_report
from evaluation.fault_report import evaluate_fault_injection_report, write_fault_report
from evaluation.retrieval_ablation import (
    evaluate_retrieval_ablation,
    format_ablation_table,
    format_category_table,
    write_ablation_report,
)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


async def _run(
    root: Path,
    *,
    live: bool,
    rewrite_model: str,
    rewrite_api_key: str,
    rewrite_base_url: str | None,
    rerank_model: str,
) -> dict[str, object]:
    ablation = await evaluate_retrieval_ablation(
        root / "ablation",
        live=live,
        rewrite_model=rewrite_model,
        rewrite_api_key=rewrite_api_key,
        rewrite_base_url=rewrite_base_url,
        rerank_model=rerank_model,
    )
    benchmark = await evaluate_benchmark(
        root / "benchmark",
        live=live,
        rewrite_model=rewrite_model,
        rewrite_api_key=rewrite_api_key,
        rewrite_base_url=rewrite_base_url,
        rerank_model=rerank_model,
    )
    fault = await evaluate_fault_injection_report(root / "fault-report")
    return {
        "retrieval_ablation": ablation,
        "benchmark": benchmark,
        "fault_injection_report": fault,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real ModelScope LLM rewrite + sentence-transformers CrossEncoder",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("artifacts/evaluation/retrieval_ablation_latest.json"),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("artifacts/evaluation/benchmark_latest.json"),
    )
    parser.add_argument(
        "--fault-output",
        type=Path,
        default=Path("artifacts/evaluation/fault_injection_report_latest.json"),
    )
    parser.add_argument(
        "--rewrite-model",
        default=os.environ.get("OMNIAGENT_REWRITE_MODEL", "Qwen/Qwen3.5-35B-A3B"),
    )
    parser.add_argument(
        "--rewrite-base-url",
        default=os.environ.get(
            "OMNIAGENT_REWRITE_BASE_URL",
            "https://api-inference.modelscope.cn/v1",
        ),
    )
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    args = parser.parse_args()
    _load_dotenv(Path(".env.omniagent"))
    rewrite_api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    if args.live and not rewrite_api_key:
        raise SystemExit("MODELSCOPE_API_KEY is required for --live")

    tmp = tempfile.mkdtemp(prefix="omniagent-seal-")
    try:
        bundle = asyncio.run(
            _run(
                Path(tmp),
                live=args.live,
                rewrite_model=args.rewrite_model,
                rewrite_api_key=rewrite_api_key,
                rewrite_base_url=args.rewrite_base_url,
                rerank_model=args.rerank_model,
            )
        )
    finally:
        try:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass
    ablation = bundle["retrieval_ablation"]
    benchmark = bundle["benchmark"]
    fault = bundle["fault_injection_report"]
    assert isinstance(ablation, dict)
    assert isinstance(benchmark, dict)
    assert isinstance(fault, dict)
    write_ablation_report(ablation, args.ablation_output)
    write_benchmark_report(benchmark, args.benchmark_output)
    write_fault_report(fault, args.fault_output)
    print(format_ablation_table(ablation))
    print()
    print(format_category_table(ablation))
    print()
    print(
        json.dumps(
            {
                "mode": "live" if args.live else "offline",
                "ablation_output": str(args.ablation_output),
                "benchmark_output": str(args.benchmark_output),
                "fault_output": str(args.fault_output),
                "ablation_case_count": ablation.get("case_count"),
                "stage_latency_ms": ablation.get("stage_latency_ms"),
                "fault_passed": fault.get("passed"),
                "fault_total": fault.get("total_cases"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
