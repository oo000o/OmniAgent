"""Run the deterministic OmniAgent evaluation baseline and save its evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from nanobot.knowledge.citations import render_retrieval_context
from nanobot.knowledge.fusion import reciprocal_rank_fusion
from nanobot.knowledge.ingest import ingest_document
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult
from nanobot.knowledge.store import KnowledgeStore
from nanobot.tasking import TaskCreate, TaskStore

CORPUS = {
    "retrieval.md": "BM25 lexical retrieval finds exact technical terms in private documents.",
    "vectors.md": "Vector embeddings retrieve semantically related passages by cosine similarity.",
    "fusion.md": "Reciprocal rank fusion combines lexical and semantic ranked result lists.",
    "citations.md": "Knowledge answers include stable source citations and character offsets.",
    "mcp.md": "The task service uses the Model Context Protocol over a stdio transport.",
    "idempotency.md": "Idempotency keys prevent duplicate task creation during model retries.",
    "locking.md": "Optimistic version locking prevents concurrent task updates from overwriting data.",
    "scheduling.md": "Cron schedules wake the agent and return reminders to the originating channel.",
    "tracing.md": "Run traces record model latency token usage tool calls retries and errors.",
    "security.md": "Document text is untrusted evidence and must never become executable instructions.",
}

CITATION_QUERIES = {
    "retrieval.md": "BM25",
    "vectors.md": "embeddings",
    "fusion.md": "Reciprocal",
    "citations.md": "offsets",
    "mcp.md": "Protocol",
    "idempotency.md": "Idempotency",
    "locking.md": "locking",
    "scheduling.md": "Cron",
    "tracing.md": "latency",
    "security.md": "untrusted",
}

RETRIEVAL_CASES = [
    ("exact term search", "BM25 lexical", "retrieval.md"),
    ("private document search", "technical terms", "retrieval.md"),
    ("keyword retrieval", "lexical retrieval", "retrieval.md"),
    ("semantic representation", "Vector embeddings", "vectors.md"),
    ("cosine search", "cosine similarity", "vectors.md"),
    ("related passages", "semantically related", "vectors.md"),
    ("merge ranked lists", "rank fusion", "fusion.md"),
    ("RRF", "Reciprocal rank fusion", "fusion.md"),
    ("lexical semantic combination", "lexical semantic", "fusion.md"),
    ("source location", "character offsets", "citations.md"),
    ("stable evidence references", "stable source citations", "citations.md"),
    ("answer provenance", "source citations", "citations.md"),
    ("stdio protocol", "stdio transport", "mcp.md"),
    ("task MCP", "Model Context Protocol", "mcp.md"),
    ("tool server transport", "task service", "mcp.md"),
    ("duplicate task prevention", "Idempotency keys", "idempotency.md"),
    ("model retry duplicate", "model retries", "idempotency.md"),
    ("safe repeated create", "duplicate task creation", "idempotency.md"),
    ("concurrent update", "concurrent task updates", "locking.md"),
    ("version conflict", "Optimistic version locking", "locking.md"),
    ("prevent overwrite", "overwriting data", "locking.md"),
    ("scheduled reminder", "Cron schedules", "scheduling.md"),
    ("origin channel delivery", "originating channel", "scheduling.md"),
    ("wake agent", "wake the agent", "scheduling.md"),
    ("token observability", "token usage", "tracing.md"),
    ("retry trace", "retries and errors", "tracing.md"),
    ("model latency", "model latency", "tracing.md"),
    ("prompt injection boundary", "untrusted evidence", "security.md"),
    ("document instructions", "executable instructions", "security.md"),
    ("evidence isolation", "Document text", "security.md"),
]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    group: str
    passed: bool
    detail: str


def _retrieval_and_citation_cases(root: Path) -> list[CaseResult]:
    store = KnowledgeStore(root / "knowledge.db")
    store.initialize()
    for name, content in CORPUS.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        ingest_document(path, store)
    results: list[CaseResult] = []
    for index, (_label, query, expected) in enumerate(RETRIEVAL_CASES, 1):
        hits = store.search_lexical(query, limit=5)
        sources = [hit.source_name for hit in hits]
        results.append(CaseResult(f"retrieval-{index:02d}", "retrieval", expected in sources, str(sources)))
    for index, (name, term) in enumerate(CITATION_QUERIES.items(), 1):
        hits = store.search_lexical(term, limit=1)
        rendered = render_retrieval_context(hits)
        passed = bool(hits) and "[K1]" in rendered and name in rendered and "chars=" in rendered
        results.append(CaseResult(f"citation-{index:02d}", "citation", passed, name))
    return results


def _task_cases(root: Path) -> list[CaseResult]:
    store = TaskStore(root / "tasks.db")
    store.initialize()
    results: list[CaseResult] = []
    for index in range(1, 11):
        request = TaskCreate(title=f"Evaluation task {index}", tags=["eval"])
        first = store.create(request, idempotency_key=f"eval-create-{index}")
        replay = store.create(request, idempotency_key=f"eval-create-{index}")
        results.append(CaseResult(f"idempotency-{index:02d}", "idempotency", first.task_id == replay.task_id, first.task_id))
    invalid_payloads = [
        {"title": ""},
        {"title": "bad priority", "priority": 0},
        {"title": "bad priority", "priority": 6},
        {"title": "bad due", "due_at": "2026-09-03T20:00:00"},
        {"title": "x" * 241},
    ]
    for index, payload in enumerate(invalid_payloads, 1):
        try:
            TaskCreate.model_validate(payload)
            passed = False
        except ValidationError:
            passed = True
        results.append(CaseResult(f"validation-{index:02d}", "validation", passed, str(payload)[:120]))
    return results


def _fusion_cases() -> list[CaseResult]:
    results: list[CaseResult] = []
    for index in range(1, 6):
        def result(chunk_id: str, source_name: str) -> KnowledgeSearchResult:
            text = "shared result"
            return KnowledgeSearchResult(
                chunk=KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id="doc",
                    text=text,
                    start_char=0,
                    end_char=len(text),
                ),
                source_path=Path(source_name),
                source_name=source_name,
                score=1.0,
                rank=1,
            )

        common = result(f"common-{index}", "common.md")
        lexical_only = result(f"lex-{index}", "lex.md")
        semantic_only = result(f"sem-{index}", "sem.md")
        fused = reciprocal_rank_fusion([[common, lexical_only], [semantic_only, common]], limit=3)
        winner = fused[0].result.chunk.chunk_id
        passed = winner == common.chunk.chunk_id
        results.append(CaseResult(f"fusion-{index:02d}", "fusion", passed, winner))
    return results


def run(output: Path) -> dict[str, object]:
    # sqlite3's context manager commits but does not immediately close handles on
    # Windows, so cleanup is best-effort after this short-lived evaluation run.
    with tempfile.TemporaryDirectory(
        prefix="omniagent-eval-", ignore_cleanup_errors=True
    ) as directory:
        root = Path(directory)
        cases = [*_retrieval_and_citation_cases(root), *_task_cases(root), *_fusion_cases()]
    passed = sum(case.passed for case in cases)
    groups: dict[str, dict[str, int | float]] = {}
    for group in sorted({case.group for case in cases}):
        selected = [case for case in cases if case.group == group]
        group_passed = sum(case.passed for case in selected)
        groups[group] = {
            "passed": group_passed,
            "total": len(selected),
            "pass_rate": round(group_passed / len(selected), 4),
        }
    report: dict[str, object] = {
        "suite": "omniagent-offline-baseline-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4),
        "groups": groups,
        "cases": [asdict(case) for case in cases],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/latest.json"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({key: report[key] for key in ("suite", "total", "passed", "pass_rate", "groups")}, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
