"""Aggregate deterministic fault-injection outcomes into a compact report."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.fault_injection import evaluate_fault_injection


async def evaluate_fault_injection_report(root: Path) -> dict[str, object]:
    """Wrap existing fault cases with resume-friendly aggregates."""

    raw = await evaluate_fault_injection(root)
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError("fault injection cases are malformed")

    duplicate_side_effect_count = 0
    version_conflict_detected = 0
    recovery_focus: list[dict[str, object]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id", ""))
        detail = str(case.get("detail", ""))
        passed = bool(case.get("passed"))
        if case_id in {
            "fault-task-replay-thrice",
            "fault-cron-replay-thrice",
            "fault-replan-interrupt",
            "fault-stale-confirm",
        }:
            recovery_focus.append(
                {"case_id": case_id, "passed": passed, "detail": detail}
            )
        if "VERSION_CONFLICT" in detail and passed:
            version_conflict_detected += 1
        if case_id.endswith("replay-thrice") and not passed:
            duplicate_side_effect_count += 1

    recovery_total = len(recovery_focus) or int(raw["total"])
    recovery_passed = (
        sum(1 for item in recovery_focus if item["passed"])
        if recovery_focus
        else int(raw["passed"])
    )

    return {
        "fixture": "omniagent-fault-injection-report-v1",
        "scope": "deterministic fault-injection test; not production success rate",
        "total_cases": int(raw["total"]),
        "passed": int(raw["passed"]),
        "pass_rate": raw.get("pass_rate"),
        "duplicate_side_effect_count": duplicate_side_effect_count,
        "recovery_completion_rate": round(recovery_passed / recovery_total, 4)
        if recovery_total
        else 0.0,
        "version_conflict_detected": version_conflict_detected,
        "focus_cases": recovery_focus,
        "raw_groups": raw.get("groups"),
        "cases": cases,
    }


def write_fault_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
