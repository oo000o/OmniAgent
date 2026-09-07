from evaluation.run import run


def test_offline_evaluation_has_at_least_fifty_passing_cases(tmp_path) -> None:
    output = tmp_path / "report.json"
    report = run(output)

    assert report["total"] == 92
    assert report["passed"] == report["total"]
    benchmark = report["retrieval_benchmark"]
    assert isinstance(benchmark, dict)
    assert benchmark["case_count"] == 12
    assert "hybrid_rrf" in benchmark["metrics"]
    career = report["career_workflow"]
    assert isinstance(career, dict)
    assert career["passed"] == career["total"]
    assert set(career["groups"]) == {
        "career_guardrail",
        "career_recovery",
        "career_success",
    }
    observability = report["observability"]
    assert isinstance(observability, dict)
    assert observability["passed"] == observability["total"]
    assert "observability" in report["groups"]
    fault = report["fault_injection"]
    assert isinstance(fault, dict)
    assert fault["passed"] == fault["total"]
    assert "fault_injection" in report["groups"]
    assert output.is_file()
