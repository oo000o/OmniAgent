from evaluation.run import run


def test_offline_evaluation_has_at_least_fifty_passing_cases(tmp_path) -> None:
    output = tmp_path / "report.json"
    report = run(output)

    assert report["total"] == 75
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
    assert output.is_file()
