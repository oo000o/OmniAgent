from evaluation.run import run


def test_offline_evaluation_has_at_least_fifty_passing_cases(tmp_path) -> None:
    output = tmp_path / "report.json"
    report = run(output)

    assert report["total"] >= 50
    assert report["passed"] == report["total"]
    assert output.is_file()
