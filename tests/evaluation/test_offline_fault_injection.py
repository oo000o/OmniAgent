from evaluation.fault_injection import evaluate_fault_injection


async def test_fault_injection_evaluation_covers_recovery_and_privacy(tmp_path) -> None:
    report = await evaluate_fault_injection(tmp_path)

    assert report["passed"] == report["total"]
    assert report["total"] == 7
    assert report["scope"].startswith("offline crash/replay")
    assert set(report["groups"]) == {"fault_injection"}
