from pathlib import Path

from evaluation.career_workflow import evaluate_career_workflow


async def test_career_workflow_evaluation_is_reproducible(tmp_path: Path) -> None:
    report = await evaluate_career_workflow(tmp_path)

    assert report["total"] >= 12
    assert report["passed"] == report["total"]
    assert report["scope"] == "offline reliability scenarios; not production traffic"
