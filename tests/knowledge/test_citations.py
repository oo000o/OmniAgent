from pathlib import Path

from nanobot.knowledge.citations import render_retrieval_context
from nanobot.knowledge.models import KnowledgeChunk, KnowledgeSearchResult


def test_context_contains_stable_citations_and_source_metadata() -> None:
    result = KnowledgeSearchResult(
        chunk=KnowledgeChunk("chunk", "doc", "Memory persists facts.", 10, 32, "Memory"),
        source_path=Path("guide.md"),
        source_name="guide.md",
        score=1.0,
        rank=1,
    )

    rendered = render_retrieval_context([result])

    assert "[K1]" in rendered
    assert "guide.md" in rendered
    assert "heading='Memory'" in rendered
    assert "chars=10-32" in rendered
    assert "untrusted evidence" in rendered
