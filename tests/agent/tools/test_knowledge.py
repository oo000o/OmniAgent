from pathlib import Path

from nanobot.agent.tools.knowledge import (
    KnowledgeAddTool,
    KnowledgeSearchTool,
    KnowledgeToolsConfig,
)


async def test_add_then_search_returns_cited_evidence(tmp_path: Path) -> None:
    source = tmp_path / "memory.md"
    source.write_text("Long-term memory persists facts across sessions.", encoding="utf-8")
    config = KnowledgeToolsConfig(database_path="state/knowledge.db")
    add = KnowledgeAddTool(workspace=tmp_path, config=config, restrict_to_workspace=True)
    search = KnowledgeSearchTool(workspace=tmp_path, config=config, restrict_to_workspace=True)

    add_result = await add.execute("memory.md")
    search_result = await search.execute("memory")

    assert "Indexed 'memory.md'" in add_result
    assert "[K1]" in search_result
    assert "persists facts" in search_result


def test_database_path_cannot_escape_workspace() -> None:
    try:
        KnowledgeToolsConfig(database_path="../outside.db")
    except ValueError as exc:
        assert "inside the workspace" in str(exc)
    else:
        raise AssertionError("escaping database path should be rejected")
