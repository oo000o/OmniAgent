from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict[str, Any]:
    with (ROOT / "examples" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_docker_career_and_mcp_share_authoritative_task_database() -> None:
    config = _load("omniagent.docker.config.json")
    tools = config["tools"]

    assert tools["career"]["taskDatabasePath"] == ".nanobot/tasks.db"
    assert (
        tools["mcpServers"]["omniagent_tasks"]["env"]["OMNIAGENT_TASK_DB"]
        == "/data/workspace/.nanobot/tasks.db"
    )


def test_career_examples_allow_a_complete_tool_driven_workflow() -> None:
    for name in ("omniagent.config.example.json", "omniagent.docker.config.json"):
        config = _load(name)
        assert config["agents"]["defaults"]["maxToolIterations"] >= 40
