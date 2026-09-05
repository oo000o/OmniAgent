from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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


def test_docker_compose_publishes_local_web_and_health_ports() -> None:
    with (ROOT / "docker-compose.omniagent.yml").open(encoding="utf-8") as handle:
        compose = yaml.safe_load(handle)

    ports = compose["services"]["nanobot-gateway"]["ports"]
    assert "127.0.0.1:8765:8765" in ports
    assert "127.0.0.1:18790:18790" in ports
    assert (
        compose["services"]["nanobot-gateway"]["environment"][
            "NANOBOT_CONFIG_TEMPLATE_MODE"
        ]
        == "overwrite"
    )
