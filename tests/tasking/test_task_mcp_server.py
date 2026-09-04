import json
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from nanobot.agent.tools.mcp import connect_mcp_servers
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig


async def test_stdio_mcp_exposes_persistent_idempotent_task_tools(tmp_path) -> None:
    environment = os.environ.copy()
    environment["OMNIAGENT_TASK_DB"] = str(tmp_path / "mcp-tasks.db")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "nanobot.tasking.mcp_server"],
        env=environment,
    )

    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "task_create",
                "task_cancel",
                "task_get",
                "task_list",
                "task_update",
            }

            arguments = {
                "title": "Prepare Agent demo",
                "idempotency_key": "demo-turn-create-1",
                "tags": ["agent"],
            }
            first = await session.call_tool("task_create", arguments)
            second = await session.call_tool("task_create", arguments)
            first_task = json.loads(first.content[0].text)
            second_task = json.loads(second.content[0].text)
            assert first_task["task_id"] == second_task["task_id"]

            update_arguments = {
                "task_id": first_task["task_id"],
                "expected_version": 1,
                "idempotency_key": "demo-turn-update-1",
                "status": "in_progress",
            }
            updated = await session.call_tool("task_update", update_arguments)
            replayed = await session.call_tool("task_update", update_arguments)
            updated_task = json.loads(updated.content[0].text)
            replayed_task = json.loads(replayed.content[0].text)
            assert updated_task == replayed_task
            assert updated_task["version"] == 2

            listed = await session.call_tool("task_list", {"status": "in_progress"})
            payload = json.loads(listed.content[0].text)
            if isinstance(payload, dict):
                tasks = payload.get("result", [payload])
            else:
                tasks = payload
            assert [task["task_id"] for task in tasks] == [first_task["task_id"]]

            cancelled = await session.call_tool(
                "task_cancel",
                {
                    "task_id": first_task["task_id"],
                    "expected_version": 2,
                    "idempotency_key": "demo-turn-cancel-1",
                    "confirm": True,
                },
            )
            assert json.loads(cancelled.content[0].text)["status"] == "cancelled"

            refused = await session.call_tool(
                "task_cancel",
                {
                    "task_id": first_task["task_id"],
                    "expected_version": 3,
                    "idempotency_key": "demo-turn-cancel-2",
                },
            )
            assert refused.isError is True
            assert "confirm=true" in refused.content[0].text


async def test_nanobot_runtime_registers_namespaced_task_tools(tmp_path) -> None:
    registry = ToolRegistry()
    connections = await connect_mcp_servers(
        {
            "omniagent_tasks": MCPServerConfig(
                type="stdio",
                command=sys.executable,
                args=["-m", "nanobot.tasking.mcp_server"],
                cwd=os.getcwd(),
                env={"OMNIAGENT_TASK_DB": str(tmp_path / "runtime-tasks.db")},
            )
        },
        registry,
    )
    try:
        assert {
            "mcp_omniagent_tasks_task_create",
            "mcp_omniagent_tasks_task_get",
            "mcp_omniagent_tasks_task_list",
            "mcp_omniagent_tasks_task_update",
            "mcp_omniagent_tasks_task_cancel",
        } <= set(registry.tool_names)
        result = await registry.execute(
            "mcp_omniagent_tasks_task_create",
            {"title": "Runtime task", "idempotency_key": "runtime-create-1"},
        )
        assert "Runtime task" in str(result)
    finally:
        for connection in connections.values():
            await connection.aclose()
