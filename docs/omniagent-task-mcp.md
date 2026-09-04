# OmniAgent Task MCP

The task server is an independent stdio MCP process backed by SQLite. It deliberately
keeps task state outside the model conversation so WebUI and chat-channel requests can
operate on the same durable records.

## Start the server

Install the project in editable mode, then run:

```bash
omniagent-task-mcp
```

Set `OMNIAGENT_TASK_DB` to choose the database file. The default is
`.nanobot/tasks.db` relative to the server working directory.

## Connect nanobot

Merge the following entry into `tools.mcpServers` in the nanobot configuration. Replace
`<PROJECT_ROOT>` with the absolute path to this repository's `source` directory.

```json
{
  "tools": {
    "mcpServers": {
      "omniagent_tasks": {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "nanobot.tasking.mcp_server"],
        "cwd": "<PROJECT_ROOT>",
        "env": {
          "OMNIAGENT_TASK_DB": ".nanobot/tasks.db"
        },
        "enabledTools": [
          "task_create",
          "task_get",
          "task_list",
          "task_update",
          "task_cancel"
        ]
      }
    }
  }
}
```

The runtime exposes these as namespaced tools such as
`mcp_omniagent_tasks_task_create`, preventing collisions with built-in tools.

## Consistency rules

- Every create, update, and cancel call carries an idempotency key. Retries with the same
  key and payload return the previous result instead of applying the mutation twice.
- Reusing a key with a different payload fails explicitly.
- Updates and cancellations require the last observed `version`. A stale version fails
  instead of silently overwriting a concurrent change.
- Cancellation is the supported delete semantic. It retains an audit trail and can be
  filtered through `task_list(status="cancelled")`. The call is rejected unless the Agent
  supplies `confirm=true`, so an ambiguous request cannot silently remove an active task.
- `due_at` must be an RFC 3339 timestamp with a timezone and is normalized to UTC.
