# OmniAgent Cross-channel Workflow

OmniAgent Version 1 uses the WebUI and Feishu. Feishu is the external channel because it is
common in domestic team workflows, supports a WebSocket long connection without a public
IP, and can be configured with an individual developer app. The WebUI remains the convenient
place to inspect long answers and execution details.

## Shared-state design

Conversation state and business state are intentionally separate:

- `agents.defaults.unifiedSession=true` maps WebUI and Feishu turns from this single-user
  deployment onto the durable `unified:default` conversation.
- Tasks are stored in the task MCP server's SQLite database, so they remain available even
  when unified conversation mode is later disabled.
- Knowledge chunks live in a separate SQLite index. Both channels call the same
  `knowledge_search` tool and receive the same citation identifiers.
- A schedule created from Feishu records the concrete Feishu chat as its delivery
  route. A schedule created from WebUI records that WebUI chat. Scheduled turns reuse the
  owning session but return to the channel that created the schedule.

Unified sessions are appropriate only for this private, single-user product. A multi-user
deployment must keep per-user session keys and add authorization to task ownership.

## Reference demo

1. In WebUI, index `study-plan.pdf` with `knowledge_add`.
2. In Feishu, ask: "根据学习计划给我创建本周三个任务，每天 20:00 检查进度。"
3. The Agent retrieves cited evidence, creates three MCP tasks with stable idempotency keys,
   then creates a session-bound Cron job in `Asia/Shanghai`.
4. In WebUI, ask to list current tasks. The same MCP database returns the Feishu-created
   tasks and the shared conversation supplies the earlier context.
5. At 20:00, Cron submits a normal Agent turn using the stored session and sends the result
   back to the originating Feishu chat.

## Delivery behavior

The existing channel manager is reused as the transport boundary. It performs bounded
retries with exponential delays and lets a channel classify non-retryable business errors.
OmniAgent config fixes `sendMaxRetries` at three attempts for the demo. Duplicate final
responses tied to the same source message are suppressed before delivery.

Retries are currently process-local. Durable delivery outbox semantics are intentionally
deferred and documented as a production-hardening item rather than claimed as complete.

## Local setup

1. Copy `examples/omniagent.config.example.json` into the normal nanobot configuration and
   replace `<PROJECT_ROOT>` with an absolute path.
2. Export `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, and `FEISHU_OPEN_ID`; do not commit them.
3. Create a Feishu developer app with Bot capability, enable `im:message`,
   `im:message.p2p_msg:readonly`, and the `im.message.receive_v1` event, then select Long
   Connection mode. Add `cardkit:card:write` for streaming cards, or disable streaming.
4. Install the Feishu channel dependency, publish the app, start the gateway, and execute
   the reference demo in a private Feishu chat.

The automated acceptance suite uses a fake external channel, a temporary Cron store, and a
real stdio MCP subprocess. This validates routing and retries without requiring a developer
credentials in CI. A live Feishu smoke test remains a release checklist item.
