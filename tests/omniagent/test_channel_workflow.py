from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.cron import CronTool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import ChannelsConfig
from nanobot.cron.bound_runner import run_bound_cron_job
from nanobot.cron.service import CronService
from nanobot.session.keys import UNIFIED_SESSION_KEY, session_key_for_channel


class _FeishuProbeChannel(BaseChannel):
    name = "feishu"
    display_name = "Feishu Probe"

    def __init__(self, config, bus, *, failures: int = 0) -> None:
        super().__init__(config, bus)
        self.failures = failures
        self.attempts: list[OutboundMessage] = []

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return

    async def send(self, msg: OutboundMessage) -> None:
        self.attempts.append(msg)
        if len(self.attempts) <= self.failures:
            raise ConnectionError("temporary Feishu outage")


async def test_shared_session_schedule_returns_to_origin_with_bounded_retry(tmp_path) -> None:
    assert session_key_for_channel("websocket", "web-1", unified_session=True) == (
        UNIFIED_SESSION_KEY
    )
    assert session_key_for_channel("feishu", "oc_demo", unified_session=True) == (
        UNIFIED_SESSION_KEY
    )

    cron = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(cron, default_timezone="Asia/Shanghai")
    with request_context(
        RequestContext(
            channel="feishu",
            chat_id="oc_demo",
            session_key=UNIFIED_SESSION_KEY,
            metadata={"message_id": "om_demo_message"},
        )
    ):
        result = await tool.execute(
            action="add",
            name="daily-progress",
            message="Check incomplete MCP tasks and summarize progress.",
            cron_expr="0 20 * * *",
            tz="Asia/Shanghai",
        )
    assert result.startswith("Created job")
    job = cron.list_jobs()[0]
    assert job.payload.origin_channel == "feishu"
    assert job.payload.origin_chat_id == "oc_demo"

    class _Agent:
        tools = SimpleNamespace(get=lambda _name: None)

        async def submit_cron_turn(self, msg):
            assert msg.session_key == "feishu:oc_demo"
            assert msg.channel == "feishu"
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Two tasks remain.",
                metadata=msg.metadata,
            )

    response = await run_bound_cron_job(job, agent=_Agent(), cron=cron)
    assert response == "Two tasks remain."

    config = SimpleNamespace(
        channels=ChannelsConfig(send_max_retries=3),
        providers=SimpleNamespace(groq=SimpleNamespace(api_key="")),
    )
    manager = ChannelManager.__new__(ChannelManager)
    manager.config = config
    manager.bus = MessageBus()
    channel = _FeishuProbeChannel(config, manager.bus, failures=2)
    with patch("nanobot.channels.manager.asyncio.sleep", new_callable=AsyncMock) as sleep:
        await manager._send_with_retry(
            channel,
            OutboundMessage(
                channel="feishu",
                chat_id="oc_demo",
                content=response,
            ),
        )

    assert len(channel.attempts) == 3
    assert sleep.await_count == 2
