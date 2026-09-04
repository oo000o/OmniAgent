import json

from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.bus.runtime_events import RuntimeEventContext, SessionTurnStarted
from nanobot.observability import RunStore
from nanobot.webui.ws_http import GatewayHTTPHandler


async def test_run_list_and_detail_routes(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    store.handle(
        SessionTurnStarted(
            RuntimeEventContext(
                channel="telegram",
                chat_id="tg-9",
                session_key="unified:default",
            )
        )
    )
    run_id = store.list()[0].run_id
    handler = GatewayHTTPHandler.__new__(GatewayHTTPHandler)
    handler.omniagent_run_store = store
    handler.check_api_token = lambda _request: True

    list_request = Request(
        "/api/omniagent/runs?session_key=unified%3Adefault&limit=10", Headers()
    )
    listed = await handler._dispatch_misc_routes(None, list_request, "/api/omniagent/runs")
    assert listed is not None and listed.status_code == 200
    payload = json.loads(listed.body)
    assert payload["runs"][0]["channel"] == "telegram"

    detail_request = Request(f"/api/omniagent/runs/{run_id}", Headers())
    detail = await handler._dispatch_misc_routes(
        None, detail_request, f"/api/omniagent/runs/{run_id}"
    )
    assert detail is not None and detail.status_code == 200
    assert json.loads(detail.body)["run"]["run_id"] == run_id


async def test_run_routes_validate_limit_and_missing_id(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.initialize()
    handler = GatewayHTTPHandler.__new__(GatewayHTTPHandler)
    handler.omniagent_run_store = store
    handler.check_api_token = lambda _request: True

    request = Request("/api/omniagent/runs?limit=999", Headers())
    response = await handler._dispatch_misc_routes(None, request, "/api/omniagent/runs")
    assert response is not None and response.status_code == 400

    missing_id = "0" * 32
    request = Request(f"/api/omniagent/runs/{missing_id}", Headers())
    response = await handler._dispatch_misc_routes(
        None, request, f"/api/omniagent/runs/{missing_id}"
    )
    assert response is not None and response.status_code == 404
