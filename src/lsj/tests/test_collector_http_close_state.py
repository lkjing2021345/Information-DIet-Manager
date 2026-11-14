"""Real HTTPX close state with synthetic transports and no network connections.

Only test-owned temporary file handles model incomplete transport cleanup. Their
explicit fixture cleanup is not a promise of production HTTPX failure recovery.
"""
import asyncio
import traceback
from types import SimpleNamespace

import httpx
import pytest

from test_collector_http_lifecycle import CLOSE_ERROR, context  # noqa: F401


REAL_HTTP_CLIENT = httpx.AsyncClient
PRIVATE = "SYNTHETIC_PRIVATE_TRANSPORT_ERROR"


@pytest.fixture
def transport_context(context, tmp_path, monkeypatch):
    created = []

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.resource = (tmp_path / "synthetic-httpx-resource.bin").open("wb")
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.close_calls = 0
            self.error = None

        async def handle_async_request(self, request):
            return httpx.Response(200, request=request,
                                  json={"choices": [{"message": {"content": "synthetic response"}}]})

        async def aclose(self):
            self.close_calls += 1
            self.started.set()
            await self.release.wait()
            if self.error is not None:
                raise self.error
            self.resource.close()

    def create_http(**kwargs):
        transport = Transport()
        actual = REAL_HTTP_CLIENT(**kwargs, transport=transport, trust_env=False)
        created.append(SimpleNamespace(transport=transport, actual=actual))
        return actual

    monkeypatch.setattr(context.module.httpx, "AsyncClient", create_http)
    yield SimpleNamespace(context=context, created=created)
    for item in created:
        item.transport.resource.close()


async def initialized_client(fixture):
    client = fixture.context.client()
    assert await client.generate("synthetic", 0.4, 10) == "synthetic response"
    assert len(fixture.created) == 1
    return client, fixture.created[0].actual, fixture.created[0].transport


async def cancel_unfinished(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", ["exception", "cancelled"])
@pytest.mark.parametrize("concurrent", [False, True])
def test_failed_real_httpx_close_never_becomes_success_or_drops_reference(transport_context, failure, concurrent):
    async def run():
        client, actual, transport = await initialized_client(transport_context)
        primary = OSError(PRIVATE)
        if failure == "exception":
            transport.error = primary
        owner = asyncio.create_task(client.close())
        tasks = [owner]
        try:
            await asyncio.wait_for(transport.started.wait(), timeout=2)
            assert actual.is_closed and client.http is actual
            if concurrent:
                entered = asyncio.Event()

                async def close_again():
                    entered.set()
                    await client.close()

                waiter = asyncio.create_task(close_again())
                tasks.append(waiter)
                await asyncio.wait_for(entered.wait(), timeout=2)
                await asyncio.sleep(0)
                assert not waiter.done(), "Second close must wait for the in-progress owner."
                assert client.http is actual and transport.close_calls == 1
            if failure == "exception":
                transport.release.set()
                with pytest.raises(OSError) as raised:
                    await owner
                assert raised.value is primary
            else:
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
            if concurrent:
                with pytest.raises(RuntimeError) as raised:
                    await waiter
                assert str(raised.value) == CLOSE_ERROR
            for _ in range(2):
                with pytest.raises(RuntimeError) as repeated:
                    await client.close()
                assert str(repeated.value) == CLOSE_ERROR
                assert PRIVATE not in "".join(traceback.format_exception(repeated.value))
                assert client.http is actual
            assert transport.close_calls == 1 and not transport.resource.closed
            assert actual.is_closed  # This public flag does not prove transport cleanup.
            with pytest.raises(RuntimeError):
                await client.generate("synthetic", 0.4, 10)
            assert len(transport_context.created) == 1
        finally:
            await cancel_unfinished(tasks)

    asyncio.run(run())


def test_concurrent_successful_close_waits_and_remains_idempotent(transport_context):
    async def run():
        client, actual, transport = await initialized_client(transport_context)
        owner = asyncio.create_task(client.close())
        tasks = [owner]
        try:
            await asyncio.wait_for(transport.started.wait(), timeout=2)
            waiter = asyncio.create_task(client.close())
            tasks.append(waiter)
            await asyncio.sleep(0)
            assert not waiter.done() and client.http is actual
            transport.release.set()
            assert await asyncio.gather(*tasks) == [None, None]
            await client.close()
            assert client.http is None and transport.resource.closed
            assert transport.close_calls == 1
        finally:
            await cancel_unfinished(tasks)

    asyncio.run(run())


def test_cancelling_close_waiter_does_not_cancel_or_poison_owner(transport_context):
    async def run():
        client, actual, transport = await initialized_client(transport_context)
        owner = asyncio.create_task(client.close())
        tasks = [owner]
        try:
            await asyncio.wait_for(transport.started.wait(), timeout=2)
            waiter = asyncio.create_task(client.close())
            tasks.append(waiter)
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not owner.done() and client.http is actual
            transport.release.set()
            await owner
            await client.close()
            assert client.http is None and transport.resource.closed
            assert transport.close_calls == 1
        finally:
            await cancel_unfinished(tasks)

    asyncio.run(run())
