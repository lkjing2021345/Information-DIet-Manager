"""HTTP ownership with real collector paths and synthetic transports only.

Transport allocation owns a temporary file handle so missed close calls are
observable without opening sockets, reading personal configuration, or models.
Failed or cancelled closes must keep an explicit failed state and reference;
even a retry-capable synthetic transport cannot establish public HTTPX recovery.
Actual HTTPX with synthetic transports is covered by test_collector_http_close_state.
"""
import asyncio
from copy import deepcopy
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"
CLOSE_ERROR = "Model client cleanup previously failed; resource state is unknown."


@pytest.fixture(params=["classifier", "sentiment"])
def context(request, monkeypatch, tmp_path):
    kind = request.param
    name = "_http_lifecycle_" + kind
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{kind}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    cls = getattr(module, kind.title() + "DataCollector")
    monkeypatch.setattr(cls, "_try_load_sbert", lambda self: None)
    monkeypatch.setattr(module, "setup_logger", lambda *_args: logging.getLogger(name))
    state = SimpleNamespace(created=[], attempts=[], posts=[], outcomes=[], post_hook=None,
                            construction_errors=[])

    class SyntheticHTTP:
        def __init__(self, **kwargs):
            state.attempts.append(kwargs)
            if state.construction_errors:
                error = state.construction_errors.pop(0)
                if error is not None:
                    raise error
            self.resource = (tmp_path / f"transport-{len(state.created)}.bin").open("wb")
            self.close_calls = 0
            self.close_outcomes = []
            self.close_hook = None
            state.created.append(self)

        async def post(self, url, **kwargs):
            if self.resource.closed:
                raise RuntimeError("Synthetic transport already closed.")
            state.posts.append((self, url, deepcopy(kwargs)))
            if state.post_hook is not None:
                await state.post_hook()
            outcome = state.outcomes.pop(0) if state.outcomes else 200
            if isinstance(outcome, BaseException):
                raise outcome
            data = ({"content": [{"text": "synthetic response"}]}
                    if "anthropic" in url else
                    {"choices": [{"message": {"content": "synthetic response"}}]})
            return module.httpx.Response(outcome, json=data,
                                         request=module.httpx.Request("POST", url))

        async def aclose(self):
            self.close_calls += 1
            if self.close_hook is not None:
                await self.close_hook()
            if self.close_outcomes:
                outcome = self.close_outcomes.pop(0)
                if outcome is not None:
                    raise outcome
            self.resource.close()

    monkeypatch.setattr(module.httpx, "AsyncClient", SyntheticHTTP)

    async def no_rate_wait(self):
        pass

    monkeypatch.setattr(module.BaseModelClient, "_rate_limit_wait", no_rate_wait)

    def cfg(provider="openai", name="synthetic-model", **kwargs):
        return module.ModelConfig(name=name, provider=provider, concurrency=2,
                                  api_key="synthetic-key", **kwargs)

    def client(provider="openai"):
        client_type = {"openai": module.OpenAIClient, "local": module.LocalOpenAICompatibleClient,
                       "anthropic": module.AnthropicClient}[provider]
        return client_type(cfg(provider))

    def collector():
        labels = ["News", "Tools"] if kind == "classifier" else ["平静", "开心"]
        runtime = module.RuntimeConfig(output=str(tmp_path / "synthetic.jsonl"), target_count=2,
                                       categories=labels, distribution=[0.5, 0.5],
                                       enable_contrast_pairs=False)
        return cls(runtime, [cfg()])

    yield SimpleNamespace(module=module, kind=kind, state=state, cfg=cfg, client=client,
                          collector=collector, logger=logging.getLogger(name))
    # Clean test-owned resources even when running a deliberately broken baseline.
    for transport in state.created:
        transport.resource.close()


@pytest.mark.parametrize("failure", ["provider", "wrapper"])
def test_later_pool_constructor_failure_allocates_no_http(context, monkeypatch, failure):
    module, state = context.module, context.state
    primary = ValueError("Synthetic wrapper initialization failure.")
    second_provider = "synthetic-unsupported"
    if failure == "wrapper":
        class FailingWrapper(module.LocalOpenAICompatibleClient):
            def __init__(self, cfg):
                super().__init__(cfg)
                raise primary
        monkeypatch.setattr(module, "LocalOpenAICompatibleClient", FailingWrapper)
        second_provider = "local"
    with pytest.raises(ValueError) as raised:
        module.ModelPool([context.cfg(name="first"), context.cfg(second_provider, "second")],
                         context.logger)
    if failure == "wrapper":
        assert raised.value is primary
    assert state.attempts == [] and state.created == []


@pytest.mark.parametrize("stage", ["DataStore", "ProgressTracker"])
def test_later_collector_constructor_failure_allocates_no_http(context, monkeypatch, stage):
    primary = OSError("Synthetic collector initialization failure.")

    def fail(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(context.module, stage, fail)
    with pytest.raises(OSError) as raised:
        context.collector()
    assert raised.value is primary
    assert context.state.attempts == [] and context.state.created == []


@pytest.mark.parametrize("provider", ["openai", "local", "anthropic"])
def test_first_use_allocates_once_and_preserves_request_and_response(context, provider):
    client = context.client(provider)
    assert context.state.attempts == []

    async def run():
        for _ in range(2):
            assert await client.generate("synthetic prompt", 0.4, 37) == "synthetic response"
        assert len(context.state.created) == 1
        transport = context.state.created[0]
        assert context.state.attempts == [{"timeout": client.cfg.timeout}]
        assert all(post[0] is transport for post in context.state.posts)
        for _, url, kwargs in context.state.posts:
            assert kwargs["json"]["model"] == client.cfg.name
            assert kwargs["json"]["max_tokens"] == 37
            assert kwargs["json"]["temperature"] == 0.4
            assert kwargs["json"]["messages"][-1] == {"role": "user", "content": "synthetic prompt"}
            if provider == "anthropic":
                assert url == "https://api.anthropic.com/v1/messages"
                assert kwargs["headers"]["x-api-key"] == "synthetic-key"
            else:
                assert url == "https://api.openai.com/v1/chat/completions"
                assert kwargs["headers"]["Authorization"] == "Bearer synthetic-key"
        await client.close()
        await client.close()
        assert transport.resource.closed and transport.close_calls == 1
        with pytest.raises(RuntimeError):
            await client.generate("synthetic", 0.5, 10)
        assert len(context.state.created) == 1 and len(context.state.posts) == 2

    asyncio.run(run())


def test_concurrent_first_requests_share_one_transport(context):
    client = context.client()
    assert context.state.created == []

    async def run():
        both_started, release = asyncio.Event(), asyncio.Event()

        async def wait_for_peer():
            if len(context.state.posts) == 2:
                both_started.set()
            await release.wait()

        context.state.post_hook = wait_for_peer
        tasks = [asyncio.create_task(client.generate("synthetic", 0.5, 10)) for _ in range(2)]
        try:
            await asyncio.wait_for(both_started.wait(), timeout=2)
            assert len(context.state.created) == 1
            assert context.state.posts[0][0] is context.state.posts[1][0]
            release.set()
            assert await asyncio.gather(*tasks) == ["synthetic response"] * 2
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
        assert context.state.created[0].resource.closed

    asyncio.run(run())


@pytest.mark.parametrize("context", ["classifier"], indirect=True)
def test_classifier_response_format_fallback_reuses_transport(context):
    client = context.client()
    context.state.outcomes = [400, 200]

    async def run():
        assert await client.generate("synthetic", 0.5, 10) == "synthetic response"
        assert len(context.state.created) == 1 and len(context.state.posts) == 2
        assert context.state.posts[0][2]["json"]["response_format"] == {"type": "json_object"}
        assert "response_format" not in context.state.posts[1][2]["json"]
        assert context.state.posts[0][0] is context.state.posts[1][0]
        await client.close()

    asyncio.run(run())


def test_pool_retry_reuses_transport_and_preserves_stats(context, monkeypatch):
    module = context.module
    context.state.outcomes = [module.httpx.NetworkError("Synthetic retry."), 200]

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    cfg = context.cfg(max_retries=2)
    pool = module.ModelPool([cfg], context.logger)

    async def run():
        assert await pool.generate("synthetic", 0.5, 10) == ("synthetic response", cfg.name)
        assert len(context.state.created) == 1 and len(context.state.posts) == 2
        stats = pool.stats[cfg.name]
        assert (stats.total_calls, stats.failed_calls, stats.success_calls) == (2, 1, 1)
        await pool.close()
        assert context.state.created[0].resource.closed

    asyncio.run(run())


def test_http_factory_error_is_retryable_after_pool_has_an_owner(context, monkeypatch):
    primary = OSError("Synthetic HTTP initialization failure before returning an object.")
    context.state.construction_errors = [primary, None]

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(context.module.asyncio, "sleep", no_sleep)
    cfg = context.cfg(max_retries=2)
    pool = context.module.ModelPool([cfg], context.logger)
    assert context.state.attempts == []

    async def run():
        assert await pool.generate("synthetic", 0.5, 10) == ("synthetic response", cfg.name)
        assert len(context.state.attempts) == 2 and len(context.state.created) == 1
        assert len(context.state.posts) == 1
        await pool.close()
        assert context.state.created[0].resource.closed

    asyncio.run(run())


def test_cancelled_collect_closes_transport_allocated_during_generate(context):
    collector = context.collector()
    assert context.state.created == []

    async def run():
        request_started, never = asyncio.Event(), asyncio.Event()

        async def block_request():
            request_started.set()
            await never.wait()

        async def dispatch():
            await collector.pool.generate("synthetic", 0.5, 10)

        context.state.post_hook = block_request
        collector._run_dispatch_loop = dispatch
        task = asyncio.create_task(collector.collect())
        try:
            await asyncio.wait_for(request_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.done() and collector.state.status == "interrupted"
            assert len(context.state.created) == 1
            assert context.state.created[0].resource.closed
            assert context.state.created[0].close_calls == 1
            assert not Path(collector.cfg.output).exists()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_close_before_use_is_safe_and_does_not_reopen(context):
    client = context.client()

    async def run():
        await client.close()
        await client.close()
        assert context.state.attempts == []
        with pytest.raises(RuntimeError):
            await client.generate("synthetic", 0.5, 10)
        assert context.state.attempts == [] and context.state.posts == []

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["exception", "cancelled"])
def test_failed_close_remains_failed_and_blocks_generate(context, failure):
    client = context.client()
    primary = (OSError("Synthetic close failure.") if failure == "exception"
               else asyncio.CancelledError("Synthetic close cancellation."))

    async def run():
        await client.generate("synthetic", 0.5, 10)
        transport = context.state.created[0]
        transport.close_outcomes = [primary, None]
        with pytest.raises(type(primary)) as raised:
            await client.close()
        assert raised.value is primary and not transport.resource.closed
        posts_before = len(context.state.posts)
        with pytest.raises(RuntimeError):
            await client.generate("synthetic", 0.5, 10)
        assert len(context.state.posts) == posts_before
        assert len(context.state.created) == 1
        for _ in range(2):
            with pytest.raises(RuntimeError) as repeated:
                await client.close()
            assert str(repeated.value) == CLOSE_ERROR
            assert repeated.value.__cause__ is None
            assert client.http is transport
        assert not transport.resource.closed and transport.close_calls == 1

    asyncio.run(run())


def test_pool_close_attempts_other_transports_and_preserves_first_error(context, monkeypatch):
    module = context.module
    configs = [context.cfg(name="first"), context.cfg("local", "second")]
    pool = module.ModelPool(configs, context.logger)
    primary, secondary = OSError("Synthetic first close."), ValueError("Synthetic second close.")

    async def run():
        for cfg in configs:
            monkeypatch.setattr(pool, "_pick_model", lambda cfg=cfg: cfg)
            await pool.generate("synthetic", 0.5, 10)
        first, second = context.state.created
        first.close_outcomes = [primary]
        second.close_outcomes = [secondary]
        with pytest.raises(OSError) as raised:
            await pool.close()
        assert raised.value is primary
        assert [item.close_calls for item in context.state.created] == [1, 1]
        assert not first.resource.closed and not second.resource.closed
        for _ in range(2):
            with pytest.raises(RuntimeError) as repeated:
                await pool.close()
            assert str(repeated.value) == CLOSE_ERROR
        assert [item.close_calls for item in context.state.created] == [1, 1]
        assert not first.resource.closed and not second.resource.closed
        assert pool.clients[configs[0].name].http is first
        assert pool.clients[configs[1].name].http is second

    asyncio.run(run())


def test_actual_task_cancellation_during_close_keeps_failed_state(context):
    client = context.client()

    async def run():
        await client.generate("synthetic", 0.5, 10)
        transport = context.state.created[0]
        close_started, never = asyncio.Event(), asyncio.Event()

        async def block_close():
            close_started.set()
            await never.wait()

        transport.close_hook = block_close
        task = asyncio.create_task(client.close())
        try:
            await asyncio.wait_for(close_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not transport.resource.closed
            with pytest.raises(RuntimeError):
                await client.generate("synthetic", 0.5, 10)
            assert len(context.state.created) == 1 and len(context.state.posts) == 1
            transport.close_hook = None
            for _ in range(2):
                with pytest.raises(RuntimeError) as repeated:
                    await client.close()
                assert str(repeated.value) == CLOSE_ERROR
            assert client.http is transport
            assert not transport.resource.closed and transport.close_calls == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
