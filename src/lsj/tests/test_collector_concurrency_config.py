"""Validate all concurrency settings before allocating any model client."""
import asyncio
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"


@pytest.fixture(params=["classifier", "sentiment"])
def context(request, monkeypatch):
    name = "_concurrency_config_" + request.param
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{request.param}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    created, generated, closed = [], [], []

    class Client:
        def __init__(self, cfg):
            self.name = cfg.name
            created.append(cfg.name)

        async def generate(self, prompt, temperature, max_tokens):
            generated.append((self.name, prompt, temperature, max_tokens))
            return "synthetic response " + self.name

        async def close(self):
            closed.append(self.name)

    for client in ("OpenAIClient", "AnthropicClient", "LocalOpenAICompatibleClient"):
        monkeypatch.setattr(module, client, Client)
    return SimpleNamespace(module=module, created=created, generated=generated, closed=closed)


@pytest.mark.parametrize("invalid", [0, -1, 1.0, 1.5, True, False])
@pytest.mark.parametrize("after_valid", [False, True])
def test_invalid_concurrency_is_rejected_before_any_client_exists(context, invalid, after_valid):
    module = context.module
    configs = ([module.ModelConfig(name="valid-first", provider="openai", concurrency=3)]
               if after_valid else [])
    configs.append(module.ModelConfig(name="invalid", provider="local", concurrency=invalid))
    with pytest.raises(ValueError, match="^Model concurrency must be a positive integer\\.$"):
        module.ModelPool(configs, logging.getLogger("synthetic-concurrency"))
    assert context.created == [] and context.generated == [] and context.closed == []


def test_positive_concurrency_keeps_all_providers_generation_and_close(context, monkeypatch):
    module = context.module
    configs = [module.ModelConfig(name=provider, provider=provider, concurrency=value)
               for provider, value in (("openai", 1), ("anthropic", 3), ("local", 10**9))]
    pool = module.ModelPool(configs, logging.getLogger("synthetic-concurrency"))
    assert context.created == [cfg.name for cfg in configs]

    async def run():
        try:
            for cfg in configs:
                monkeypatch.setattr(pool, "_pick_model", lambda cfg=cfg: cfg)
                result = await asyncio.wait_for(pool.generate("synthetic prompt", 0.5, 10), timeout=1)
                assert result == ("synthetic response " + cfg.name, cfg.name)
        finally:
            await pool.close()
        assert [entry[0] for entry in context.generated] == [cfg.name for cfg in configs]
        assert context.closed == [cfg.name for cfg in configs]

    asyncio.run(run())
