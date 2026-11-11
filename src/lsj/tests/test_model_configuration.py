"""Real configuration/import/cache paths with stubbed optional ML dependencies.

No model is downloaded, trained or inferred. All filesystem writes are confined
to pytest temporary directories; the production logger is stubbed on import.
"""

import importlib.util
import logging
import os
from pathlib import Path
import sys
import types

import pytest


ALGORITHMS = Path(__file__).resolve().parents[1] / "src" / "algorithms"
CACHE_KEYS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME")
TRAIN_MODULES = ("classifier_train", "sentiment_train")


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache_environment(monkeypatch, tmp_path):
    for key in CACHE_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Expanding ~ must not inspect or write the actual user's cache.
    home = tmp_path / "用户 home"
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    helper = load_module(monkeypatch, "_hf_cache_fixture", ALGORITHMS / "utils" / "hf_cache.py")
    return helper, home


@pytest.mark.parametrize("selected", [*CACHE_KEYS, None])
def test_cache_precedence_and_resolution_have_no_side_effects(cache_environment, monkeypatch, tmp_path, selected):
    helper, home = cache_environment
    paths = {key: tmp_path / f"{key}-中文 🚀" for key in CACHE_KEYS}
    if selected is not None:
        for key in CACHE_KEYS[CACHE_KEYS.index(selected):]:
            monkeypatch.setenv(key, str(paths[key]))
    before = {key: os.environ.get(key) for key in CACHE_KEYS}
    if selected in CACHE_KEYS[:2]:
        expected = paths[selected]
    elif selected == "HF_HOME":
        expected = paths[selected] / "hub"
    elif selected == "XDG_CACHE_HOME":
        expected = paths[selected] / "huggingface" / "hub"
    else:
        expected = home / ".cache" / "huggingface" / "hub"
    assert helper.resolve_hub_cache() == expected
    assert {key: os.environ.get(key) for key in CACHE_KEYS} == before
    assert not expected.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["~/缓存 🚀", "$IDM_TEST_CACHE_ROOT/缓存 🚀", "${IDM_TEST_CACHE_ROOT}/缓存 🚀"])
def test_cache_expands_home_and_environment(cache_environment, monkeypatch, value):
    helper, home = cache_environment
    monkeypatch.setenv("IDM_TEST_CACHE_ROOT", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", value)
    assert helper.resolve_hub_cache() == home / "缓存 🚀"
    assert not home.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows environment expansion syntax")
def test_windows_percent_variable_expansion(cache_environment, monkeypatch):
    helper, home = cache_environment
    monkeypatch.setenv("IDM_TEST_CACHE_ROOT", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", "%IDM_TEST_CACHE_ROOT%/缓存 🚀")
    assert helper.resolve_hub_cache() == home / "缓存 🚀"
    assert not home.exists()


@pytest.mark.parametrize("key", ["HF_HOME", "XDG_CACHE_HOME"])
def test_home_expansion_precedes_final_hub_expansion(cache_environment, monkeypatch, key):
    helper, home = cache_environment
    monkeypatch.setenv("IDM_TEST_CACHE_ROOT", "~/缓存 🚀")
    monkeypatch.setenv(key, "${IDM_TEST_CACHE_ROOT}")
    expected = home / "缓存 🚀"
    if key == "XDG_CACHE_HOME":
        expected /= "huggingface"
    assert helper.resolve_hub_cache() == expected / "hub"
    assert not home.exists()


@pytest.fixture
def import_model_module(cache_environment, monkeypatch):
    helper, _home = cache_environment

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("Model loading/download must be explicitly stubbed by this test")

    class FakeModel:
        from_pretrained = staticmethod(unexpected_model_call)

    stub("utils", __path__=[str(ALGORITHMS / "utils")])
    monkeypatch.setitem(sys.modules, "utils.hf_cache", helper)
    stub("utils.logger", setup_logger=lambda *_a: logging.getLogger("model-configuration-fixture"))
    stub("classifier", ContentClassifier=object)
    stub("jieba")
    stub("yaml")
    stub("sklearn")
    stub("sklearn.metrics", **{name: object for name in (
        "accuracy_score", "classification_report", "f1_score", "confusion_matrix", "precision_recall_fscore_support",
    )})
    stub("sklearn.model_selection", train_test_split=object)
    torch = stub("torch", Tensor=object, __version__="configuration-fixture", no_grad=lambda: lambda function: function)
    torch.nn = stub("torch.nn")
    torch.nn.functional = stub("torch.nn.functional")
    stub("torch.optim", AdamW=object)
    stub("torch.utils")
    stub("torch.utils.data", DataLoader=object, Dataset=object)
    stub("transformers", AutoModelForSequenceClassification=FakeModel, AutoTokenizer=FakeModel,
         DataCollatorWithPadding=object, get_linear_schedule_with_warmup=object,
         BertForSequenceClassification=FakeModel, BertTokenizer=FakeModel)
    stub("huggingface_hub", snapshot_download=unexpected_model_call)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    original_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "cntext" else original_find(name, *a, **k))
    monkeypatch.setattr(sys, "path", list(sys.path))
    return lambda name: load_module(monkeypatch, f"_model_configuration_{name}", ALGORITHMS / f"{name}.py")


@pytest.mark.parametrize("name", ["sentiment", *TRAIN_MODULES])
@pytest.mark.parametrize("endpoint", [None, "https://synthetic-hub.invalid"])
def test_model_import_preserves_hf_environment_and_does_not_create_cache(import_model_module, monkeypatch, tmp_path, name, endpoint):
    cache = tmp_path / "指定 缓存 🚀"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "legacy-transformers"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_TOKEN", "synthetic-import-only")
    if endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    keys = (*CACHE_KEYS, "TRANSFORMERS_CACHE", "HF_ENDPOINT", "HF_HUB_OFFLINE", "HF_TOKEN")
    before = {key: os.environ.get(key) for key in keys}
    module = import_model_module(name)
    assert {key: os.environ.get(key) for key in keys} == before
    assert list(tmp_path.iterdir()) == []
    if name in TRAIN_MODULES:
        assert module.HF_HUB_CACHE_DIR == cache


def test_classifier_training_log_reference_redacts_source_text(import_model_module):
    module = import_model_module("classifier_train")
    private_text = "SYNTHETIC_PRIVATE_PAGE_CONTENT"
    reference = module._safe_text_reference(private_text)
    assert private_text not in reference
    assert "redacted" in reference and "len=" in reference and "sha256=" in reference


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_explicit_prepare_creates_selected_cache_only(import_model_module, monkeypatch, tmp_path, name):
    selected = tmp_path / "指定 缓存 🚀"
    unused_home = tmp_path / "unused-home"
    monkeypatch.setenv("HF_HUB_CACHE", str(selected))
    monkeypatch.setenv("HF_HOME", str(unused_home))
    module = import_model_module(name)
    assert not selected.exists()
    assert module.configure_huggingface_cache() == selected
    assert (selected / "persistent_models").is_dir()
    assert not unused_home.exists()
    assert os.environ["HF_HOME"] == str(unused_home)


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_local_model_bypasses_cache_preparation_and_download(import_model_module, monkeypatch, tmp_path, name):
    selected = tmp_path / "unused-cache"
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(selected))
    module = import_model_module(name)
    assert module.ensure_model_cached(str(local_model)) == str(local_model)
    assert not selected.exists()


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_invalid_cache_directory_does_not_fall_back_elsewhere(import_model_module, monkeypatch, tmp_path, name):
    selected = tmp_path / "cache-is-a-file"
    selected.write_text("preserve", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(selected))
    module = import_model_module(name)
    with pytest.raises(OSError):
        module.configure_huggingface_cache()
    assert selected.read_text(encoding="utf-8") == "preserve"
    assert list(tmp_path.iterdir()) == [selected]


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_explicit_download_uses_selected_cache_and_reuses_complete_snapshot(import_model_module, monkeypatch, tmp_path, name):
    selected = tmp_path / "指定 缓存 🚀"
    monkeypatch.setenv("HF_HUB_CACHE", str(selected))
    module = import_model_module(name)
    calls = []

    def synthetic_download(**kwargs):
        calls.append(kwargs)
        directory = Path(kwargs["local_dir"])
        # Synthetic empty files check only the existing local-cache discovery.
        for filename in ("config.json", "tokenizer.json", "model.safetensors"):
            (directory / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "snapshot_download", synthetic_download)
    expected = selected / "persistent_models" / "synthetic__fixture-model"
    assert module.ensure_model_cached("synthetic/fixture-model") == str(expected)
    assert calls[0]["cache_dir"] == str(selected)
    assert calls[0]["local_dir"] == str(expected)
    assert module.ensure_model_cached("synthetic/fixture-model") == str(expected)
    assert len(calls) == 1


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_transformers_fallback_receives_selected_cache(import_model_module, monkeypatch, tmp_path, name):
    selected = tmp_path / "fallback-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(selected))
    module = import_model_module(name)
    calls = []
    saved = []

    class SyntheticModel:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return cls()

        def save_pretrained(self, directory):
            saved.append(directory)

    monkeypatch.setattr(module, "snapshot_download", None)
    names = ("AutoTokenizer", "AutoModelForSequenceClassification") if name == "classifier_train" else ("BertTokenizer", "BertForSequenceClassification")
    for model_name in names:
        monkeypatch.setattr(module, model_name, SyntheticModel)
    expected = selected / "persistent_models" / "synthetic__fixture-model"
    assert module.ensure_model_cached("synthetic/fixture-model") == str(expected)
    assert len(calls) == 2
    assert all(source == "synthetic/fixture-model" and kwargs["cache_dir"] == str(selected) for source, kwargs in calls)
    assert saved == [expected, expected]
