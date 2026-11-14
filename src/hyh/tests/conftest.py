import pytest

ADMIN_TOKEN = "a" * 43
COLLECTOR_TOKEN = "c" * 43


@pytest.fixture(autouse=True)
def isolated_access_keys(monkeypatch):
    monkeypatch.setenv("IDM_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("IDM_COLLECTOR_TOKEN", COLLECTOR_TOKEN)
    monkeypatch.delenv("IDM_FRONTEND_ORIGINS", raising=False)
