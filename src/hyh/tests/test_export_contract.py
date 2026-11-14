"""Export HTTP contracts using synthetic records and temporary SQLite only."""
import csv
import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


BASE_TS = 1790208000000
CSV_CASES = [
    ("/export/lsj", {"view": "analysis"},
     ["id", "title", "url", "analysis_text", "text", "channel", "ts", "source", "lang", "author", "tags", "meta"]),
    ("/export/lsj", {"view": "raw"},
     ["id", "url", "title", "text", "ts", "source", "lang", "channel", "author", "tags", "meta", "created_at"]),
    ("/export/lsj/training", {}, ["input", "label", "ts", "url", "title", "source"]),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "synthetic-export.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


def collect(client, index=0, **overrides):
    record = {"url": f"https://example.com/export/{index}", "title": f"Synthetic page {index}",
              "text": f"Synthetic export {index}", "ts": BASE_TS + index,
              "source": "import", "channel": "edu"}
    record.update(overrides)
    response = client.post("/collect", json=record)
    assert response.status_code == 200, response.text
    assert response.json()["inserted"] == 1


@pytest.mark.parametrize("endpoint,params,columns", CSV_CASES)
@pytest.mark.parametrize("empty_window", [False, True])
def test_empty_csv_has_stable_columns_and_is_readable_by_pandas(client, endpoint, params, columns, empty_window):
    params = {**params, "fmt": "csv"}
    if empty_window:
        collect(client)
        params["from_ts"] = BASE_TS + 1
    response = client.get(endpoint, params=params)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"].endswith('.csv"')
    reader = csv.DictReader(io.StringIO(response.text))
    assert reader.fieldnames == columns and list(reader) == []
    frame = pd.read_csv(io.StringIO(response.text))
    assert list(frame.columns) == columns and frame.empty


def test_training_csv_retains_header_when_all_rows_are_filtered(client):
    collect(client, url="http://[fd00::1]/synthetic")
    response = client.get("/export/lsj/training", params={"fmt": "csv"})
    assert response.status_code == 200
    frame = pd.read_csv(io.StringIO(response.text))
    assert list(frame.columns) == CSV_CASES[2][2] and frame.empty


@pytest.mark.parametrize("endpoint,params,columns", CSV_CASES)
def test_csv_round_trips_quotes_unicode_and_literal_training_text(client, endpoint, params, columns):
    title = '=SUM(1,2) "测试"\nsecond line'
    text = '=1+1,"quoted"\n下一行'
    collect(client, title=title, text=text, tags=["测试", 'a,"b'], meta={"nested": [1, "二"]})
    response = client.get(endpoint, params={**params, "fmt": "csv"})
    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert reader.fieldnames == columns and len(rows) == 1
    assert rows[0]["title"] == title
    frame = pd.read_csv(io.StringIO(response.text), keep_default_na=False)
    assert list(frame.columns) == columns and frame.iloc[0]["title"] == title
    if endpoint.endswith("training"):
        # Preserve the existing whitespace normalization, without formula escaping.
        expected = '=1+1,"quoted" 下一行'
        assert rows[0]["input"] == frame.iloc[0]["input"] == expected
        assert rows[0]["label"] == "edu"
    else:
        assert rows[0]["text"] == frame.iloc[0]["text"] == text
        assert json.loads(rows[0]["tags"]) == ["测试", 'a,"b']
        assert json.loads(rows[0]["meta"]) == {"nested": [1, "二"]}
        if params["view"] == "analysis":
            assert rows[0]["analysis_text"] == text


INTERNAL_URLS = [
    "http://127.0.0.2/page", "http://127.255.255.254/page", "http://10.1.2.3/page",
    "http://172.16.0.1/page", "http://172.31.255.254/page", "http://192.168.1.1/page",
    "http://169.254.1.2/page", "http://100.64.0.1/page", "http://0.0.0.0/page",
    "http://224.0.0.1/page", "http://255.255.255.255/page", "http://[::]/page",
    "http://[::1]/page", "http://[fc00::1]/page", "http://[fd00::1]/page",
    "http://[fe80::1]/page", "http://[fec0::1]/page", "http://[feff::1]/page",
    "http://[ff02::1]/page", "http://[::127.0.0.1]/page", "http://[::192.168.1.1]/page",
    "http://[4000::1]/page", "http://[::ffff:127.0.0.2]/page",
    "http://[::ffff:192.168.1.1]/page", "http://[::ffff:100.64.0.1]/page",
    "http://[::ffff:224.0.0.1]/page", "http://LOCALHOST./page", "http://sub.localhost/page",
    "http://sub.localhost./page", "http://host.local/page", "http://host.local./page",
]
PUBLIC_URLS = [
    "https://example.com/page", "https://10.example.com/page", "https://172.16.example.com/page",
    "https://192.168.example.com/page", "https://localhost.example.com/page",
    "https://host.local.example.com/page", "http://8.8.8.8/page", "http://172.15.0.1/page",
    "http://172.32.0.1/page", "http://[2001:4860:4860::8888]/page", "http://[::ffff:8.8.8.8]/page",
]


def training_inputs(response, fmt):
    assert response.status_code == 200, response.text
    if fmt == "json":
        rows = response.json()
    elif fmt == "jsonl":
        rows = [json.loads(line) for line in response.text.splitlines()]
    else:
        rows = list(csv.DictReader(io.StringIO(response.text)))
    return [row["input"] for row in rows]


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_training_filters_address_literals_and_local_names_without_losing_public_hosts(client, fmt, monkeypatch):
    import socket

    def no_dns(*args, **kwargs):
        pytest.fail("Export must not resolve or contact stored browsing hosts")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    for index, url in enumerate(INTERNAL_URLS + PUBLIC_URLS):
        collect(client, index, url=url)
    response = client.get("/export/lsj/training", params={"fmt": fmt})
    assert training_inputs(response, fmt) == [
        f"Synthetic export {index}" for index in range(len(INTERNAL_URLS), len(INTERNAL_URLS) + len(PUBLIC_URLS))
    ]
    response = client.get("/export/lsj/training", params={"fmt": fmt, "exclude_internal": "false"})
    assert training_inputs(response, fmt) == [
        f"Synthetic export {index}" for index in range(len(INTERNAL_URLS) + len(PUBLIC_URLS))
    ]


@pytest.mark.parametrize("view", ["analysis", "raw"])
def test_general_export_remains_unfiltered(client, view):
    collect(client, url="http://[fd00::1]/synthetic")
    response = client.get("/export/lsj", params={"view": view})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["items"][0]["text"] == "Synthetic export 0"


@pytest.mark.parametrize("endpoint", ["/export/lsj", "/export/lsj/training"])
def test_export_still_requires_administrator_capability(client, endpoint):
    response = client.get(endpoint, headers={"Authorization": "Bearer " + "c" * 43})
    assert response.status_code == 403
