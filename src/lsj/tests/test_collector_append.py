"""Exercise real collector stores against synthetic legacy CSV/JSONL files."""
import csv
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


UTILS = Path(__file__).resolve().parents[1] / "src" / "algorithms" / "utils"


@pytest.fixture(params=["classifier", "sentiment"])
def store_fixture(request, monkeypatch):
    kind = request.param
    name = f"_append_fixture_{kind}"
    spec = importlib.util.spec_from_file_location(name, UTILS / f"{kind}_data_collector.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    text_field = "input" if kind == "classifier" else "text"
    labels = ["News", "Tools"] if kind == "classifier" else ["平静", "开心"]

    def record(number):
        text = f'合成记录 {number}, "quoted"\n下一行'
        label = labels[number % 2]
        return module.Record(
            entry_id=module.compute_entry_id(text, label), label=label,
            created_at=123.0 + number, **{text_field: text},
        )

    def row(record):
        result = record.to_dict()
        result["id"] = record.entry_id
        if kind == "classifier":
            result.update(text=record.input, title=record.input)
        return result

    def store(path):
        return module.DataStore(str(path), logging.getLogger(name))

    return SimpleNamespace(kind=kind, text_field=text_field, record=record, row=row, store=store, normalize=module.normalize_text)


def assert_reloaded(fixture, store, records):
    loaded, stats = store.load_existing_records()
    assert stats["invalid"] == 0
    assert stats["dedup_removed"] == 0
    assert [(r.entry_id, getattr(r, fixture.text_field), r.label) for r in loaded] == [
        (r.entry_id, fixture.normalize(getattr(r, fixture.text_field)), r.label) for r in records
    ]


@pytest.mark.parametrize("ending", ["", "\n", "\r", "\r\n"])
@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("schema", ["full", "minimal", "alias_extra"])
@pytest.mark.parametrize("header_only", [False, True])
def test_csv_append_preserves_header_mapping_and_existing_bytes(
    store_fixture, tmp_path, ending, bom, schema, header_only,
):
    fixture = store_fixture
    records = [fixture.record(i) for i in range(3)]
    if schema == "full":
        fields = list(reversed(records[0].to_dict()))
    elif schema == "minimal":
        fields = ["label", fixture.text_field]
    else:
        fields = ["custom", "label", "id", "title" if fixture.kind == "classifier" else "text"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    if not header_only:
        row = fixture.row(records[0])
        writer.writerow({field: row.get(field, "original custom value") for field in fields})
    prefix = (("\ufeff" if bom else "") + buffer.getvalue()[:-2] + ending).encode("utf-8")
    path = tmp_path / "records.csv"
    path.write_bytes(prefix)
    store = fixture.store(path)
    store.append_records([records[1]])
    # A second append must continue to use the original schema, without another header.
    store.append_records([records[2]])
    assert path.read_bytes().startswith(prefix)
    assert_reloaded(fixture, store, records[1:] if header_only else records)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == fields
    assert all(None not in row for row in rows)
    if "custom" in fields:
        assert rows[-1]["custom"] == rows[-2]["custom"] == ""
        if not header_only:
            assert rows[0]["custom"] == "original custom value"


@pytest.mark.parametrize("text_alias", ["input", "text", "title"])
@pytest.mark.parametrize("store_fixture", ["classifier"], indirect=True)
def test_classifier_csv_all_supported_text_aliases(store_fixture, tmp_path, text_alias):
    fixture = store_fixture
    path = tmp_path / "alias.csv"
    path.write_text(f"id,label,{text_alias}\n", encoding="utf-8")
    records = [fixture.record(1)]
    store = fixture.store(path)
    store.append_records(records)
    assert_reloaded(fixture, store, records)


@pytest.mark.parametrize("header", ["unsupported,label", "id,model", "text,label,label", "text,label,", "text,label,extra,extra"])
def test_csv_unsupported_header_fails_before_any_write(store_fixture, tmp_path, header):
    fixture = store_fixture
    if fixture.kind == "classifier":
        header = header.replace("text", "input")
    path = tmp_path / "invalid.csv"
    prefix = header.encode("utf-8")
    path.write_bytes(prefix)
    with pytest.raises(OSError, match="CSV header is not supported"):
        fixture.store(path).append_records([fixture.record(1)])
    assert path.read_bytes() == prefix


@pytest.mark.parametrize("ending", ["", "\n", "\r", "\r\n", " ", "\t"])
@pytest.mark.parametrize("bom", [False, True])
def test_jsonl_append_separates_existing_last_record(store_fixture, tmp_path, ending, bom):
    fixture = store_fixture
    records = [fixture.record(i) for i in range(3)]
    prefix = (("\ufeff" if bom else "") + json.dumps(records[0].to_dict(), ensure_ascii=False) + ending).encode("utf-8")
    path = tmp_path / "records.jsonl"
    path.write_bytes(prefix)
    store = fixture.store(path)
    assert_reloaded(fixture, store, records[:1])
    store.append_records(records[1:])
    assert path.read_bytes().startswith(prefix)
    assert_reloaded(fixture, store, records)


@pytest.mark.parametrize("extension", ["csv", "jsonl"])
@pytest.mark.parametrize("initial", [None, b"", b"\xef\xbb\xbf"])
def test_append_creates_parseable_new_or_empty_file(store_fixture, tmp_path, extension, initial):
    fixture = store_fixture
    path = tmp_path / "nested" / f"records.{extension}"
    if initial is not None:
        path.parent.mkdir()
        path.write_bytes(initial)
    records = [fixture.record(1)]
    store = fixture.store(path)
    store.append_records(records)
    assert_reloaded(fixture, store, records)
    if initial is not None:
        assert path.read_bytes().startswith(initial)
    if extension == "csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            assert next(csv.reader(stream)) == list(records[0].to_dict())


def test_empty_batch_does_not_touch_existing_or_missing_files(store_fixture, tmp_path):
    for name in ("missing.csv", "invalid.csv"):
        path = tmp_path / name
        if name == "invalid.csv":
            path.write_bytes(b"unsupported,header")
        store_fixture.store(path).append_records([])
        assert path.exists() == (name == "invalid.csv")
        if path.exists():
            assert path.read_bytes() == b"unsupported,header"
