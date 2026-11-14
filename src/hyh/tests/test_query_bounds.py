"""Integer query boundaries through real HTTP routes and temporary SQLite."""
import pytest
from fastapi.testclient import TestClient

from src.hyh import app as api, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "query-bounds.sqlite3")
    with TestClient(api.app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                    headers={"Authorization": "Bearer " + "a" * 43}) as client:
        yield client


@pytest.mark.parametrize("endpoint", ["jobs", "result"])
@pytest.mark.parametrize("job_id", [2**63, -(2**63) - 1, 10**100, -(10**100), 2**63 - 1, -(2**63), 0])
def test_missing_or_unrepresentable_job_identifiers_return_the_same_not_found(client, endpoint, job_id):
    response = client.get(f"/analyze/{endpoint}/{job_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "job not found"}


@pytest.mark.parametrize("stored_id", [1, 2**63 - 1])
def test_existing_completed_job_is_accessible_including_maximum_sqlite_identifier(client, stored_id):
    # The empty window creates a real completed job without invoking any model.
    created = client.post("/analyze/run_full")
    assert created.status_code == 200
    original_id = created.json()["job_id"]
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET id = ? WHERE id = ?", (stored_id, original_id))
    job = client.get(f"/analyze/jobs/{stored_id}")
    result = client.get(f"/analyze/result/{stored_id}")
    assert job.status_code == result.status_code == 200
    assert job.json()["id"] == result.json()["job_id"] == stored_id
    assert result.json()["status"] == "completed"
    assert result.json()["result"]["analysis_status"] == "empty"


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
def test_existing_uncompleted_job_preserves_result_conflict(client, status):
    job_id = client.post("/analyze/run_full").json()["job_id"]
    with db.get_conn() as conn:
        conn.execute("UPDATE analysis_jobs SET status = ? WHERE id = ?", (status, job_id))
    job = client.get(f"/analyze/jobs/{job_id}")
    result = client.get(f"/analyze/result/{job_id}")
    assert job.status_code == 200 and job.json()["status"] == status
    assert result.status_code == 409 and result.json() == {"detail": f"job not completed: {status}"}


def seed(client):
    for index in range(3):
        response = client.post("/collect", json={
            "url": f"https://example.invalid/query/{index}", "title": f"Synthetic {index}",
            "text": "Synthetic query boundary", "ts": 1790208000000 + index, "source": "import",
        })
        assert response.status_code == 200 and response.json()["inserted"] == 1


@pytest.mark.parametrize("params", [
    {"page": 2**63 - 1, "page_size": 1}, {"page": 2**63, "page_size": 200},
    {"page": 10**100, "page_size": 200}, {"page": 10**100, "limit": 3},
])
def test_legacy_pagination_extreme_offsets_are_safe_empty_pages(client, params):
    seed(client)
    response = client.get("/items", params=params)
    assert response.status_code == 200
    result = response.json()
    assert result["page"] == params["page"] and result["total"] == 3 and result["items"] == []
    assert result["page_size"] == params.get("limit", params.get("page_size"))


def test_normal_legacy_pages_and_invalid_page_still_use_existing_contract(client):
    seed(client)
    first = client.get("/items", params={"page": 1, "page_size": 2}).json()
    second = client.get("/items", params={"page": 2, "page_size": 2}).json()
    assert first["total"] == second["total"] == 3
    assert [row["id"] for row in first["items"]] == [3, 2]
    assert [row["id"] for row in second["items"]] == [1]
    assert client.get("/items?page=0").status_code == 422


def test_cursor_pagination_still_rejects_page_number_parameters(client):
    response = client.get("/items", params={"pagination": "cursor", "page": 10**100})
    assert response.status_code == 400
    assert response.json() == {"detail": "Cursor pagination uses page_size and cursor only."}
