# Information Diet Manager API Contract (hyh)

## 1. Scope and Boundaries
- `hyh` is the backend API orchestration/storage layer.
- `hyh` accepts raw browsing records and persists them.
- `hyh` can trigger `lsj` analysis pipeline, but does not re-implement algorithm logic.

## 2. Data Model (Minimum Viable)
### 2.1 `items` (raw records)
Required fields:
- `url`: string, valid URL
- `title`: string, non-empty
- `ts`: integer, Unix epoch milliseconds
- `source`: `plugin | import`

Optional fields:
- `text`: string (if empty, backend falls back to `title`)
- `lang`, `channel`, `author`, `tags`, `meta`

Backend normalization:
- `text` empty -> fallback to `title`
- long text truncated to first `1000` chars
- `url_hash` and `content_hash` generated server-side

### 2.2 `embeddings`
- one row per `item_id`
- stores vector blob, model id, vector dim
- generated on item insert; analysis only backfills missing vectors

### 2.3 `stats_daily`
- per-day aggregate snapshot
- fields:
  - `total_count`
  - `channel_counts` (JSON)
  - `repeat_ratio`
  - `negative_ratio`
  - `avg_sentiment`

### 2.4 `analysis_runs`
- append-only analysis history for each run
- stores full API payload + cached flag

### 2.5 `analysis_jobs`
- orchestration state machine for API jobs
- statuses: `queued | running | completed | failed`
- stores input window, cache hit, duration, error, result payload

## 3. API Surface
### 3.1 Ingestion
- `POST /collect`
  - body: single `IngestItem`
  - response: `{"inserted": n, "duplicates": m, "failed": 0}`

- `POST /import`
  - body: `multipart/form-data` file (`csv/json/jsonl`)
  - response: `{"inserted": n, "duplicates": m, "failed": k}`

- `GET /items?page=1&page_size=20`
  - list stored raw items

### 3.2 Stats (lightweight)
- `POST /analyze/run?force=false&backfill_limit=2000`
  - fast aggregate without full algorithm pipeline
  - returns cached result when no new items

- `GET /dashboard/summary`
  - latest daily aggregate snapshot

- `GET /dashboard/visualization?days=7&from_ts=&to_ts=&limit_rows=5000`
  - on-demand visualization payload for frontend charts
  - loads raw items in the requested time window, runs `lsj` classify/sentiment/similarity/evaluator pipeline,
    and returns chart-ready global + category time series
  - `from_ts` / `to_ts` are optional Unix epoch milliseconds; if omitted, backend uses `days`
  - response shape:
    - `window`
      - `from_ts`, `to_ts`, `limit_rows`, `input_count`
    - `global`
      - `time_series`: `[{date, count, avg_polarity, avg_similarity, repeat_ratio, negative_ratio, positive_ratio, neutral_ratio}]`
      - `category_distribution`, `sentiment_distribution`, `similarity_histogram`, `hourly_distribution`
    - `categories`
      - keyed by normalized category name such as `entertainment`, `learning`, `news`, `social`
      - each item: `{alias, label, time_series}`
      - alias mapping is stable for frontend drill-down:
        - `entertainment -> ent`
        - `learning -> edu`
        - `news -> news`
        - `social -> soc`
        - `shopping -> shopping`
        - `tools -> tools`
        - `other -> other`
    - `category_aliases`
      - same mapping table for client-side lookup
    - `pipeline_warning`
      - present when `lsj` dependencies are unavailable and payload is degraded
    - `generated_at`

- `GET /analyze/history?limit=20`
  - latest run history records

### 3.3 Full analysis orchestration (`hyh` -> `lsj`)
- `POST /analyze/run_full?force=false&from_ts=&to_ts=&limit_rows=5000`
  - load raw records from `items`
  - call `lsj` pipeline:
    1. classify -> `category`
    2. sentiment -> `sentiment`, `polarity`
    3. similarity -> `similarity`
    4. evaluator -> report
  - persist:
    - `stats_daily` (for dashboard)
    - `analysis_runs` (history)
    - `analysis_jobs` (job lifecycle)
  - if `lsj` runtime dependencies are missing, API still completes with degraded metrics and
    returns `pipeline_warning` in payload

- `GET /analyze/jobs/{job_id}`
  - job status + metadata

- `GET /analyze/result/{job_id}`
  - completed job result payload

## 4. Important Clarification
- Client upload payload does **not** include:
  - `category`, `sentiment`, `polarity`, `similarity`
- These are derived fields generated inside analysis pipeline before evaluator.

## 5. Caching and Idempotency
- same input window + same `item_max_created_at` reuses previous completed result
- cache events are recorded in `analysis_jobs` (`cache_hit = 1`)

## 6. Error Contract
- invalid input -> `400`
- job not found -> `404`
- result requested before completion -> `409`
- pipeline/internal failure -> `500` with `job_id` in detail
