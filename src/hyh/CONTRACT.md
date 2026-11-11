# Information Diet Manager API Contract (hyh)

## 1. Scope and Boundaries
- `hyh` is the backend API orchestration/storage layer.
- `hyh` accepts raw browsing records and persists them.
- `hyh` can trigger `lsj` analysis pipeline, but does not re-implement algorithm logic.

## 2. Data Model (Minimum Viable)
### 2.1 `items` (raw records)
Required fields:
- `url`: HTTP/HTTPS URL, at most 2048 characters
- `title`: string, trimmed before validation; 1 to 300 characters after trimming
- `ts`: strict integer Unix epoch milliseconds, from `0` through `253402300799999` (year 9999); strings, booleans and floats are invalid for `/collect`
- `source`: `plugin | import`

Optional fields:
- `text`: string, at most 2000 characters (if empty, backend falls back to `title`)
- `lang`, `channel`, `author`: strings of at most 16, 32 and 120 characters respectively
- `tags`: at most 10 strings, at most 40 characters each
- `meta`: object containing only JSON values and string keys, with at most 8 nested object/array levels (root object is level 1); finite numbers only. Its compact JSON serialization with literal Unicode must fit in 8192 UTF-8 bytes
- optional fields may be omitted or `null`; unknown top-level properties are ignored for compatibility
- string limits count Unicode code points; `meta` has a byte limit instead. Strings must be valid Unicode encodable as UTF-8; lone surrogate escapes are rejected, while valid pairs such as emoji are accepted

Backend normalization:
- `text` empty -> fallback to `title`
- long text truncated to first `1000` chars
- `url_hash` and `content_hash` generated server-side

### 2.2 `embeddings`
- one row per `item_id`
- stores vector blob, model id, vector dim
- generated on item insert; analysis only backfills missing vectors

### 2.3 `stats_daily`
- compatibility projection of whole-library page statistics, keyed by generation day; it is not a count of pages visited that day
- legacy windowed model runs no longer write this table; old rows are not trusted as the global cache
- fields:
  - `total_count`
  - `channel_counts` (JSON)
  - `repeat_ratio`
  - `negative_ratio`
  - `avg_sentiment`

### 2.4 `analysis_runs`
- append-only analysis history within the retention window; the newest 365 rows are retained and older rows are pruned at startup and after terminal analysis writes
- stores full API payload + cached flag

### 2.5 `analysis_jobs`
- orchestration state machine for API jobs
- statuses: `queued | running | completed | failed`
- stores input window, cache hit, duration, error, result payload
- queued/running jobs are never pruned; at most the newest 500 completed/failed jobs are retained, so an old cache key may be recomputed and an old job ID may return `404`

## 3. API Surface
### 3.0 Local access boundary
- Use the single-process loopback launcher `python scripts/run_backend.py`; the app also checks the actual peer IP, `Host`, and `Origin`. An OS file lock rejects another IDM API process using the same database.
- `GET /health` is the only unauthenticated endpoint and returns only `{"status":"ok"}`. Local peer/Host/Origin checks still apply. It indicates process availability, not that analysis or all storage operations are healthy.
- All other routes require `Authorization: Bearer <key>`. Keys in cookies or query parameters are not accepted. Invalid/missing keys return `401`; insufficient scope returns `403`.
- The **collector** key can only `POST /collect` and `GET /session`. The **admin** key can use all data and analysis routes. `GET /session` returns `{"role":"admin"}` or `{"role":"collector"}`.
- Keys are generated from separate 32-byte random values and loaded at process startup. Default credentials live beside the database as `idm.credentials.json`; logs print the path only. Optional environment keys must both be present, distinct, and 43–128 ASCII URL-safe characters. Empty or partial configuration stops startup.
- Default exact browser origins: `http://127.0.0.1:5173`, `http://localhost:5173`, `http://127.0.0.1:4173`, `http://localhost:4173`. `IDM_FRONTEND_ORIGINS` can replace these with comma-separated exact loopback origins. Chrome extension origins must match `chrome-extension://[a-p]{32}`; knowing an extension ID does not bypass key authentication.
- CORS permits GET/POST/DELETE and Authorization/Content-Type/X-IDM-Confirm headers. Credential cookies are disabled. Allowed preflight requests do not require a key; subsequent data requests do.
- Protected responses include `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. Online docs and the OpenAPI route are disabled; this file is the maintained contract.
- Authenticated request bodies are limited to 64 KiB normally, 10 MiB for `/import`, and 20 MiB for `/data/restore`, including bodies without Content-Length. The total read deadline is 10 seconds. At most four authenticated requests enter body processing at once. Data operations share an ordering lock; dashboard and full model computation instead use short SQLite transactions and version/job-state checks, permitting collection and deletion during inference. Both analysis routes retain a shared single model lock until their actual worker finishes, even after repeated caller cancellation.
- A request that cannot obtain capacity within 0.1 seconds or the data-operation gate within 2 seconds returns `503` with `Retry-After: 5`. Oversize bodies return `413`, malformed Content-Length or duplicate sensitive headers `400`, slow bodies `408`. These are resource bounds, not a completed load/denial-of-service certification. Model execution itself still lacks a hard execution deadline.

### 3.1 Ingestion
- `POST /collect`
  - body: single `IngestItem`
  - successful response is exactly one of `{"inserted": 1, "duplicates": 0, "failed": 0}` or `{"inserted": 0, "duplicates": 1, "failed": 0}`
  - acknowledgement is returned only after the database transaction commits; storage errors must not be acknowledged as success
  - retry after a lost response is safe under the existing normalized-URL unique constraint: a previously committed page returns `duplicates: 1`
  - `duplicates` means the page URL already exists, not that a new visit or updated title/text was saved; repeated URLs retain their original record
  - clients may remove a queued record only after validating non-negative integer acknowledgement fields, `failed == 0`, and `inserted + duplicates == 1`; HTTP 2xx alone is insufficient
  - schema validation failures return `422`. Keep rejected records available for inspection/correction; do not silently drop them or continuously retry an unchanged invalid payload
  - validation error entries contain only `type`, `loc` and `msg`; they do not echo request values or exception contexts. Invalid metadata such as non-finite numbers or lone surrogate escapes is also rejected as `422`

- `POST /import`
  - body: `multipart/form-data` file (`csv/json/jsonl`)
  - response: `{"inserted": n, "duplicates": m, "failed": k}`
  - each parsed item uses the same field limits; invalid items contribute to `failed`. The import parser retains its legacy conversion of textual timestamps before model validation
  - a storage failure rolls back the entire insert transaction, including earlier valid rows and their embeddings; it does not return a partial-success acknowledgement

- `GET /items?page=1&page_size=20`
  - legacy offset pagination; each response is internally consistent, but separate pages can shift under concurrent writes
- `GET /items?pagination=cursor&page_size=50&cursor=...`
  - omit `cursor` for the first page; return `page`, `page_size`, `total`, `items`, `pagination: "cursor"`, `snapshot_at`, `has_more` and nullable `next_cursor`
  - `total` and `snapshot_at` stay fixed for the traversal; later inserts are outside its initial maximum ID and appear after refresh
  - deletion, update or restore invalidates the cursor with `409 items_snapshot_expired`; rollbacks preserve it
  - cursors are signed metadata, not credentials; restart invalidates outstanding cursors and access still requires the admin key
  - keep page size unchanged; do not mix cursor and legacy page/limit parameters

Both pagination modes preserve legitimate stored `tags`/`meta` JSON values, including SQL NULL, JSON null, zero and older valid shapes. Malformed JSON, non-finite numbers (including exponent overflow), invalid UTF-8 text and more than 64 nested containers return `409` with `detail.code: "stored_item_invalid"`. A returned page fails as a whole without disclosing the invalid value or rewriting records/revisions. This is distinct from cursor expiry; a valid earlier page remains readable when only a later page contains invalid JSON. These checks do not enforce the current ingestion business schema on older records or impose a byte/RSS budget.

### 3.2 Stats (lightweight)
- `POST /analyze/run?force=false&backfill_limit=2000`
  - fast aggregate without full algorithm pipeline
  - counts all saved pages with `statistics_scope: "all_saved_pages"`, `statistics_version: 1` and a `data_version`
  - `repeat_metric: "duplicate_normalized_content_hash_fraction"`; `repeat_ratio` is `(valid hashes - distinct hashes) / valid hashes`, with `repeat_sample_count` disclosed; missing/empty hashes are excluded and no valid hashes means null
  - `negative_ratio` and `avg_sentiment` are null because this route does not measure sentiment
  - SQL groups with missing, unknown or aliased channel names are added into their canonical bucket without overwriting one another
  - reuses only explicitly versioned global payloads matching the database identity/revision, count, maximum ID and creation timestamp; same-millisecond inserts and destructive changes invalidate them
  - records each explicit request in history, including cache hits

- `GET /dashboard/summary`
  - the same whole-library statistics contract as `/analyze/run`; never returns a windowed full-analysis result
  - a matching cached summary retains its original generation time and does not add a history row on every poll

- `GET /dashboard/visualization?days=7&from_ts=&to_ts=&limit_rows=5000&force=false`
  - on-demand visualization payload for frontend charts
  - loads raw items ordered by `(ts, id)`, runs `lsj` classify/sentiment/similarity, then independent per-metric aggregation; legacy scoring preprocessing is not used
  - `from_ts` / `to_ts` are optional Unix epoch milliseconds; if omitted, backend uses `days`
  - `force=true` bypasses the cache; the dashboard uses it for every explicit analysis request
  - cache keys carry visualization schema version 3, resolved time boundaries, database identity/revision and maximum ID; only ready results without warnings are reusable
  - a moving window or changed dataset cannot reuse an old key; a cache hit retains its original generation timestamp
  - creation and publication check the destructive-data revision inside write transactions; deletion/restore during computation returns `409 analysis_snapshot_expired` without republishing old results
  - response shape:
    - `window`
      - `from_ts`, `to_ts`, `limit_rows`, `input_count`
      - `available_count`: all saved records in the requested window, counted in the same SQLite read snapshot
      - `truncated`: true when the window contains more records than were loaded
      - `processed_count`: number of processed records, not the number of valid measurements for every metric
      - selection remains ascending by timestamp; a truncated result is the earliest portion, not a representative sample
    - `analysis_status`: `empty | insufficient_data | unavailable | failed | ready`
      - `empty`: no input records in the requested window
      - `insufficient_data`: fewer than `minimum_records` input records; optional model dependencies are not loaded
      - `unavailable`: the pipeline could not run (dependency/configuration/runtime failure)
      - `failed`: visualization postprocessing failed
      - `ready`: chart data was produced; this does not certify model accuracy
      - non-ready responses contain empty chart collections, never invented neutral/zero measurements
    - `minimum_records`: currently 5, an implementation guard; not a scientifically validated sample-size requirement
    - `date_timezone`: currently `UTC`; a rolling seven-day window can intersect eight calendar dates
    - `category_counts`: absolute counts of valid categories; missing similarity or sentiment does not drop a record's category
      - retains `shopping` and `tools` as separate categories
    - `global`
      - `time_series`: `[{date, count, comparison_count, sentiment_count, polarity_count, avg_polarity, avg_similarity, repeat_ratio, negative_ratio, positive_ratio, neutral_ratio}]`
      - similarity/repeat denominators contain only valid original adjacent pairs; the first record has no predecessor; cross-date/category pairs belong to the later record
      - each sentiment proportion uses valid three-class labels as its own denominator; unavailable measurements are null, never fabricated neutral or zero
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
      - diagnostic warning for unavailable/failed analysis; clients must not treat it as a successful measurement
    - `generated_at`
    - `coverage`: `record_count`, `timestamp_count`, `category_count`, `comparison_count`, `sentiment_count`; independent valid quantities, not a model-accuracy claim

- `GET /analyze/history?limit=20`
  - latest run history records, up to the requested limit (maximum 200); `total` is the number of retained rows and `returned` is the number in this response; history is bounded by the 365-row retention policy above

### 3.3 Full analysis orchestration (`hyh` -> `lsj`)
- `POST /analyze/run_full?force=false&from_ts=&to_ts=&limit_rows=5000`
  - load raw records from `items`
  - call `lsj` pipeline:
    1. classify -> `category`
    2. sentiment -> `sentiment`, `polarity`
    3. similarity -> `similarity`
    4. evaluator -> report
  - persist:
    - `analysis_runs` (history)
    - `analysis_jobs` (job lifecycle)
  - returns `statistics_scope: "analysis_window"`, `statistics_version: 2`, `analysis_status: "ready" | "empty"` and the requested `window`; it never overwrites whole-library statistics
  - returns `experimental: true` and an `experimental_notice`; embedded legacy score/health-like fields are informational research output only and must not be presented as a health, psychological, personality, or diagnostic assessment
  - top-level `comparison_count`, `sentiment_count` and `polarity_count` expose each metric's valid denominator; the first page has no predecessor, invalid/out-of-range measurements and invalid flags are excluded, and empty denominators produce null
  - top-level `repeat_metric: "legacy_adjacent_text_similarity_threshold_fraction"` uses valid original adjacent comparisons; the embedded quick/full scoring reports remain experimental legacy outputs with their own preprocessing, not independently validated measurements
  - unavailable dependencies/pipeline return 503, incomplete samples return 422, and unexpected failures return 500; each has a safe fixed message, failed status and `job_id`, without fabricated neutral measurements or raw exception details
  - a savepoint rolls back partially written successful results while retaining the failed job; a database failure that also prevents saving the job cannot guarantee a durable failure record
  - reports containing NaN/infinite JSON numbers, invalid UTF-8 text or more than 64 nested JSON containers are rejected during persistence with a safe failed `500`; they are not saved as completed or silently reinterpreted as null
  - all global/full/dashboard analysis cache reads reject non-finite numbers (including exponent overflow), invalid JSON/Unicode and more than 64 nested containers. Invalid caches are recomputed without deleting historical rows; global statistics recalculate from items and refresh their daily projection
  - a running job is committed before inference, which holds no SQLite connection or data-operation lock. New collection does not change that job's input snapshot; subsequent requests use a new cache key. Deletion/restoration/direct updates invalidate publication with `409 analysis_snapshot_expired`; a removed or non-running original job also invalidates publication, including an empty-library clear with unchanged record revision
  - caller cancellation does not stop model execution: the model gate remains held until the worker exits, and its result may still be saved if the data snapshot is valid. Hard model termination/time limits remain unimplemented
  - empty input does not load optional models; ready means the legacy computation completed, not that its model/scoring accuracy has been accepted

- `GET /analyze/jobs/{job_id}`
  - job status + metadata
  - nonexistent integer IDs, including values outside SQLite's signed 64-bit range, return `404` without database binding overflow

- `GET /analyze/result/{job_id}`
  - completed job result payload
  - the same missing-ID rule applies; an existing incomplete job still returns `409`

Historical job/result/history reads return `409` with `detail.code: "stored_analysis_invalid"` for malformed JSON, non-finite numbers, invalid UTF-8 text or more than 64 nested containers. The root object/array counts as level 1. These reads do not rewrite history or silently convert corruption to null; legitimate SQL NULL/JSON null remain missing measurements. A history list fails as a whole if a returned row is invalid. These checks do not constitute complete report business-schema validation.

### 3.4 Page-record lifecycle (admin only)
- `GET /data/backup`
  - downloads UTF-8 JSON named `idm-pages-backup.json`; at most 10,000 records and 20 MiB
  - envelope: `format: "idm-page-records"`, `version: 1`, `exported_at` (Unix milliseconds), `items`, `sha256`
  - each item contains only `url,title,text,ts,source,lang,channel,author,tags,meta` under the ingestion contract; internal IDs/hashes/created-at timestamps, credentials, extension queues and derived analysis are excluded
  - SHA-256 covers the items array encoded with Python `json.dumps(ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))` as UTF-8. It detects accidental damage, not deliberate tampering by someone who can replace the file
  - records are read in one SQLite snapshot; byte limits are checked during collection and against the final envelope. Oversize data returns `413`; incompatible legacy records return `409` without altering storage
  - only backup fields are materialized after SQLite checks their combined stored bytes against a 128 KiB row budget. This includes legacy values in numeric/source fields; unexported hashes are never loaded. Exceptionally bloated legacy JSON whitespace/escapes can now return `413` even if normalized data would fit
  - preparation has a 15-second cooperative deadline covering SQLite progress, remaining lock-wait time and processing stages. Expiry or database unavailability returns safe `503` with `Retry-After: 5`, before successful download headers. Connections and unfinished buffers are closed on failure. Blocking OS/extension calls are not forcibly terminated, and the 20 MiB file limit is not a process-memory limit
  - the fully encoded backup closes its DB snapshot before delivery and releases the data-operation lock at successful response start. It shares the two-download admission limit with exports, sends at most 64 KiB per chunk with exact Content-Length, and has a 30-second delivery deadline. Cancelling preparation retains admission until the worker finishes and closes its unconsumed buffer. The 20 MiB backup format/restore limit is unchanged
- `DELETE /items/{item_id}` with `X-IDM-Confirm: delete-record`
  - deletes the page and its embedding; clears all derived analysis history/jobs/stats in the same transaction
  - returns `{"deleted":1,"analysis_cleared":true}`, or `404` when absent/outside the positive SQLite integer range
- `DELETE /data` with `X-IDM-Confirm: delete-all`
  - deletes all page records, embeddings and derived analysis atomically; returns `{"deleted":n,"analysis_cleared":true}`
- `POST /data/restore` with `X-IDM-Confirm: replace-records`
  - body is the complete backup JSON, not multipart upload. The explicit confirmation header is also required for an empty backup
  - validates format/version/schema/checksum before deleting anything; one transaction then replaces every page record, regenerates embeddings and clears derived analysis
  - regenerates internal IDs/creation timestamps. The normalized page fields round-trip; this is a logical backup, not a byte-for-byte SQLite restore or migration of all historical database formats
  - duplicate normalized URLs are an error; any failure rolls back deletion and all partial inserts. Returns `{"restored":n,"analysis_cleared":true}` on commit
  - request-task cancellation waits for the replacement worker to commit or roll back and close its connection before releasing data-operation/request capacity. Cancellation does not promise to undo an already authorized restore
  - bad schema/checksum/duplicates return `422`; storage failures return `500`; missing/wrong confirmation returns `400`
- Transactional revision and original-job-state checks prevent either analysis route from republishing results after successful delete/restore. External updates also invalidate results through schema triggers; directly removing triggers or replacing live database files is unsupported.
- Pause the extension and clear its pending queue before deletion/restoration. Already received requests, later uploads or subsequent browsing can recreate records. Downloaded backups and extension storage are outside these API transactions. No forensic erasure guarantee is made.
- Client timeouts/disconnects do not certify rollback. Read current data before retrying a destructive action.

### 3.5 Programmatic data export (admin only)
- `GET /export/lsj?view=analysis|raw&fmt=json|jsonl|csv` exports the selected saved-page fields without the training privacy filter.
- `GET /export/lsj/training?fmt=json|jsonl|csv` returns training rows with `input,label,ts,url,title,source`; existing whitespace normalization, maximum input length and deduplication remain in effect.
- Training defaults to `exclude_internal=true`: local domain names and non-public, multicast, reserved or site-local IP literals are excluded. IPv4-mapped IPv6 addresses use the embedded IPv4 classification; trailing domain dots are normalized. Disabling the option preserves those records.
- This filter performs no DNS lookup or network request. It is not anonymization and does not detect private content in a public URL's path, query or text.
- CSV column order is fixed for each view; zero-row results include the same header as nonempty results. CSV escaping preserves original values, including text beginning with formula characters; these exports target programmatic readers, not execution as spreadsheet formulas.
- General export `tags`/`meta` preserve SQL NULL, JSON null and valid legacy JSON shapes, including UTF-8 JSON bytes. Malformed/non-finite/invalid-Unicode or more-than-64-container-deep values return the existing fixed `409` before delivery; they are never silently replaced with missing values. Training exports do not include or decode these unrelated fields. See `docs/export-json-integrity-2026-10-03.md`.
- Export reads one consistent database snapshot incrementally and prepares a temporary file before sending a successful response. The default/max scanned row counts remain 5,000/200,000 and apply before training filtering/deduplication. JSON object field order is not part of the contract.
- Resource budgets: 64 MiB total encoded output including wrappers/headers; 128 KiB combined source-field bytes per row checked before materializing the row in Python; 32 MiB training deduplication accounting (exact Python strings plus conservative per-entry overhead); at most two exports including their delivery phase. Select a smaller time window after a budget rejection.
- Output/row/deduplication overflow returns `413`; preparation deadline or database unavailability returns `503`; temporary storage failure returns `507`; incompatible legacy values return `409`. Preparation uses a 15-second cooperative deadline, including SQLite progress checks; blocking OS calls are not hard-interrupted.
- Successful export responses include an exact `Content-Length`. After preparation the DB is closed and downloads release the data-operation lock, while retaining bounded request/export capacity. Downloads have a 30-second total deadline and chunks of at most 64 KiB; consumers must reject incomplete downloads. Concurrent deletion/restoration does not alter an already prepared snapshot or write it back.
- Temporary artifacts are explicitly closed on completion, delivery failure, disconnect and cancellation. Cancellation during preparation waits for the bounded worker and closes any result before releasing capacity. Windows files receive a private ACL before any browsing text is written. The ordinary export is not the versioned `/data/backup` format; backup encoding and restore limits remain separate, while delivery admission is shared. See `docs/bounded-exports-2026-10-03.md` for evidence and remaining limits.

## 4. Important Clarification
- The dashboard reads timestamped saved-record snapshots from `GET /items`, independently of analysis; pagination stays in one snapshot until refresh.
- Saved items are deduplicated pages, not visit events or measured reading duration.
- The dashboard requests visualization only on explicit user action. It displays the result as an experimental snapshot.
- Visualization and legacy full-analysis failures/unavailability are recorded as failed jobs, not successful measurements.
- Missing dates/metrics are unknown values, not zero. Zero proportions are valid observations.
- `/analyze/run` and `/dashboard/summary` measure saved-page statistics; `/analyze/run_full` retains experimental legacy scoring for its explicit window. They are distinct contracts, and the current dashboard does not consume them.
- Client upload payload does **not** include:
  - `category`, `sentiment`, `polarity`, `similarity`
- These are derived fields generated inside analysis pipeline before evaluator.

## 5. Caching and Idempotency
- full-analysis cache schema version 2 includes the input window and database identity/revision/count/maximum ID/creation timestamp; only completed, warning-free, explicitly versioned results are reusable
- dashboard visualization uses its separate schema version 3 and resolved time window; global lightweight statistics use version 1 and the dataset state
- cache events are recorded in `analysis_jobs` (`cache_hit = 1`)

## 6. Error Contract
- Startup holds exclusive database process ownership while marking leftover `queued`/`running` analysis jobs `failed`, before the service becomes ready. The fixed reason is `Analysis was interrupted before this service start; run it again.`; `finished_at`/`updated_at` record recovery detection time, not the actual interruption time. Other job fields and statuses remain unchanged; no automatic retry or result generation occurs. Recovery is atomic and failure prevents startup. See `docs/analysis-restart-recovery-2026-10-03.md`.
- Database scopes preserve the first business/setup/commit error when ordinary rollback or close errors also occur; they always attempt close after rollback. A standalone commit/close failure remains visible. This is not a guarantee of cleanup after device failure or of rollback after a commit; see `docs/database-cleanup-errors-2026-10-03.md`.
- Collection, import, deletion, restore, lightweight statistics/summary, model analysis and export/backup preparation retain worker ownership through repeated request cancellation. Upload files remain open until import workers stop reading. Their existing success/error and transaction contracts remain unchanged; cancellation is not an undo operation.
- Each application lifespan tracks these executor Futures and stops admitting new work before shutdown. Already accepted requests that have not started work return fixed `503` with `Retry-After: 5` after closing begins. Lifespan retains the OS database lock until registered workers finish, including Uvicorn's graceful-timeout cancellation path. This does not impose a hard model deadline, cover unregistered read handlers, or guarantee behavior under process crashes/OS force-kill; permanently blocked workers can keep graceful shutdown waiting.
- invalid `/collect` request fields -> `422`; malformed/unsupported import files -> `400`
- job not found -> `404`
- result requested before completion -> `409`
- storage failure during ingestion -> `500`, without a successful acknowledgement; only normalized-URL uniqueness conflicts count as duplicates, other database constraint failures do not
- legacy full-analysis unavailable/incomplete/unexpected failures -> `503` / `422` / `500` with `job_id`, `status: "failed"` and a safe `detail` string
- full analysis and both `/export/lsj` routes accept `from_ts` / `to_ts` only from `0` through `253402300799999`; out-of-range parameters return `422` before SQLite access, and reversed valid boundaries return `400`
