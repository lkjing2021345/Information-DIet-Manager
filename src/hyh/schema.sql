-- SQLite schema draft for Information Diet Manager (MVP)

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT,
    ts INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('plugin', 'import')),

    lang TEXT,
    channel TEXT,
    author TEXT,
    tags TEXT,  -- JSON array
    meta TEXT,  -- JSON object

    url_hash TEXT,      -- hash of normalized url
    content_hash TEXT,  -- hash of normalized title + text
    created_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_items_url_hash
ON items (url_hash);

CREATE INDEX IF NOT EXISTS idx_items_content_hash
ON items (content_hash);

CREATE INDEX IF NOT EXISTS idx_items_ts
ON items (ts);

CREATE INDEX IF NOT EXISTS idx_items_channel
ON items (channel);

-- Embeddings (optional for MVP; can store vector blob or file path)
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL UNIQUE,
    vector BLOB,
    vector_path TEXT,
    model TEXT,
    dim INTEGER,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
);

-- Daily aggregated stats (store analysis results to avoid recompute)
CREATE TABLE IF NOT EXISTS stats_daily (
    day TEXT PRIMARY KEY, -- YYYY-MM-DD
    total_count INTEGER NOT NULL,
    channel_counts TEXT,  -- JSON object
    repeat_ratio REAL,    -- duplicates/total by content_hash
    negative_ratio REAL,  -- placeholder for sentiment
    avg_sentiment REAL,   -- placeholder for sentiment
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
