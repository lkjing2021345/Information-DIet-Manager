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

