-- ═══════════════════════════════════════════════════════════════════
-- pixxEngine Pipeline Schema — Identity Bank + Detection Storage
-- Migration 002 — 2026-02-15
-- ═══════════════════════════════════════════════════════════════════
-- Uses the existing 'pipeline' schema and 'vector' extension.
-- Tables:
--   pipeline.identities        — one row per (project, bib) centroid
--   pipeline.identity_shards   — multi-appearance shard centroids
--   pipeline.photos            — one row per processed photo
--   pipeline.subjects          — one row per detected person
--   pipeline.timing_hits       — audit trail for FL timing matches
-- ═══════════════════════════════════════════════════════════════════

CREATE SCHEMA IF NOT EXISTS pipeline;

-- ─────────────────────────────────────────────────────────────────
-- 1. identities — the "identity bank" (one centroid per bib)
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline.identities (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      TEXT NOT NULL,
    bib             TEXT,                          -- NULL = unmatched ghost
    face_centroid   vector(512),                   -- InsightFace ArcFace
    reid_centroid   vector(768),                   -- DINOv2 ViT-B/14
    sighting_count  INTEGER NOT NULL DEFAULT 1,
    enrollment_type TEXT,                          -- golden_sample / blind_trust / etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One identity per (project, bib).  NULLs (ghosts) are not constrained.
CREATE UNIQUE INDEX IF NOT EXISTS uix_identity_proj_bib
    ON pipeline.identities (project_id, bib) WHERE bib IS NOT NULL;

-- HNSW vector indexes for fast similarity search
CREATE INDEX IF NOT EXISTS ix_identity_face_hnsw
    ON pipeline.identities USING hnsw (face_centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS ix_identity_reid_hnsw
    ON pipeline.identities USING hnsw (reid_centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS ix_identity_project
    ON pipeline.identities (project_id);


-- ─────────────────────────────────────────────────────────────────
-- 2. identity_shards — multi-centroid appearance variants
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline.identity_shards (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id     UUID NOT NULL REFERENCES pipeline.identities(id) ON DELETE CASCADE,
    shard_index     INTEGER NOT NULL,
    face_centroid   vector(512),
    reid_centroid   vector(768),
    sighting_count  INTEGER NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uix_shard_identity_idx
    ON pipeline.identity_shards (identity_id, shard_index);

CREATE INDEX IF NOT EXISTS ix_shard_face_hnsw
    ON pipeline.identity_shards USING hnsw (face_centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS ix_shard_reid_hnsw
    ON pipeline.identity_shards USING hnsw (reid_centroid vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);


-- ─────────────────────────────────────────────────────────────────
-- 3. photos — one row per processed photo
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline.photos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      TEXT NOT NULL,
    photo_id        INTEGER,                       -- VPS Photo pk
    file_id         TEXT,                          -- VPS file hash
    file_path       TEXT NOT NULL,
    capture_time    TIMESTAMPTZ,
    corrected_time  DOUBLE PRECISION,              -- Unix epoch after offset correction
    camera_serial   TEXT,
    camera_model    TEXT,
    sub_event_id    INTEGER,
    sub_event_name  TEXT,
    is_finish_line  BOOLEAN NOT NULL DEFAULT FALSE,
    subject_count   INTEGER NOT NULL DEFAULT 0,
    matched_count   INTEGER NOT NULL DEFAULT 0,
    inference_ms    DOUBLE PRECISION,
    status          TEXT NOT NULL DEFAULT 'processing',
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uix_photo_project_path
    ON pipeline.photos (project_id, file_path);

CREATE INDEX IF NOT EXISTS ix_photo_project_status
    ON pipeline.photos (project_id, status);

CREATE INDEX IF NOT EXISTS ix_photo_vps_id
    ON pipeline.photos (photo_id) WHERE photo_id IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────
-- 4. subjects — one detected person per photo
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline.subjects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id        UUID NOT NULL REFERENCES pipeline.photos(id) ON DELETE CASCADE,
    identity_id     UUID REFERENCES pipeline.identities(id) ON DELETE SET NULL,

    -- Normalised bounding box (0.0–1.0)
    bbox_x          DOUBLE PRECISION NOT NULL,
    bbox_y          DOUBLE PRECISION NOT NULL,
    bbox_w          DOUBLE PRECISION NOT NULL,
    bbox_h          DOUBLE PRECISION NOT NULL,
    -- Pixel bounding box
    px1             INTEGER,
    py1             INTEGER,
    px2             INTEGER,
    py2             INTEGER,

    confidence      DOUBLE PRECISION,
    area_pct        DOUBLE PRECISION,

    -- Face biometrics
    face_quality    DOUBLE PRECISION,
    face_enc        BYTEA,                         -- encrypted face embedding

    -- Body / ReID biometrics
    reid_enc        BYTEA,                         -- encrypted ReID embedding

    -- OCR result (best bib read)
    ocr_bib         TEXT,
    ocr_confidence  DOUBLE PRECISION,

    -- Resolved identity
    assigned_bib    TEXT,
    match_type      TEXT,                          -- golden_sample / face_strict / ocr_validated / etc.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_subject_photo
    ON pipeline.subjects (photo_id);

CREATE INDEX IF NOT EXISTS ix_subject_identity
    ON pipeline.subjects (identity_id) WHERE identity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_subject_bib
    ON pipeline.subjects (assigned_bib) WHERE assigned_bib IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────
-- 5. timing_hits — finish-line timing audit trail
-- ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline.timing_hits (
    id              SERIAL PRIMARY KEY,
    project_id      TEXT NOT NULL,
    identity_id     UUID NOT NULL REFERENCES pipeline.identities(id) ON DELETE CASCADE,
    bib             TEXT NOT NULL,
    match_type      TEXT NOT NULL,
    camera_serial   TEXT,
    delta_seconds   DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (project_id, identity_id)
);

-- ═══════════════════════════════════════════════════════════════════
-- Done.
-- ═══════════════════════════════════════════════════════════════════
