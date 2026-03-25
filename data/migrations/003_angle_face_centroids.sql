-- ═══════════════════════════════════════════════════════════════════
-- Migration 003 — Multi-Angle Face Centroids
-- 2026-06-09
-- ═══════════════════════════════════════════════════════════════════
-- Adds left-profile and right-profile face centroid columns to the
-- identities table.  The existing face_centroid becomes the "frontal"
-- centroid.  Ghost adoption and in-burst clustering compare against
-- all three and take the best match.
-- ═══════════════════════════════════════════════════════════════════

ALTER TABLE pipeline.identities
    ADD COLUMN IF NOT EXISTS face_centroid_left  vector(512),
    ADD COLUMN IF NOT EXISTS face_centroid_right vector(512);

-- HNSW indexes for nearest-neighbour search on profile centroids
CREATE INDEX IF NOT EXISTS ix_identity_face_left_hnsw
    ON pipeline.identities USING hnsw (face_centroid_left vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS ix_identity_face_right_hnsw
    ON pipeline.identities USING hnsw (face_centroid_right vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
