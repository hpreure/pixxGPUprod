-- Migration 004: Remove multi-angle face centroid columns
-- The angle-bucketed centroids (left/right profile) provided zero
-- incremental matching value while adding per-enrollment overhead.
-- Frontal centroid alone covers all production matching.

DROP INDEX IF EXISTS pipeline.ix_identity_face_left_hnsw;
DROP INDEX IF EXISTS pipeline.ix_identity_face_right_hnsw;

ALTER TABLE pipeline.identities
    DROP COLUMN IF EXISTS face_centroid_left,
    DROP COLUMN IF EXISTS face_centroid_right;
