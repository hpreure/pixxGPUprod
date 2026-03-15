-- Migration 005: Partial index for ghost adoption performance
--
-- The adopt_ghosts_for_bib() function filters on
--   project_id = %s AND bib IS NULL AND face_centroid IS NOT NULL
-- This partial index lets PostgreSQL use an index scan instead of
-- a sequential scan over the full identities table.
--
-- Run:  psql -h 192.168.1.100 -U pixxadmin -d pixxengine -f data/migrations/005_ghost_adoption_index.sql

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_identity_ghosts
    ON pipeline.identities (project_id)
    WHERE bib IS NULL AND face_centroid IS NOT NULL;
