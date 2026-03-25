-- Enable pgvector extension for native cosine similarity math
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the dedicated schema
CREATE SCHEMA IF NOT EXISTS pipeline;

-- ==========================================
-- 1. PHOTOS TABLE
-- Tracks the status of processed images
-- ==========================================
CREATE TABLE pipeline.photos (
    id UUID PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL,
    photo_id BIGINT NOT NULL,
    file_path TEXT NOT NULL,
    is_finish_line BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'pending',
    subject_count INT DEFAULT 0,
    matched_count INT DEFAULT 0,
    inference_ms FLOAT DEFAULT 0.0,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent duplicate processing of the same file
    CONSTRAINT uix_photo_project_path UNIQUE (project_id, file_path)
);

-- ==========================================
-- 2. IDENTITIES TABLE (The Vector Gallery)
-- Stores the "Golden Centroids" for matching
-- ==========================================
CREATE TABLE pipeline.identities (
    id UUID PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL,
    bib VARCHAR(20),
    
    -- Use pgvector native types for the centroids
    -- InsightFace (buffalo_l) outputs 512 dimensions
    face_centroid vector(512),
    -- DINOv2 (ViT-Base) outputs 768 dimensions
    reid_centroid vector(768),
    
    enrollment_type VARCHAR(30) DEFAULT 'ghost',
    sighting_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- A project can only have one master identity per bib number
    -- (Partial indexes allow multiple NULL bibs for Ghosts)
    CONSTRAINT uix_identity_proj_bib UNIQUE (project_id, bib)
);

-- Create HNSW or IVFFlat indexes for fast nearest-neighbor search
-- Using cosine distance vector_ops
CREATE INDEX idx_identities_face_cosine ON pipeline.identities 
USING hnsw (face_centroid vector_cosine_ops);

CREATE INDEX idx_identities_reid_cosine ON pipeline.identities 
USING hnsw (reid_centroid vector_cosine_ops);

-- ==========================================
-- 3. SUBJECTS TABLE (The Audit Trail)
-- Stores every single bounding box detection
-- ==========================================
CREATE TABLE pipeline.subjects (
    id UUID PRIMARY KEY,
    photo_id UUID NOT NULL REFERENCES pipeline.photos(id) ON DELETE CASCADE,
    identity_id UUID REFERENCES pipeline.identities(id) ON DELETE SET NULL,
    
    -- Normalized Box [0.0 - 1.0]
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_w FLOAT NOT NULL,
    bbox_h FLOAT NOT NULL,
    
    -- Pixel Box
    px1 INT NOT NULL,
    py1 INT NOT NULL,
    px2 INT NOT NULL,
    py2 INT NOT NULL,
    
    -- AI Confidence & Quality metrics
    confidence FLOAT NOT NULL,
    area_pct FLOAT DEFAULT 0.0,
    face_quality FLOAT DEFAULT 0.0,
    
    -- Store per-frame vectors as raw bytes (bytea) to save RAM and disk space.
    -- We do NOT need to do math on these; they are strictly for debugging/auditing.
    face_enc BYTEA,
    reid_enc BYTEA,
    
    -- OCR Results
    ocr_bib VARCHAR(20),
    ocr_confidence FLOAT,
    
    -- Resolution Results
    assigned_bib VARCHAR(20),
    match_type VARCHAR(30),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups when rendering bounding boxes on a specific photo
CREATE INDEX idx_subjects_photo_id ON pipeline.subjects(photo_id);
-- Index for finding all photos a specific identity appears in
CREATE INDEX idx_subjects_identity_id ON pipeline.subjects(identity_id);