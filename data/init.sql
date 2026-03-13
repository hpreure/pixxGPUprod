-- pixxEngine Database Schema
-- PostgreSQL 16+ with pgvector
-- ============================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for similarity search

-- ============================================
-- USERS TABLE (for reference/FK)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- PROJECTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_user_id ON projects(user_id);

-- ============================================
-- MEDIA FILES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS media_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    file_path VARCHAR(1024) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50), -- 'image', 'video'
    file_size BIGINT,
    mime_type VARCHAR(100),
    width INTEGER,
    height INTEGER,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'error'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    -- Timing & location enrichment (from VPS EXIF data)
    capture_time TIMESTAMP WITH TIME ZONE,
    camera_serial VARCHAR(100),
    camera_model VARCHAR(100),
    sub_event_id INTEGER,
    sub_event_name VARCHAR(255),
    is_finish_line BOOLEAN DEFAULT false,
    photo_id INTEGER  -- VPS photo ID for cross-reference
);

CREATE INDEX idx_media_files_project_id ON media_files(project_id);
CREATE INDEX idx_media_files_status ON media_files(status);
CREATE INDEX idx_media_files_file_path ON media_files(file_path);

-- ============================================
-- DETECTIONS TABLE (Raw inference results)
-- ============================================
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_file_id UUID REFERENCES media_files(id) ON DELETE CASCADE,
    file_path VARCHAR(1024) NOT NULL,
    object_count INTEGER DEFAULT 0,
    inference_time_ms FLOAT,
    model_version VARCHAR(100),
    raw JSONB, -- Full raw detection output
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_detections_media_file_id ON detections(media_file_id);
CREATE INDEX idx_detections_file_path ON detections(file_path);
CREATE INDEX idx_detections_created_at ON detections(created_at);

-- ============================================
-- PERSONS TABLE (Individual person detections)
-- ============================================
CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    media_file_id UUID REFERENCES media_files(id) ON DELETE CASCADE,
    
    -- Bounding Box (YOLO format: x_center, y_center, width, height normalized)
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    confidence FLOAT,
    
    -- Pixel coordinates (for convenience)
    pixel_x1 INTEGER,
    pixel_y1 INTEGER,
    pixel_x2 INTEGER,
    pixel_y2 INTEGER,
    
    -- Re-ID Feature Vector (ENCRYPTED - TransReID 768-dim typically)
    reid_vector_encrypted BYTEA,
    
    -- Cluster assignment (for grouping same person across images)
    cluster_id INTEGER,
    
    -- Quality metadata (detection pipeline filtering)
    zone_status VARCHAR(20) DEFAULT 'valid',  -- 'valid', 'left_edge', 'right_edge'
    area_pct FLOAT DEFAULT 0.0,               -- Person crop area as fraction of image area
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_persons_detection_id ON persons(detection_id);
CREATE INDEX idx_persons_media_file_id ON persons(media_file_id);
CREATE INDEX idx_persons_cluster_id ON persons(cluster_id);
CREATE INDEX idx_persons_zone_status ON persons(zone_status);
CREATE INDEX idx_persons_media_zone ON persons(media_file_id, zone_status);

-- ============================================
-- RUNNERS TABLE (Unified identities with pgvector)
-- Stores centroid vectors for multi-modal identity fusion
-- ============================================
CREATE TABLE IF NOT EXISTS runners (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(255) NOT NULL,
    last_bib VARCHAR(50),
    face_centroid vector(512),  -- InsightFace embedding dimension
    reid_centroid vector(768),  -- DINOv2 ViT-Base embedding dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW indexes for high-speed project-wide similarity lookups
-- m=16, ef_construction=64 are standard production balances for speed vs accuracy
CREATE INDEX idx_runners_face_hnsw ON runners USING hnsw (face_centroid vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_runners_reid_hnsw ON runners USING hnsw (reid_centroid vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Standard index for project scoping
CREATE INDEX idx_runners_project ON runners(project_id);

-- Add runner_id FK to persons table
ALTER TABLE persons ADD COLUMN IF NOT EXISTS runner_id UUID REFERENCES runners(id);
CREATE INDEX IF NOT EXISTS idx_persons_runner_id ON persons(runner_id);

-- ============================================
-- FACES TABLE (Face detections within persons)
-- ============================================
CREATE TABLE IF NOT EXISTS faces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES persons(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    
    -- Bounding Box relative to original image
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    confidence FLOAT,
    
    -- Face Embedding (ENCRYPTED - InsightFace 512-dim typically)
    face_embedding_encrypted BYTEA,
    
    -- Face Quality Score
    quality_score FLOAT,
    
    -- Optional: matched identity
    identity_id UUID,
    identity_confidence FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_faces_person_id ON faces(person_id);
CREATE INDEX idx_faces_detection_id ON faces(detection_id);
CREATE INDEX idx_faces_identity_id ON faces(identity_id);

-- ============================================
-- BIBS TABLE (Race bib detections)
-- ============================================
CREATE TABLE IF NOT EXISTS bibs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES persons(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    
    -- Bounding Box
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    confidence FLOAT,
    
    -- OCR Results
    bib_number VARCHAR(50),
    ocr_confidence FLOAT,
    ocr_raw_text VARCHAR(255), -- Raw OCR output before cleaning
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_bibs_person_id ON bibs(person_id);
CREATE INDEX idx_bibs_detection_id ON bibs(detection_id);
CREATE INDEX idx_bibs_bib_number ON bibs(bib_number);

-- ============================================
-- TEXT REGIONS TABLE (Text found on bibs)
-- ============================================
CREATE TABLE IF NOT EXISTS text_regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bib_id UUID REFERENCES bibs(id) ON DELETE CASCADE,
    
    -- Bounding Box relative to bib crop
    bbox_x FLOAT NOT NULL,
    bbox_y FLOAT NOT NULL,
    bbox_width FLOAT NOT NULL,
    bbox_height FLOAT NOT NULL,
    confidence FLOAT,
    
    -- OCR Results
    text_content VARCHAR(255),
    ocr_confidence FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_text_regions_bib_id ON text_regions(bib_id);

-- ============================================
-- IDENTITIES TABLE (Known persons for matching)
-- ============================================
CREATE TABLE IF NOT EXISTS identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255),
    external_id VARCHAR(255), -- e.g., race registration ID
    
    -- Reference face embedding (ENCRYPTED)
    reference_face_encrypted BYTEA,
    
    -- Reference ReID vector (ENCRYPTED)
    reference_reid_encrypted BYTEA,
    
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_identities_project_id ON identities(project_id);
CREATE INDEX idx_identities_external_id ON identities(external_id);

-- ============================================
-- PROCESSING JOBS TABLE (Track inference jobs)
-- ============================================
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'queued', -- 'queued', 'processing', 'completed', 'failed'
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_processing_jobs_project_id ON processing_jobs(project_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);

-- ============================================
-- AUDIT LOG TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID,
    action VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    old_data JSONB,
    new_data JSONB,
    performed_by INTEGER REFERENCES users(id),
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX idx_audit_log_performed_at ON audit_log(performed_at);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_identities_updated_at
    BEFORE UPDATE ON identities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- PARTICIPANT INFO TABLE (timing data from race results)
-- ============================================
CREATE TABLE IF NOT EXISTS participant_info (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(255) NOT NULL,
    bib_number VARCHAR(50) NOT NULL,
    finish_time VARCHAR(50),           -- Raw finish time string (e.g. "7:31:13.9 AM")
    finish_seconds FLOAT,              -- Parsed finish time in seconds since midnight
    extra_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, bib_number)
);

CREATE INDEX IF NOT EXISTS idx_participant_project ON participant_info(project_id);
CREATE INDEX IF NOT EXISTS idx_participant_bib ON participant_info(bib_number);

-- ============================================
-- CAMERA OFFSETS TABLE (calibrated per project+camera)
-- ============================================
CREATE TABLE IF NOT EXISTS camera_offsets (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(255) NOT NULL,
    camera_serial VARCHAR(100) NOT NULL,
    camera_model VARCHAR(100),
    offset_seconds FLOAT NOT NULL,      -- Median offset: capture_time - finish_time
    sample_count INTEGER DEFAULT 0,     -- Number of bibs used for calibration
    mad_seconds FLOAT,                  -- Median Absolute Deviation (quality metric)
    calibrated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, camera_serial)
);

CREATE INDEX IF NOT EXISTS idx_camera_offsets_project ON camera_offsets(project_id);

-- ============================================
-- TIMING MATCHES TABLE (GPU-resolved bib→participant mappings)
-- ============================================
CREATE TABLE IF NOT EXISTS timing_matches (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(255) NOT NULL,
    runner_id UUID REFERENCES runners(id) ON DELETE CASCADE,
    participant_bib VARCHAR(50) NOT NULL,
    match_type VARCHAR(50) NOT NULL,
    camera_serial VARCHAR(100),
    time_delta_seconds FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, runner_id)
);

CREATE INDEX IF NOT EXISTS idx_timing_matches_project ON timing_matches(project_id);
CREATE INDEX IF NOT EXISTS idx_timing_matches_runner ON timing_matches(runner_id);

-- ============================================
-- INITIAL DATA (Optional test user)
-- ============================================
INSERT INTO users (id, username, email) 
VALUES (1, 'system', 'system@pixxengine.local')
ON CONFLICT (id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pixxadmin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pixxadmin;
