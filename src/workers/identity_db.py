"""
Pipeline DB — PostgreSQL helpers for the ``pipeline.*`` schema
================================================================
Manages identity centroids, photo records, subject detections,
and timing audit trail in the local PostgreSQL database.

Tables (all in ``pipeline`` schema):
    identities      – one row per (project, bib) centroid bank slot
    identity_shards – multi-centroid appearance variants
    photos          – one row per processed photo
    subjects        – one row per detected person in a photo
    timing_hits     – audit trail for FL timing confirmations
"""

import json
import logging
import uuid
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

from src.detection_config import settings, detection_settings

logger = logging.getLogger(__name__)

# Short alias for detection-logic config (see src/detection_config.py)
_cfg = detection_settings

# ── Pipeline Constants ─────────────────────────────────────────────
CENTROID_MOMENTUM = 0.8          # EMA: 80% old + 20% new
SHARD_SPLIT_THRESHOLD = 0.20     # cosine distance to trigger shard
SHARD_MAX_PER_IDENTITY = 4
SHARD_MIN_SIGHTINGS = 3          # primary must be stable first


# ═══════════════════════════════════════════════════════════════════
# Connection Management
# ═══════════════════════════════════════════════════════════════════

def _connect():
    """Create a new database connection with pgvector adapter."""
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    register_vector(conn)
    return conn


@contextmanager
def get_cursor():
    """Yield a cursor, commit on success, rollback on error."""
    conn = _connect()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════
# Participant Data  (public schema)
# ═══════════════════════════════════════════════════════════════════

def load_participants(project_id: str) -> Dict[str, float]:
    """Return {bib_number: finish_seconds} for timed participants."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT bib_number, finish_time
            FROM public.api_participantinfo
            WHERE project_id = %s AND finish_time IS NOT NULL
        """, (project_id,))
        result = {}
        for bib, ft in cur.fetchall():
            secs = _parse_finish_time(ft)
            if secs is not None:
                result[bib] = secs
        logger.info("Loaded %d timed participants for project %s", len(result), project_id)
        return result


def load_all_bibs(project_id: str) -> set:
    """Return the set of valid bib numbers that have a recorded finish_time.

    DNF / no-show participants (empty or NULL finish_time) are excluded
    because they never crossed the finish line — any OCR reading of
    their bib inside a finish-line frame is a misread of a different
    runner's bib.  Accepting them poisons identity centroids with the
    wrong person's biometrics.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT bib_number FROM public.api_participantinfo
            WHERE project_id = %s
              AND bib_number IS NOT NULL
              AND finish_time IS NOT NULL
              AND finish_time != ''
        """, (project_id,))
        return {row[0] for row in cur.fetchall()}


def load_registered_bibs(project_id: str) -> set:
    """Return ALL registered bib numbers regardless of finish_time.

    This includes participants whose timing data is missing (e.g. timing
    provider malfunction).  Used by Rule 9 to distinguish "registered
    participant without timing" from "completely unknown bib".
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT bib_number FROM public.api_participantinfo
            WHERE project_id = %s
              AND bib_number IS NOT NULL
        """, (project_id,))
        bibs = {row[0] for row in cur.fetchall()}
        logger.info("Loaded %d registered bibs for project %s",
                    len(bibs), project_id)
        return bibs


def _parse_finish_time(time_str) -> Optional[float]:
    """Parse finish time string to seconds since midnight.

    Handles: "9:48:18", "7:28:20.7 AM", "1:30:05.6 PM", etc.
    """
    if not time_str or not isinstance(time_str, str):
        return None
    time_str = time_str.strip()
    try:
        # Detect and strip AM/PM
        is_pm = False
        is_am = False
        upper = time_str.upper()
        if upper.endswith("PM"):
            is_pm = True
            time_str = time_str[:-2].strip()
        elif upper.endswith("AM"):
            is_am = True
            time_str = time_str[:-2].strip()

        parts = time_str.split(':')
        if len(parts) < 2:
            return None
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0

        # 12-hour clock adjustment
        if is_pm and h != 12:
            h += 12
        elif is_am and h == 12:
            h = 0

        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return None


# ═══════════════════════════════════════════════════════════════════
# Vector Helpers
# ═══════════════════════════════════════════════════════════════════

def _vec_to_pg(vec) -> Optional[np.ndarray]:
    """L2-normalise a vector for pgvector storage.

    The pgvector psycopg2 adapter serialises the returned ndarray to
    wire format automatically — no Python string formatting needed.
    """
    if vec is None:
        return None
    arr = np.array(vec, dtype=np.float32).flatten()
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr


def _blend(existing_vec, new_vec, momentum=CENTROID_MOMENTUM) -> Optional[np.ndarray]:
    """EMA blend: result = old*m + new*(1-m), L2-normalised."""
    if new_vec is None:
        return existing_vec
    new = np.array(new_vec, dtype=np.float32).flatten()
    if existing_vec is not None:
        old = np.asarray(existing_vec, dtype=np.float32).flatten()
        blended = old * momentum + new * (1 - momentum)
    else:
        blended = new
    n = np.linalg.norm(blended)
    if n > 0:
        blended = blended / n
    return blended


def _cosine_sim(a_vec, b_vec) -> float:
    """Cosine similarity between two vectors (list / ndarray)."""
    if a_vec is None or b_vec is None:
        return 0.0
    a = np.asarray(a_vec, dtype=np.float32).flatten()
    b = np.asarray(b_vec, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))





# ═══════════════════════════════════════════════════════════════════
# Photo CRUD  (pipeline.photos)
# ═══════════════════════════════════════════════════════════════════

def upsert_photo(cur, *, project_id: str, photo_id: int,
                 file_path: str, capture_time=None,
                 corrected_time: float = None,
                 camera_serial: str = None, camera_model: str = None,
                 is_finish_line: bool = False,
                 file_id: str = None) -> str:
    """Ensure a pipeline.photos row exists. Returns the UUID."""
    cur.execute(
        "SELECT id FROM pipeline.photos WHERE project_id = %s AND file_path = %s",
        (project_id, file_path),
    )
    row = cur.fetchone()
    if row:
        return str(row[0])

    pid = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO pipeline.photos
            (id, project_id, photo_id, file_id, file_path,
             capture_time, corrected_time, camera_serial, camera_model,
             is_finish_line)
        VALUES (%s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s)
        ON CONFLICT (project_id, file_path) DO NOTHING
        RETURNING id
    """, (pid, project_id, photo_id, file_id, file_path,
          capture_time, corrected_time, camera_serial, camera_model,
          is_finish_line))
    result = cur.fetchone()
    return str(result[0]) if result else pid


def update_photo_status(cur, photo_uuid: str, status: str,
                        subject_count: int = 0, matched_count: int = 0,
                        inference_ms: float = 0.0):
    """Update photo row after processing."""
    cur.execute("""
        UPDATE pipeline.photos SET
            status = %s, subject_count = %s, matched_count = %s,
            inference_ms = %s, processed_at = now()
        WHERE id = %s
    """, (status, subject_count, matched_count, inference_ms, photo_uuid))


# ═══════════════════════════════════════════════════════════════════
# Identity CRUD  (pipeline.identities)
# ═══════════════════════════════════════════════════════════════════


# ── Enrollment type rank (lower = higher confidence) ─────────────
# Used to decide whether an incoming enrollment should overwrite an
# existing centroid when the similarity gate rejects a blend.
_ENROLL_RANK = {
    "golden_sample":           0,
    "golden_partial":          1,
    "golden_delayed":          2,
    "error_map_timing":        3,
    "ocr_unvalidated":         4,
    "ocr_registered":          4,
    "blind_trust":             5,
    "hint_remainder":          6,
    "ghost_adopted":           7,
    "ghost":                   8,
}


def _biometric_sim(face_vec, existing_face_pg,
                   reid_vec=None, existing_reid_pg=None) -> float:
    """Best available biometric similarity (face preferred, ReID fallback)."""
    face_sim = _cosine_sim(face_vec, existing_face_pg)
    if face_sim > 0:
        return face_sim
    return _cosine_sim(reid_vec, existing_reid_pg)


def release_subjects_to_ghost(
    cur, identity_id: str, project_id: str,
    incoming_rank: int,
    old_face_pg, old_reid_pg,
) -> Optional[str]:
    """Move all lower-confidence subjects from *identity_id* to a new ghost.

    Releases every subject whose ``match_type`` has a rank **>=**
    *incoming_rank* (i.e. equal or worse confidence than the incoming
    enrollment).  This catches ``hint_remainder``, ``ghost_adopted``,
    ``blind_trust``, etc. in one sweep — not just the single type that
    originally created the identity.

    Creates a fresh ghost identity whose centroid is the **old** centroid
    (before the rank override replaced it).  This allows the ghost adoption
    sweep to match the released subjects to a *different* confirmed identity
    later.

    Returns the new ghost identity UUID, or None if nothing was released.
    """
    # Build the set of match_types whose rank >= incoming_rank
    types_to_release = [
        mt for mt, rank in _ENROLL_RANK.items() if rank >= incoming_rank
    ]
    if not types_to_release:
        return None

    cur.execute("""
        SELECT id FROM pipeline.subjects
        WHERE identity_id = %s AND match_type = ANY(%s)
    """, (identity_id, types_to_release))
    rows = cur.fetchall()
    if not rows:
        return None

    subject_ids = [str(r[0]) for r in rows]

    # Create a new ghost identity with the OLD centroid data
    ghost_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO pipeline.identities
            (id, project_id, bib, face_centroid, reid_centroid, enrollment_type)
        VALUES (%s, %s, NULL, %s::vector, %s::vector, 'ghost')
    """, (ghost_id, project_id, old_face_pg, old_reid_pg))

    # Re-assign subjects to the ghost identity
    cur.execute("""
        UPDATE pipeline.subjects
        SET identity_id = %s, assigned_bib = NULL, match_type = 'ghost'
        WHERE id = ANY(%s::uuid[])
    """, (ghost_id, subject_ids))
    released = cur.rowcount

    logger.info(
        "subjects_released_to_ghost identity=%s ghost=%s "
        "released_types=%s released=%d",
        identity_id, ghost_id, types_to_release, released,
    )
    return ghost_id


def enroll_identity(cur, project_id: str, bib: str,
                    face_vec=None, reid_vec=None,
                    enrollment_type: str = "golden_sample") -> Tuple[str, Optional[str]]:
    """
    Create or update an identity for a confirmed bib from finish-line enrollment.
    Returns (identity_uuid, released_ghost_uuid_or_None).

    When a higher-priority enrollment overrides a lower-priority one, the
    old subjects are released to a new ghost identity whose centroid carries
    the old biometric data — making them eligible for ghost adoption.

    Uses INSERT … ON CONFLICT so that two workers enrolling the same bib
    simultaneously never trigger ``uix_identity_proj_bib``.
    """
    rid = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO pipeline.identities
            (id, project_id, bib, face_centroid, reid_centroid, enrollment_type)
        VALUES (%s, %s, %s, %s::vector, %s::vector, %s)
        ON CONFLICT (project_id, bib) WHERE bib IS NOT NULL
        DO UPDATE SET updated_at = now()
        RETURNING id, face_centroid, reid_centroid, sighting_count, enrollment_type
    """, (rid, project_id, bib,
          _vec_to_pg(face_vec), _vec_to_pg(reid_vec),
          enrollment_type))
    row = cur.fetchone()
    rid = str(row[0])
    existing_face_pg = row[1]
    existing_reid_pg = row[2]
    sighting_count = row[3] or 1
    existing_enrollment = row[4] or "ghost"

    # Blend biometric vectors if provided
    released_ghost_id = None
    if face_vec is not None or reid_vec is not None:
        incoming_rank = _ENROLL_RANK.get(enrollment_type, 8)
        existing_rank = _ENROLL_RANK.get(existing_enrollment, 8)

        has_existing = (existing_face_pg is not None
                        or existing_reid_pg is not None)
        has_incoming = (face_vec is not None or reid_vec is not None)

        if has_existing and has_incoming and sighting_count >= 2:
            face_sim = _cosine_sim(face_vec, existing_face_pg)
            reid_sim = _cosine_sim(reid_vec, existing_reid_pg)
            # Face is authoritative when both sides have vectors;
            # ReID baseline between strangers (~0.5) is too high to
            # gate on.  Only fall back to ReID when face is absent.
            has_face_both = (face_vec is not None
                            and existing_face_pg is not None)
            has_reid_both = (reid_vec is not None
                            and existing_reid_pg is not None)
            if has_face_both:
                sim = face_sim
            elif has_reid_both:
                sim = reid_sim
            else:
                sim = 0.0

            # ── Rank override ─────────────────────────────────────
            # A higher-priority enrollment always replaces a lower-
            # priority one.  The old subjects are released to a ghost
            # and can be ghost-adopted back if biometrics truly match.
            if incoming_rank < existing_rank:
                logger.info(
                    "enrollment_rank_override bib=%s sim=%.3f "
                    "incoming=%s(rank=%d) existing=%s(rank=%d) "
                    "sightings=%d → replacing centroid + ghost release",
                    bib, sim, enrollment_type, incoming_rank,
                    existing_enrollment, existing_rank, sighting_count,
                )
                released_ghost_id = release_subjects_to_ghost(
                    cur, rid, project_id,
                    incoming_rank,
                    existing_face_pg, existing_reid_pg,
                )
                cur.execute("""
                    UPDATE pipeline.identities SET
                        face_centroid   = %s::vector,
                        reid_centroid   = COALESCE(%s::vector, reid_centroid),
                        enrollment_type = %s,
                        sighting_count  = 2,
                        updated_at      = now()
                    WHERE id = %s
                """, (_vec_to_pg(face_vec),
                      _vec_to_pg(reid_vec),
                      enrollment_type, rid))
                return (rid, released_ghost_id)

            # ── Similarity gate ───────────────────────────────────
            # Same or lower rank: reject blend if biometrics diverge.
            if sim < _cfg.ENROLLMENT_MIN_SIM:
                logger.warning(
                    "enrollment_blend_rejected_to_ghost bib=%s sim=%.3f "
                    "threshold=%.2f incoming=%s existing=%s "
                    "sightings=%d → diverting to ghost",
                    bib, sim, _cfg.ENROLLMENT_MIN_SIM,
                    enrollment_type, existing_enrollment,
                    sighting_count,
                )
                ghost_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO pipeline.identities
                        (id, project_id, bib, face_centroid,
                         reid_centroid, enrollment_type)
                    VALUES (%s, %s, NULL, %s::vector,
                            %s::vector, 'ghost')
                """, (ghost_id, project_id,
                      _vec_to_pg(face_vec), _vec_to_pg(reid_vec)))
                return (ghost_id, None)

        shard_created = _update_or_create_shard(
            cur, rid, face_vec, reid_vec,
            existing_face_pg, existing_reid_pg,
            sighting_count,
        )

        promoted_type = enrollment_type if incoming_rank < existing_rank else None

        if not shard_created:
            new_face = _blend(existing_face_pg, face_vec)
            new_reid = _blend(existing_reid_pg, reid_vec)
            set_clauses = [
                "face_centroid = COALESCE(%s::vector, face_centroid)",
                "reid_centroid = COALESCE(%s::vector, reid_centroid)",
            ]
            params = [new_face, new_reid]
            if promoted_type:
                set_clauses.append("enrollment_type = %s")
                params.append(promoted_type)
            set_clauses += [
                "sighting_count = sighting_count + 1",
                "updated_at = now()",
            ]
            params.append(rid)
            cur.execute("""
                UPDATE pipeline.identities SET {sets}
                WHERE id = %s
            """.format(sets=", ".join(set_clauses)), params)
        else:
            if promoted_type:
                cur.execute("""
                    UPDATE pipeline.identities SET
                        enrollment_type = %s,
                        sighting_count  = sighting_count + 1,
                        updated_at      = now()
                    WHERE id = %s
                """, (promoted_type, rid))
            else:
                cur.execute("""
                    UPDATE pipeline.identities SET
                        sighting_count = sighting_count + 1,
                        updated_at = now()
                    WHERE id = %s
                """, (rid,))

    return (rid, None)


def ensure_identity(cur, project_id: str, bib: Optional[str],
                    face_vec=None, reid_vec=None,
                    enrollment_type: str = "ghost") -> str:
    """
    Get or create identity for *bib*. EMA-blend or shard new vectors.
    For ghosts (bib=None), always creates a new row.
    Returns identity UUID.

    Uses INSERT … ON CONFLICT for non-NULL bibs so that two workers
    hitting the same (project_id, bib) simultaneously never trigger
    a ``uix_identity_proj_bib`` duplicate-key error.
    """
    if bib is not None:
        # ── Atomic upsert: INSERT or return existing row ──────────
        rid = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO pipeline.identities
                (id, project_id, bib, face_centroid, reid_centroid, enrollment_type)
            VALUES (%s, %s, %s, %s::vector, %s::vector, %s)
            ON CONFLICT (project_id, bib) WHERE bib IS NOT NULL
            DO UPDATE SET updated_at = now()
            RETURNING id, face_centroid, reid_centroid, sighting_count, enrollment_type
        """, (rid, project_id, bib,
              _vec_to_pg(face_vec), _vec_to_pg(reid_vec),
              enrollment_type))
        row = cur.fetchone()
        rid = str(row[0])
        existing_face_pg = row[1]
        existing_reid_pg = row[2]
        sighting_count = row[3] or 1
        existing_enrollment = row[4] or "ghost"

        # If face/reid vecs are provided, blend them in
        if face_vec is not None or reid_vec is not None:
            incoming_rank = _ENROLL_RANK.get(enrollment_type, 8)
            existing_rank = _ENROLL_RANK.get(existing_enrollment, 8)

            has_existing = (existing_face_pg is not None
                           or existing_reid_pg is not None)
            has_incoming = (face_vec is not None or reid_vec is not None)
            if has_existing and has_incoming and sighting_count >= 2:
                face_sim = _cosine_sim(face_vec, existing_face_pg)
                reid_sim = _cosine_sim(reid_vec, existing_reid_pg)
                # Face is authoritative when both sides have vectors;
                # ReID baseline between strangers (~0.5) is too high
                # to gate on.  Only fall back when face is absent.
                has_face_both = (face_vec is not None
                                and existing_face_pg is not None)
                has_reid_both = (reid_vec is not None
                                and existing_reid_pg is not None)
                if has_face_both:
                    sim = face_sim
                elif has_reid_both:
                    sim = reid_sim
                else:
                    sim = 0.0

                # ── Rank override ─────────────────────────────────
                if incoming_rank < existing_rank:
                    logger.info(
                        "ensure_identity_rank_override bib=%s sim=%.3f "
                        "incoming=%s(rank=%d) existing=%s(rank=%d) "
                        "sightings=%d → replacing centroid + ghost release",
                        bib, sim, enrollment_type, incoming_rank,
                        existing_enrollment, existing_rank, sighting_count,
                    )
                    released_ghost_id = release_subjects_to_ghost(
                        cur, rid, project_id,
                        incoming_rank,
                        existing_face_pg, existing_reid_pg,
                    )
                    cur.execute("""
                        UPDATE pipeline.identities SET
                            face_centroid   = %s::vector,
                            reid_centroid   = COALESCE(%s::vector, reid_centroid),
                            enrollment_type = %s,
                            sighting_count  = 2,
                            updated_at      = now()
                        WHERE id = %s
                    """, (_vec_to_pg(face_vec),
                          _vec_to_pg(reid_vec),
                          enrollment_type, rid))
                    return rid

                # ── Similarity gate ───────────────────────────────
                if sim < _cfg.ENROLLMENT_MIN_SIM:
                    logger.warning(
                        "ensure_identity_blend_rejected_to_ghost "
                        "bib=%s sim=%.3f threshold=%.2f "
                        "incoming=%s existing=%s sightings=%d "
                        "→ diverting to ghost",
                        bib, sim, _cfg.ENROLLMENT_MIN_SIM,
                        enrollment_type, existing_enrollment,
                        sighting_count,
                    )
                    ghost_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO pipeline.identities
                            (id, project_id, bib, face_centroid,
                             reid_centroid, enrollment_type)
                        VALUES (%s, %s, NULL, %s::vector,
                                %s::vector, 'ghost')
                    """, (ghost_id, project_id,
                          _vec_to_pg(face_vec), _vec_to_pg(reid_vec)))
                    return ghost_id

            promoted_type = enrollment_type if incoming_rank < existing_rank else None

            shard_created = _update_or_create_shard(
                cur, rid, face_vec, reid_vec,
                existing_face_pg, existing_reid_pg,
                sighting_count,
            )
            if not shard_created:
                new_face = _blend(existing_face_pg, face_vec)
                new_reid = _blend(existing_reid_pg, reid_vec)
                set_clauses = [
                    "face_centroid = COALESCE(%s::vector, face_centroid)",
                    "reid_centroid = COALESCE(%s::vector, reid_centroid)",
                ]
                params = [new_face, new_reid]
                if promoted_type:
                    set_clauses.append("enrollment_type = %s")
                    params.append(promoted_type)
                set_clauses += [
                    "sighting_count = sighting_count + 1",
                    "updated_at = now()",
                ]
                params.append(rid)
                cur.execute("""
                    UPDATE pipeline.identities SET {sets}
                    WHERE id = %s
                """.format(sets=", ".join(set_clauses)), params)
            else:
                if promoted_type:
                    cur.execute("""
                        UPDATE pipeline.identities SET
                            enrollment_type = %s,
                            sighting_count  = sighting_count + 1,
                            updated_at      = now()
                        WHERE id = %s
                    """, (promoted_type, rid))
                else:
                    cur.execute("""
                        UPDATE pipeline.identities SET
                            sighting_count = sighting_count + 1,
                            updated_at = now()
                        WHERE id = %s
                    """, (rid,))
        return rid
    else:
        # Ghost identity — no bib, always creates a new row
        rid = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO pipeline.identities
                (id, project_id, bib, face_centroid, reid_centroid, enrollment_type)
            VALUES (%s, %s, %s, %s::vector, %s::vector, %s)
        """, (rid, project_id, bib,
              _vec_to_pg(face_vec), _vec_to_pg(reid_vec),
              enrollment_type))
        return rid


# ═══════════════════════════════════════════════════════════════════
# Multi-Centroid Shards
# ═══════════════════════════════════════════════════════════════════

def _get_all_shards(cur, identity_id: str) -> list:
    """Return all shards: [(shard_id, face_pg, reid_pg, sighting_count)]."""
    cur.execute("""
        SELECT id, face_centroid, reid_centroid, sighting_count
        FROM pipeline.identity_shards
        WHERE identity_id = %s ORDER BY shard_index ASC
    """, (identity_id,))
    return [(str(r[0]), r[1], r[2], r[3]) for r in cur.fetchall()]


def _update_or_create_shard(cur, identity_id: str,
                            face_vec, reid_vec,
                            primary_face_pg, primary_reid_pg,
                            primary_sighting_count: int) -> bool:
    """Check if new vectors should create a shard or blend. Returns True if shard created."""
    if primary_sighting_count < SHARD_MIN_SIGHTINGS:
        return False

    primary_face_sim = _cosine_sim(face_vec, primary_face_pg)
    primary_reid_sim = _cosine_sim(reid_vec, primary_reid_pg)
    primary_dist = 1.0 - max(primary_face_sim, primary_reid_sim)

    if primary_dist < SHARD_SPLIT_THRESHOLD:
        return False

    shards = _get_all_shards(cur, identity_id)
    for shard_id, s_face_pg, s_reid_pg, s_count in shards:
        s_face_sim = _cosine_sim(face_vec, s_face_pg)
        s_reid_sim = _cosine_sim(reid_vec, s_reid_pg)
        shard_dist = 1.0 - max(s_face_sim, s_reid_sim)

        if shard_dist < SHARD_SPLIT_THRESHOLD:
            new_face = _blend(s_face_pg, face_vec)
            new_reid = _blend(s_reid_pg, reid_vec)
            cur.execute("""
                UPDATE pipeline.identity_shards SET
                    face_centroid  = COALESCE(%s::vector, face_centroid),
                    reid_centroid  = COALESCE(%s::vector, reid_centroid),
                    sighting_count = sighting_count + 1,
                    updated_at     = now()
                WHERE id = %s
            """, (new_face, new_reid, shard_id))
            return False

    if len(shards) >= SHARD_MAX_PER_IDENTITY:
        return False

    new_idx = len(shards) + 1
    cur.execute("""
        INSERT INTO pipeline.identity_shards
            (id, identity_id, shard_index, face_centroid, reid_centroid)
        VALUES (%s, %s, %s, %s::vector, %s::vector)
    """, (str(uuid.uuid4()), identity_id, new_idx,
          _vec_to_pg(face_vec), _vec_to_pg(reid_vec)))

    logger.info("Created shard #%d for identity %s (dist=%.3f)",
                new_idx, identity_id[:8], primary_dist)
    return True


# ═══════════════════════════════════════════════════════════════════
# Subject Recording  (pipeline.subjects)
# ═══════════════════════════════════════════════════════════════════

def delete_subjects_for_photos(cur, photo_uuids: list) -> int:
    """Delete all existing subject rows for the given photo UUIDs.

    Called before re-inserting subjects so that reprocessing a burst
    does not accumulate duplicate rows.
    Returns the number of rows deleted.
    """
    if not photo_uuids:
        return 0
    cur.execute(
        "DELETE FROM pipeline.subjects WHERE photo_id = ANY(%s::uuid[])",
        (photo_uuids,),
    )
    deleted = cur.rowcount
    if deleted:
        logger.info("Cleaned %d stale subjects for %d photos", deleted, len(photo_uuids))
    return deleted


def record_subjects_batch(cur, subjects_data: List[dict]):
    """Bulk-insert pipeline.subjects rows in ONE query using execute_values.

    Each dict in *subjects_data* must contain the keys:
        id, photo_id, identity_id,
        bbox_x, bbox_y, bbox_w, bbox_h,
        px1, py1, px2, py2,
        confidence, area_pct,
        face_quality, face_enc, reid_enc,
        ocr_bib, ocr_confidence,
        assigned_bib, match_type
    """
    if not subjects_data:
        return

    query = """
        INSERT INTO pipeline.subjects
            (id, photo_id, identity_id,
             bbox_x, bbox_y, bbox_w, bbox_h,
             px1, py1, px2, py2,
             confidence, area_pct,
             face_quality, face_enc, reid_enc,
             ocr_bib, ocr_confidence,
             assigned_bib, match_type)
        VALUES %s
    """

    values = [
        (
            s['id'], s['photo_id'], s['identity_id'],
            s['bbox_x'], s['bbox_y'], s['bbox_w'], s['bbox_h'],
            s['px1'], s['py1'], s['px2'], s['py2'],
            s['confidence'], s.get('area_pct', 0.0),
            s['face_quality'],
            psycopg2.Binary(s['face_enc']) if s.get('face_enc') else None,
            psycopg2.Binary(s['reid_enc']) if s.get('reid_enc') else None,
            s.get('ocr_bib'), s.get('ocr_confidence'),
            s.get('assigned_bib'), s.get('match_type'),
        )
        for s in subjects_data
    ]

    psycopg2.extras.execute_values(cur, query, values, page_size=100)


# ═══════════════════════════════════════════════════════════════════
# Ghost Adoption
# ═══════════════════════════════════════════════════════════════════

def adopt_ghosts_for_bib(cur, project_id: str, bib: str) -> tuple:
    """
    Re-assign ghost subjects whose vectors match the confirmed bib identity.

    Uses pgvector SQL-side cosine filtering (``<=>`` operator) so that
    ghost vectors never leave PostgreSQL unless they actually match.
    NULL-centroid ghosts are excluded at the SQL level.

    Returns (adopted_count, confirmed_id, deleted_ghost_ids).
    """
    # ── Step 1: SQL-side biometric match ──────────────────────────
    # A CTE fetches the confirmed identity, then ghosts are filtered
    # in-database using pgvector's cosine-distance operator.  Only
    # matching ghost IDs come back — no Python vector loop.
    cur.execute("""
        WITH confirmed AS (
            SELECT id, face_centroid, reid_centroid
            FROM pipeline.identities
            WHERE project_id = %s AND bib = %s
            LIMIT 1
        )
        SELECT c.id  AS confirmed_id,
               g.id  AS ghost_id
        FROM   confirmed c,
               pipeline.identities g
        WHERE  g.project_id = %s
          AND  g.bib IS NULL
          AND  g.id != c.id
          AND  g.face_centroid IS NOT NULL
          AND  c.face_centroid IS NOT NULL
          AND (
                -- Path 1: face-strict
                (1 - (g.face_centroid <=> c.face_centroid)) >= %s
                -- Path 2: strong ReID + moderate face
             OR (    g.reid_centroid IS NOT NULL
                 AND c.reid_centroid IS NOT NULL
                 AND (1 - (g.reid_centroid <=> c.reid_centroid)) >= %s
                 AND (1 - (g.face_centroid <=> c.face_centroid)) >= %s)
                -- Path 3: ReID-solo gate
             OR (    g.reid_centroid IS NOT NULL
                 AND c.reid_centroid IS NOT NULL
                 AND (1 - (g.reid_centroid <=> c.reid_centroid)) >= %s
                 AND (1 - (g.face_centroid <=> c.face_centroid)) >= %s)
              )
    """, (
        project_id, bib,
        project_id,
        _cfg.CASCADE_FACE_STRICT,
        _cfg.CASCADE_REID_STRONG, _cfg.CASCADE_FACE_SOFT,
        _cfg.CASCADE_REID_SOLO,   _cfg.CASCADE_REID_SOLO_FACE,
    ))
    rows = cur.fetchall()
    if not rows:
        return (0, None, set())

    confirmed_id = str(rows[0][0])
    matched_ghost_ids = [str(r[1]) for r in rows]

    # ── Step 2: Batch OCR-aware veto (single round-trip) ──────────
    cur.execute("""
        SELECT identity_id, ARRAY_AGG(DISTINCT ocr_bib)
        FROM   pipeline.subjects
        WHERE  identity_id = ANY(%s::uuid[])
          AND  ocr_bib IS NOT NULL
        GROUP  BY identity_id
    """, (matched_ghost_ids,))
    ghost_ocr_map = {str(r[0]): r[1] for r in cur.fetchall()}

    # ── Co-Photo Exclusion: ghosts sharing a photo with the
    #    confirmed identity are proven different people. ────────
    cur.execute("""
        SELECT DISTINCT gs.identity_id::text
        FROM   pipeline.subjects gs
        WHERE  gs.identity_id = ANY(%s::uuid[])
          AND  EXISTS (
                  SELECT 1 FROM pipeline.subjects cs
                  WHERE  cs.identity_id = %s::uuid
                    AND  cs.photo_id = gs.photo_id
               )
    """, (matched_ghost_ids, confirmed_id))
    _co_photo_ghosts = {str(r[0]) for r in cur.fetchall()}

    vetted_ghost_ids: list[str] = []
    for gid in matched_ghost_ids:
        if gid in _co_photo_ghosts:
            continue
        ghost_bibs = ghost_ocr_map.get(gid, [])
        if any(gb != bib and not bib_is_compatible(gb, bib)
               for gb in ghost_bibs):
            continue
        vetted_ghost_ids.append(gid)

    if not vetted_ghost_ids:
        return (0, confirmed_id, set())

    # ── Step 3: Adopt subjects + purge empty ghosts ───────────────
    cur.execute("""
        UPDATE pipeline.subjects
        SET identity_id = %s, assigned_bib = %s, match_type = 'ghost_adopted'
        WHERE identity_id = ANY(%s::uuid[]) AND assigned_bib IS NULL
    """, (confirmed_id, bib, vetted_ghost_ids))
    adopted = cur.rowcount

    cur.execute("""
        DELETE FROM pipeline.identities
        WHERE id = ANY(%s::uuid[])
          AND NOT EXISTS (
              SELECT 1 FROM pipeline.subjects WHERE identity_id = pipeline.identities.id
          )
        RETURNING id
    """, (vetted_ghost_ids,))
    deleted_ghost_ids = {str(row[0]) for row in cur.fetchall()}

    return (adopted, confirmed_id, deleted_ghost_ids)


# ═══════════════════════════════════════════════════════════════════
# Deductive Identity Lookup
# ═══════════════════════════════════════════════════════════════════

def load_identity_centroids(
    project_id: str, bibs: set,
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return {bib: (face_centroid, reid_centroid)} for bibs that already
    have confirmed (non-ghost) identities in the DB.

    Used by the deductive elimination step in the cascade to recognise
    bio clusters that match runners identified in other bursts.
    """
    if not bibs:
        return {}
    bib_list = sorted(bibs)
    with get_cursor() as cur:
        cur.execute("""
            SELECT bib, face_centroid, reid_centroid
            FROM   pipeline.identities
            WHERE  project_id = %s
              AND  bib = ANY(%s)
              AND  bib IS NOT NULL
              AND  face_centroid IS NOT NULL
        """, (project_id, bib_list))
        rows = cur.fetchall()
    result: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for bib, face_c, reid_c in rows:
        face_vec = np.array(face_c, dtype=np.float32) if face_c is not None else None
        reid_vec = np.array(reid_c, dtype=np.float32) if reid_c is not None else None
        result[bib] = (face_vec, reid_vec)
    return result


# ═══════════════════════════════════════════════════════════════════
# Course Photo — Identity Matching Helpers
# ═══════════════════════════════════════════════════════════════════

def load_confirmed_identities(
    cur, project_id: str,
) -> Dict[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return {bib: (identity_id, face_centroid, reid_centroid)} for all
    confirmed (non-ghost) identities in the project.

    Used by the course photo matching path to build the in-RAM candidate
    set for fuzzy bib lookup and biometric validation.
    """
    cur.execute("""
        SELECT id, bib, face_centroid, reid_centroid
        FROM   pipeline.identities
        WHERE  project_id = %s
          AND  bib IS NOT NULL
    """, (project_id,))
    result: Dict[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for row_id, bib, face_c, reid_c in cur.fetchall():
        face_vec = np.array(face_c, dtype=np.float32) if face_c is not None else None
        reid_vec = np.array(reid_c, dtype=np.float32) if reid_c is not None else None
        result[bib] = (str(row_id), face_vec, reid_vec)
    return result


def load_confirmed_bib_strings(
    cur, project_id: str,
) -> Dict[str, str]:
    """Return {bib: identity_id} for all confirmed (non-ghost) identities.

    Lightweight string-only query (~5ms) used as the first stage of
    course photo matching: fuzzy bib matching needs only strings, not
    the 512+768-dim vectors that dominate transfer time.
    """
    cur.execute("""
        SELECT id, bib
        FROM   pipeline.identities
        WHERE  project_id = %s
          AND  bib IS NOT NULL
    """, (project_id,))
    return {bib: str(row_id) for row_id, bib in cur.fetchall()}


def load_centroids_for_bibs(
    cur, project_id: str, bibs: set,
) -> Dict[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Return {bib: (identity_id, face_centroid, reid_centroid)} for specific bibs.

    Second stage of course photo matching: after fuzzy string matching
    narrows to 1-3 candidates, fetch only their vectors for biometric
    validation.  Replaces the 25 MB bulk download with a targeted fetch.
    """
    if not bibs:
        return {}
    bib_list = sorted(bibs)
    cur.execute("""
        SELECT id, bib, face_centroid, reid_centroid
        FROM   pipeline.identities
        WHERE  project_id = %s
          AND  bib = ANY(%s)
          AND  bib IS NOT NULL
    """, (project_id, bib_list))
    result: Dict[str, Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for row_id, bib, face_c, reid_c in cur.fetchall():
        face_vec = np.array(face_c, dtype=np.float32) if face_c is not None else None
        reid_vec = np.array(reid_c, dtype=np.float32) if reid_c is not None else None
        result[bib] = (str(row_id), face_vec, reid_vec)
    return result


def find_nearest_identities(
    cur, project_id: str,
    face_vec, reid_vec,
    top_k: int = 5,
) -> List[Tuple[str, str, float, float]]:
    """pgvector KNN search against confirmed (non-ghost) identities.

    Uses the HNSW index on face_centroid for initial candidate retrieval,
    then computes both face and ReID similarity for the top-K results.

    Returns list of (identity_id, bib, face_sim, reid_sim) sorted by
    face_sim descending.  Empty list if no face vector provided.
    """
    if face_vec is None:
        return []
    face_pg = _vec_to_pg(face_vec)
    reid_pg = _vec_to_pg(reid_vec) if reid_vec is not None else None

    # KNN by face centroid, then compute both similarities server-side
    if reid_pg is not None:
        cur.execute("""
            SELECT id, bib,
                   1 - (face_centroid <=> %s::vector) AS face_sim,
                   CASE WHEN reid_centroid IS NOT NULL
                        THEN 1 - (reid_centroid <=> %s::vector)
                        ELSE 0.0 END AS reid_sim
            FROM   pipeline.identities
            WHERE  project_id = %s
              AND  bib IS NOT NULL
              AND  face_centroid IS NOT NULL
            ORDER  BY face_centroid <=> %s::vector
            LIMIT  %s
        """, (face_pg, reid_pg, project_id, face_pg, top_k))
    else:
        cur.execute("""
            SELECT id, bib,
                   1 - (face_centroid <=> %s::vector) AS face_sim,
                   0.0 AS reid_sim
            FROM   pipeline.identities
            WHERE  project_id = %s
              AND  bib IS NOT NULL
              AND  face_centroid IS NOT NULL
            ORDER  BY face_centroid <=> %s::vector
            LIMIT  %s
        """, (face_pg, project_id, face_pg, top_k))

    results: List[Tuple[str, str, float, float]] = []
    for row_id, bib, f_sim, r_sim in cur.fetchall():
        results.append((str(row_id), bib, float(f_sim), float(r_sim)))
    return results


# ═══════════════════════════════════════════════════════════════════
# Bib Compatibility
# ═══════════════════════════════════════════════════════════════════

def bib_is_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """True if *a* could be a partial/truncated OCR read of *b* or vice-versa.

    Rules (at least 2 digits must match):
      - Exact match
      - Single-digit Hamming (same length)
      - Prefix / suffix
      - In-order subsequence
    """
    if not a or not b:
        return False
    if a == b:
        return True

    la, lb = len(a), len(b)
    if la == lb:
        if la < 2:
            return False
        return sum(1 for x, y in zip(a, b) if x != y) == 1

    long_s, short_s = (b, a) if la < lb else (a, b)
    if len(short_s) < 2:
        return False

    it = iter(long_s)
    return all(ch in it for ch in short_s)


# ═══════════════════════════════════════════════════════════════════
# Scribe Worker — Atomic DB Writes
# ═══════════════════════════════════════════════════════════════════

def scribe_upsert_photo(cur, photo_data: dict) -> str:
    """Upsert a photo row, returning the **actual** DB UUID.

    For new photos, inserts with the deterministic UUIDv5.
    For existing photos (reprocessing), returns the existing uuid
    and updates metadata so that subsequent subject FK references
    always use the correct primary key.
    """
    project_id = photo_data["project_id"]
    file_path = photo_data["file_path"]
    proposed_id = photo_data["uuid"]

    # Check for existing row (reprocessing case)
    cur.execute(
        "SELECT id FROM pipeline.photos WHERE project_id = %s AND file_path = %s",
        (project_id, file_path),
    )
    row = cur.fetchone()
    if row:
        existing_id = str(row[0])
        cur.execute("""
            UPDATE pipeline.photos SET
                corrected_time = %s,
                is_finish_line = %s
            WHERE id = %s
        """, (
            photo_data.get("corrected_time"),
            photo_data.get("is_finish_line", False),
            existing_id,
        ))
        return existing_id

    # New photo — insert with UUIDv5
    cur.execute("""
        INSERT INTO pipeline.photos
            (id, project_id, photo_id, file_path,
             corrected_time, is_finish_line)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (project_id, file_path) DO NOTHING
        RETURNING id
    """, (
        proposed_id, project_id,
        photo_data.get("photo_id"),
        file_path,
        photo_data.get("corrected_time"),
        photo_data.get("is_finish_line", False),
    ))
    result = cur.fetchone()
    return str(result[0]) if result else proposed_id


def execute_scribe_writes(task: dict):
    """Execute an atomic scribe task — one PostgreSQL transaction.

    Used by the DB-Scribe worker (consuming from the ``scribe_tasks``
    RabbitMQ queue).

    Atomic sequence per burst:
      1. Upsert photo rows
      2. Delete stale subjects   (prevents reprocessing duplicates)
      3. Insert new subjects
      4. Update photo status

    The ``task`` dict must contain raw ``bytes`` for face_enc/reid_enc
    (the scribe worker decodes base64 before calling this function).
    """
    photos = task.get("photos", [])
    subject_data = task.get("subjects", [])
    status_data = task.get("photo_status", [])

    with get_cursor() as cur:
        # 1. Upsert photos → build uuid5-to-actual mapping
        uuid5_to_actual: Dict[str, str] = {}
        for p in photos:
            actual_id = scribe_upsert_photo(cur, p)
            uuid5_to_actual[p["uuid"]] = actual_id

        # 2. Delete stale subjects for the ACTUAL photo UUIDs
        actual_uuids = list(set(uuid5_to_actual.values()))
        if actual_uuids:
            delete_subjects_for_photos(cur, actual_uuids)

        # 3. Remap subject photo_ids (uuid5 → actual) + bulk insert
        for s in subject_data:
            s["photo_id"] = uuid5_to_actual.get(s["photo_id"], s["photo_id"])
        if subject_data:
            record_subjects_batch(cur, subject_data)

        # 4. Update photo status
        for ps in status_data:
            actual_uuid = uuid5_to_actual.get(
                ps["photo_uuid"], ps["photo_uuid"]
            )
            update_photo_status(
                cur, actual_uuid, ps["status"],
                subject_count=ps.get("subject_count", 0),
                matched_count=ps.get("matched_count", 0),
                inference_ms=ps.get("inference_ms", 0),
            )

    logger.info(
        "Scribe writes: %d photos, %d subjects, %d status updates",
        len(photos), len(subject_data), len(status_data),
    )


# ═══════════════════════════════════════════════════════════════════
# Project-Level Wipe  (used by the audit dashboard)
# ═══════════════════════════════════════════════════════════════════

def wipe_project(project_id: str) -> Dict[str, int]:
    """Delete ALL pipeline data for *project_id*.

    Deletion order respects FK constraints:
      timing_hits → subjects → identity_shards → identities → photos

    Returns {table_name: rows_deleted}.
    """
    counts: Dict[str, int] = {}
    with get_cursor() as cur:
        # timing_hits
        cur.execute(
            "DELETE FROM pipeline.timing_hits WHERE project_id = %s",
            (project_id,),
        )
        counts["timing_hits"] = cur.rowcount

        # subjects (via photo_id FK)
        cur.execute("""
            DELETE FROM pipeline.subjects
            WHERE photo_id IN (
                SELECT id FROM pipeline.photos WHERE project_id = %s
            )
        """, (project_id,))
        counts["subjects"] = cur.rowcount

        # identity_shards (via identity_id FK)
        cur.execute("""
            DELETE FROM pipeline.identity_shards
            WHERE identity_id IN (
                SELECT id FROM pipeline.identities WHERE project_id = %s
            )
        """, (project_id,))
        counts["identity_shards"] = cur.rowcount

        # identities
        cur.execute(
            "DELETE FROM pipeline.identities WHERE project_id = %s",
            (project_id,),
        )
        counts["identities"] = cur.rowcount

        # photos
        cur.execute(
            "DELETE FROM pipeline.photos WHERE project_id = %s",
            (project_id,),
        )
        counts["photos"] = cur.rowcount

    logger.info(
        "Wiped project %s: %s",
        project_id,
        ", ".join(f"{t}={n}" for t, n in counts.items()),
    )
    return counts
