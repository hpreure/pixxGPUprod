"""
ID Cluster Worker — Identity Clustering & Rule Engine (Pipeline V3)
====================================================================
Tier 2 of the pixxEngine Inference Pipeline V3.

Consumes ``raw_inference_results`` from the GPU worker (local RabbitMQ).
Executes feature-space clustering across cameras and a 13-rule
deterministic cascade for identity resolution.  Publishes identity
intents to the ``scribe_tasks`` queue for atomic DB writes by the
Master Scribe worker.

This worker is **stateless** — it performs no database writes.  All DB
mutations (identity upserts, centroid blending, ghost adoption) are
deferred to the Master Scribe.

Timing-Lock Design Note:
  The V1.5 cpu_worker used a per-frame "timing-lock" hard veto during
  identity clustering to prevent merging detections whose timing-confirmed
  bibs conflict.  In V3, this protection is naturally provided by the
  clustering merge logic's three-step decision per cluster candidate:
    1. Hard Veto — two crops with conflicting OCR cannot merge.
    2. Hard Anchor — two crops with compatible OCR always merge.
    3. Biometric Gravitation — when OCR doesn't resolve the decision
       (at least one side lacks OCR), face cosine similarity decides.
  Therefore, the timing-lock mechanism is not needed in V3 and is not
  carried forward.

Launch::

    source pixxEngine_venv/bin/activate
    python -m src.workers.id_cluster
"""

import base64
import json
import logging
import signal
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog

from src.detection_config import settings, detection_settings
from src.encryption import get_encryptor
from src.messaging import create_local_connection
from src.workers import identity_db as db
from src.workers.scribe_publisher import publish_scribe_task
from src.metrics.log_config import configure as _configure_structlog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_configure_structlog()
logger = structlog.get_logger("id_cluster")

QUEUE_IN = "raw_inference_results"

# Short alias for detection-logic config (see src/detection_config.py)
_cfg = detection_settings

from src.workers.detection_common import (
    status as _status, CYAN, GREEN, YELLOW, BOLD, RESET,
    parse_corrected_time as _parse_corrected_time,
    suppress_overlapping_persons,
    deterministic_photo_uuid,
    encrypt_vec,
)
from src.metrics.burst_logger import log_burst_metrics, log_exception


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _reid_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _photo_epoch_to_sod(epoch: Optional[float]) -> Optional[float]:
    """Convert a corrected_time epoch (float) to seconds since midnight."""
    if epoch is None:
        return None
    try:
        dt = datetime.fromtimestamp(epoch)
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    except (OSError, ValueError, OverflowError):
        return None


def _encrypt_vec_b64(vec: Optional[np.ndarray]) -> Optional[str]:
    """Encrypt a numpy vector and return base64 string for JSON transport."""
    if vec is None:
        return None
    enc_bytes = encrypt_vec(vec)
    if enc_bytes is None:
        return None
    return base64.b64encode(enc_bytes).decode("ascii")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

class CropDetection:
    """A single person detection from a single image."""

    __slots__ = (
        "frame_idx", "photo_id", "bbox", "confidence", "is_blurry",
        "face_quality", "face_yaw", "bibs", "img_width", "img_height",
        "face_vector_b64", "reid_vector_b64",
        "face_vector", "reid_vector", "_had_face",
        "corrected_sod",
    )

    def __init__(self, frame_idx: int, photo_id: str, person_data: dict,
                 img_width: int, img_height: int,
                 corrected_sod: Optional[float] = None):
        self.frame_idx = frame_idx
        self.photo_id = photo_id
        self.corrected_sod = corrected_sod
        self.bbox = person_data["bbox"]
        self.confidence = person_data["confidence"]
        self.is_blurry = person_data.get("is_blurry", False)
        self.face_quality = person_data.get("face_quality", 0.0)
        self.face_yaw = person_data.get("face_yaw", 0.0)
        self.bibs = person_data.get("bibs", [])
        self.img_width = img_width
        self.img_height = img_height

        # Fernet-decrypt biometric vectors from GPU worker payload
        self.face_vector_b64 = person_data.get("face_vector_b64")
        self.reid_vector_b64 = person_data.get("reid_vector_b64")
        self.face_vector = self._deserialize(self.face_vector_b64)
        self.reid_vector = self._deserialize(self.reid_vector_b64)

        # FQ gate: null biometric vectors when face quality is too low
        # to trust.  OCR, bbox, and cluster membership are preserved.
        self._had_face = self.face_vector is not None
        _fq_floor = _cfg.FQ_MIN_BIOMETRIC
        if _fq_floor and self.face_quality < _fq_floor:
            self.face_vector = None
            self.reid_vector = None
            self.face_vector_b64 = None
            self.reid_vector_b64 = None

    @staticmethod
    def _deserialize(b64: Optional[str]) -> Optional[np.ndarray]:
        """Fernet-decrypt a base64 biometric vector from the GPU worker."""
        if not b64:
            return None
        try:
            return get_encryptor().decrypt_vector_b64(b64)
        except Exception as exc:
            logger.warning("biometric_decrypt_failed", error=str(exc))
            return None

    @property
    def best_bib(self) -> Tuple[Optional[str], float]:
        """Return the highest-confidence bib reading from this crop."""
        if not self.bibs:
            return None, 0.0
        best = max(self.bibs, key=lambda x: x.get("ocr_confidence", 0.0))
        return best.get("bib_number"), best.get("ocr_confidence", 0.0)


class IdentityCluster:
    """Cross-camera cluster of CropDetections believed to be the same person."""

    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.detections: List[CropDetection] = []
        self.photo_ids: Set[str] = set()

        # Consensus fields (updated after each crop addition)
        self.consensus_bib: Optional[str] = None
        self.consensus_conf: float = 0.0
        self.best_face_vec: Optional[np.ndarray] = None
        self.best_face_quality: float = 0.0
        self.best_face_yaw: float = 0.0
        self.blended_reid_vec: Optional[np.ndarray] = None

        # Assignment (set by the deterministic cascade)
        self.assigned_bib: Optional[str] = None
        self.match_type: Optional[str] = None

    def add_crop(self, crop: CropDetection):
        """Add a crop to this cluster and update photo_id set."""
        self.detections.append(crop)
        self.photo_ids.add(str(crop.photo_id))

    def compute_consensus(self):
        """Recompute consensus OCR, best face, and blended ReID."""
        valid = [d for d in self.detections if not d.is_blurry] or self.detections

        # ── Consensus OCR — positional character voting ───────────
        readings = []
        for d in valid:
            for b in d.bibs:
                bn = b.get("bib_number")
                conf = b.get("ocr_confidence", 0.0)
                if bn and conf >= _cfg.MIN_OCR_CONF and len(bn) >= _cfg.MIN_BIB_DIGITS:
                    readings.append((bn, conf))

        if readings:
            by_length: dict = defaultdict(list)
            for bib, conf in readings:
                by_length[len(bib)].append((bib, conf))
            best_length = max(by_length, key=lambda k: sum(c for _, c in by_length[k]))
            candidates = by_length[best_length]

            result_chars = []
            max_conf_sum = 0.0
            for pos in range(best_length):
                votes: dict = defaultdict(float)
                for bib, conf in candidates:
                    votes[bib[pos]] += conf
                best_char = max(votes, key=votes.__getitem__)
                result_chars.append(best_char)
                # Use the single strongest reading's confidence for this
                # position so that the final score stays in [0, 1].
                pos_max = max(
                    conf for bib, conf in candidates if bib[pos] == best_char
                )
                max_conf_sum += pos_max

            self.consensus_bib = "".join(result_chars)
            self.consensus_conf = max_conf_sum / best_length
        else:
            self.consensus_bib = None
            self.consensus_conf = 0.0

        # ── Best face (pick highest quality) ──────────────────────
        faces = [(d.face_vector, d.face_quality, d.face_yaw) for d in valid if d.face_vector is not None]
        if faces:
            best = max(faces, key=lambda x: x[1])
            self.best_face_vec = best[0]
            self.best_face_quality = best[1]
            self.best_face_yaw = best[2]

        # ── Blended ReID (mean, L2-normalised) ────────────────────
        reids = [d.reid_vector for d in valid if d.reid_vector is not None]
        if reids:
            mean_vec = np.mean(reids, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-8:
                mean_vec /= norm
            self.blended_reid_vec = mean_vec

    def has_valid_biometrics(self) -> bool:
        """True if the cluster has at least one usable biometric vector."""
        return self.best_face_vec is not None or self.blended_reid_vec is not None

    def has_multiple_conflicting_high_conf_bibs(
        self,
        valid_bibs: set = frozenset(),
        burst_hints: Set[str] = frozenset(),
        registered_bibs: set = frozenset(),
    ) -> bool:
        """True if the cluster has genuinely conflicting bib readings.

        Only the single best bib per crop is considered (highest OCR
        confidence).  Secondary detections (e.g. a neighbouring
        runner's bib leaking into the person crop) are ignored.

        Three-level resolution hierarchy (highest trust first):

        1. **Hint-centric** — if any single ``burst_hint`` is
           ``bib_is_compatible`` with ALL valid bibs in the cluster,
           the hint is the true bib and every other reading is OCR
           noise.  Not a conflict.
        2. **Registered-but-unfinished** — if no hint resolves it but
           ``consensus_bib`` is in ``registered_bibs`` (not in
           ``valid_bibs``), the runner is on-course without timing
           data yet.  If all valid bibs are compatible with the
           consensus → not a conflict.
        3. **Irreconcilable** — no hint and no unfinished-participant
           anchor → true multi-bib collision.
        """
        best_bibs: Set[str] = set()
        for d in self.detections:
            bib, conf = d.best_bib
            if bib and conf >= _cfg.MIN_OCR_CONF:
                best_bibs.add(bib)
        # Keep only valid bibs (registered participants with finish times)
        if valid_bibs:
            best_bibs = {b for b in best_bibs if b in valid_bibs}
        if len(best_bibs) < 2:
            return False

        # Level 1: Hint-centric — timing hardware is ground truth
        for hint in burst_hints:
            if all(db.bib_is_compatible(b, hint) for b in best_bibs):
                return False

        # Level 2: Registered-but-unfinished (on-course runner)
        if (self.consensus_bib
                and self.consensus_bib in registered_bibs
                and self.consensus_bib not in valid_bibs):
            if all(db.bib_is_compatible(b, self.consensus_bib)
                   for b in best_bibs):
                return False

        # Level 3: Irreconcilable — true multi-bib
        return True

    def assign(self, bib: Optional[str], match_type: str):
        """Set the resolved identity assignment."""
        self.assigned_bib = bib
        self.match_type = match_type


# ═══════════════════════════════════════════════════════════════════
# Reference Data Cache  (read-only, populated at startup / cache miss)
# ═══════════════════════════════════════════════════════════════════

class ProjectReferenceCache:
    """Caches valid_bibs, registered_bibs, and timed_participants per project.

    These are the ONLY DB reads id_cluster performs — once at startup
    per project, cached across bursts.
    """

    def __init__(self):
        self._bibs: Dict[str, set] = {}
        self._registered: Dict[str, set] = {}
        self._timed: Dict[str, Dict[str, float]] = {}

    def get_valid_bibs(self, project_id: str) -> set:
        if project_id not in self._bibs:
            self._bibs[project_id] = db.load_all_bibs(project_id)
        return self._bibs[project_id]

    def get_registered_bibs(self, project_id: str) -> set:
        if project_id not in self._registered:
            self._registered[project_id] = db.load_registered_bibs(project_id)
        return self._registered[project_id]

    def get_timed_participants(self, project_id: str) -> Dict[str, float]:
        if project_id not in self._timed:
            self._timed[project_id] = db.load_participants(project_id)
        return self._timed[project_id]


_ref_cache = ProjectReferenceCache()


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Feature-Space Clustering
# ═══════════════════════════════════════════════════════════════════

def cluster_burst_detections(
    images: List[dict],
    burst_hints: Optional[Set[str]] = None,
) -> List[IdentityCluster]:
    """Cluster person crops across all cameras using OCR + face biometrics.

    Merge decision per (crop, cluster) pair — five steps:

      1. **Ambiguous Partial Pre-Filter** — if the crop's OCR bib is
         NOT an exact hint but is ``bib_is_compatible`` with 2+ hints,
         the reading is ambiguous (e.g. "32" ↔ 3223 & 3224).  The
         crop skips the cluster loop entirely and is forced into a
         solo cluster so the cascade can ghost it via Rule 10.  ONLY
         an exact match in ``burst_hints`` overrides this —
         ``valid_bibs`` is intentionally not checked because a bib
         that finished hours ago is not proof of physical mat presence.
      2. **Hard Veto** — if both have OCR and the bibs conflict, skip.
      3. **Hint Disambiguation Veto** — if both have compatible (but
         non-identical) OCR and *both* bibs are in ``burst_hints``,
         they are physically co-present on the mat → treat as conflict.
         STRICTLY hints-only to avoid fracturing clusters over typos
         of runners who finished hours apart.
      4. **Biometric Gravitation (PRIMARY)** — merge if face cosine
         similarity >= FACE_MODERATE_SIM AND reid >= CLUSTER_REID_MIN.
         Biometrics decide identity; OCR is not consulted.
      5. **OCR Anchor (FALLBACK)** — if biometrics didn't match but
         OCR is compatible, merge UNLESS both face AND reid are below
         hostile thresholds (clearly different people sharing a partial
         bib reading).  **Suppressed** when the crop's OCR is an
         ambiguous partial (compatible with ≥2 hints) — such crops
         must merge via biometrics or remain solo.

    Rule 13 (Concurrent Frame Veto): crops from the same photo_id
    can never merge into the same cluster.

    Consensus OCR, best face, and blended ReID are recomputed after
    every crop addition so subsequent crops see up-to-date cluster state.
    """
    _hints = burst_hints or set()
    clusters: List[IdentityCluster] = []

    # Sort images by camera_serial (then photo_id) so same-camera
    # crops cluster first, producing tighter intra-camera centroids
    # before cross-camera crops attempt to join.
    images = sorted(images, key=lambda img: (
        img.get("camera_serial") or "", img.get("photo_id", 0)))

    for frame_idx, img in enumerate(images):
        if not img.get("success") or not img.get("persons"):
            continue

        pid = str(img["photo_id"])
        img_w = img.get("img_width", 1920)
        img_h = img.get("img_height", 1280)

        _ct = _parse_corrected_time(img.get("corrected_time"))
        _sod = _photo_epoch_to_sod(_ct) if _ct is not None else None

        crops = [CropDetection(frame_idx, pid, p, img_w, img_h, _sod)
                 for p in img["persons"]]

        # Hard gate: faceless rejection
        crops = [c for c in crops if c._had_face]

        # Containment-aware NMS (suppress duplicate detections)
        crops = suppress_overlapping_persons(crops)

        for crop in crops:
            matched_cluster = None
            crop_bib, crop_conf = crop.best_bib
            crop_has_ocr = (
                crop_bib is not None
                and crop_conf >= _cfg.MIN_OCR_CONF
                and len(crop_bib) >= _cfg.MIN_BIB_DIGITS
            )

            # ── 1. Ambiguous Partial Pre-Filter ──────────────────────
            # If the OCR partial matches ≥2 hints, flag it so that
            # Step 5 (OCR Anchor) is suppressed — but Step 4
            # (Biometric Gravitation) still runs.  This ensures
            # ambiguous partials can still merge biometrically
            # instead of being quarantined into solo clusters.
            is_ambiguous_ocr = False
            if crop_has_ocr and crop_bib not in _hints:
                compat_hints = [h for h in _hints if db.bib_is_compatible(crop_bib, h)]
                if len(compat_hints) >= 2:
                    is_ambiguous_ocr = True

            for cluster in clusters:
                # ── Rule 13: Concurrent Frame Veto ──
                if str(crop.photo_id) in cluster.photo_ids:
                    continue

                ocr_match = False
                ocr_conflict = False

                if crop_has_ocr and cluster.consensus_bib:
                    if db.bib_is_compatible(crop_bib, cluster.consensus_bib):
                        ocr_match = True
                    else:
                        ocr_conflict = True

                # ── 2. Hard Veto (Clearly different people) ──
                if ocr_conflict:
                    continue

                # ── 3. Hint Disambiguation Veto (The 3223 vs 3224 collision) ──
                # STRICTLY _hints only. If they are both physically on the mat,
                # they are not a typo of each other.
                if (ocr_match
                        and crop_bib != cluster.consensus_bib
                        and crop_bib in _hints
                        and cluster.consensus_bib in _hints):
                    continue

                # ── 4. Biometric Gravitation (PRIMARY — dual-gate) ──
                # Both face AND ReID must agree to merge.
                if (crop.face_vector is not None
                        and cluster.best_face_vec is not None):
                    face_sim = _reid_cosine(crop.face_vector, cluster.best_face_vec)
                    reid_sim = 0.0
                    if (crop.reid_vector is not None
                            and cluster.blended_reid_vec is not None):
                        reid_sim = _reid_cosine(crop.reid_vector, cluster.blended_reid_vec)
                    if face_sim >= _cfg.FACE_MODERATE_SIM and reid_sim >= _cfg.CLUSTER_REID_MIN:
                        matched_cluster = cluster
                        break

                # ── 5. OCR Anchor (FALLBACK — compatible OCR) ──
                # Only fires when biometric gravitation did not
                # match (vectors missing or below dual-gate).
                # Suppressed for ambiguous partials — they must
                # merge on biometrics or stay solo.
                if ocr_match and not is_ambiguous_ocr:
                    bio_hostile = False
                    if (crop.face_vector is not None
                            and cluster.best_face_vec is not None):
                        ck_face = _reid_cosine(crop.face_vector, cluster.best_face_vec)
                        ck_reid = 0.0
                        if (crop.reid_vector is not None
                                and cluster.blended_reid_vec is not None):
                            ck_reid = _reid_cosine(crop.reid_vector, cluster.blended_reid_vec)
                        if ck_face < _cfg.OCR_ANCHOR_HOSTILE_FACE or ck_reid < _cfg.OCR_ANCHOR_HOSTILE_REID:
                            bio_hostile = True
                    if not bio_hostile:
                        matched_cluster = cluster
                        break

            # Fallback: start a new cluster (handles Ambiguous Partials automatically)
            if matched_cluster is None:
                matched_cluster = IdentityCluster()
                clusters.append(matched_cluster)

            matched_cluster.add_crop(crop)
            matched_cluster.compute_consensus()

    return clusters


# ═══════════════════════════════════════════════════════════════════
# Burst Hint Calculation
# ═══════════════════════════════════════════════════════════════════

def calculate_burst_hints(
    images: List[dict],
    timed_participants: Dict[str, float],
    window_s: float,
) -> Set[str]:
    """Return the set of bibs whose finish_time falls within ±window_s
    of any photo's corrected_time in the burst."""
    if not timed_participants:
        return set()
    hints: Set[str] = set()
    for img in images:
        ct = _parse_corrected_time(img.get("corrected_time"))
        if ct is None:
            continue
        sod = _photo_epoch_to_sod(ct)
        if sod is None:
            continue
        for bib, finish_sod in timed_participants.items():
            if abs(sod - finish_sod) <= window_s:
                hints.add(bib)
    return hints


# ═══════════════════════════════════════════════════════════════════
# Phase 2: The 13-Rule Deterministic Cascade
# ═══════════════════════════════════════════════════════════════════

def run_cascade(
    clusters: List[IdentityCluster],
    burst_hints: Set[str],
    valid_bibs: set,
    timed_participants: Dict[str, float],
    burst_sod: Optional[float] = None,
    registered_bibs: Optional[set] = None,
    project_id: Optional[str] = None,
) -> None:
    """Apply the 13-rule deterministic cascade to resolve cluster identities.

    Mutates each cluster's ``assigned_bib`` and ``match_type`` in place.
    Clusters with OCR are evaluated first (highest confidence first).
    """
    _registered = registered_bibs or set()
    unassigned = sorted(
        clusters,
        key=lambda c: (c.consensus_bib is not None, c.consensus_conf),
        reverse=True,
    )
    unassigned = list(unassigned)
    claimed_hints: Set[str] = set()

    for c in list(unassigned):
        # ── Rule 11: Multi-Bib Collision ──────────────────────────
        if c.has_multiple_conflicting_high_conf_bibs(
                valid_bibs, burst_hints, _registered):
            c.assign(None, "ghost_multi_bib")
            unassigned.remove(c)
            continue

        if c.consensus_bib:
            # ── Rule 1: The Golden Match (exact OCR + timing hint) ──
            if (c.consensus_bib in burst_hints
                    and c.consensus_bib not in claimed_hints):
                c.assign(c.consensus_bib, "golden_sample")
                claimed_hints.add(c.consensus_bib)
                unassigned.remove(c)
                continue

            avail_hints = burst_hints - claimed_hints
            compatible = [
                h for h in avail_hints
                if db.bib_is_compatible(c.consensus_bib, h)
            ]

            # ── Rule 10: Ambiguous Partial (1:N substring conflict) ──
            if len(compatible) > 1:
                c.assign(None, "ghost_ambiguous_partial")
                unassigned.remove(c)
                continue

            # ── Rule 2: Partial Golden (1:1 substring match) ──────
            if len(compatible) == 1:
                c.assign(compatible[0], "golden_partial")
                claimed_hints.add(compatible[0])
                unassigned.remove(c)
                continue

            # ── Rule 3: Error-Map Rescue ──────────────────────────
            if c.consensus_bib in _cfg.OCR_ERROR_MAP:
                mapped = _cfg.OCR_ERROR_MAP[c.consensus_bib]
                rescues = [h for h in mapped if h in avail_hints]
                if len(rescues) == 1:
                    c.assign(rescues[0], "error_map_timing")
                    claimed_hints.add(rescues[0])
                    unassigned.remove(c)
                    continue

            # ── Rule 12: Lingering Finisher (outside HINT_WINDOW_S) ──
            if c.consensus_bib in timed_participants:
                if burst_sod is not None:
                    finish_sod = timed_participants[c.consensus_bib]
                    delta = abs(burst_sod - finish_sod)
                    if delta > _cfg.DELAYED_MAX_DELTA_S:
                        # OCR bib is temporally incompatible with this
                        # photo.  Before ghosting, check whether a
                        # timing-compatible alternative exists via
                        # bib_is_compatible (e.g. OCR "3907" → bib 3947
                        # whose finish_time is only seconds away).
                        timing_rescues = [
                            bib for bib, fsod in timed_participants.items()
                            if bib != c.consensus_bib
                            and bib not in claimed_hints
                            and db.bib_is_compatible(c.consensus_bib, bib)
                            and abs(burst_sod - fsod) <= _cfg.DELAYED_MAX_DELTA_S
                        ]
                        if len(timing_rescues) == 1:
                            # Unambiguous reroute
                            c.assign(timing_rescues[0], "golden_delayed")
                            unassigned.remove(c)
                            continue
                        # 0 or 2+ matches → ghost (don't guess)
                        c.assign(None, "ghost")
                        unassigned.remove(c)
                        continue
                c.assign(c.consensus_bib, "golden_delayed")
                unassigned.remove(c)
                continue

            # ── Rule 6: Unvalidated Orphan (valid bib, no timing) ──
            if c.consensus_bib in valid_bibs:
                c.assign(c.consensus_bib, "ocr_unvalidated")
                unassigned.remove(c)
                continue

            # ── Rule 9a: Registered Orphan (registered bib, no timing) ──
            if c.consensus_bib in _registered:
                c.assign(c.consensus_bib, "ocr_registered")
                unassigned.remove(c)
                continue

            # ── Rule 9b: Hard Conflict ────────────────────────────
            c.assign(None, "ghost")
            unassigned.remove(c)

    # ── Evaluate remaining clusters (no OCR) ──────────────────────
    bio_clusters = [c for c in unassigned if c.has_valid_biometrics()]

    # ── Per-cluster hint pools (Solution A) ───────────────────────
    #   Each bio cluster sees only hints whose finish_time is within
    #   ±HINT_WINDOW_S of its member photos' corrected_times.
    for c in bio_clusters:
        sods = {d.corrected_sod for d in c.detections
                if d.corrected_sod is not None}
        c_hints: Set[str] = set()
        for sod in sods:
            for bib, fsod in timed_participants.items():
                if abs(sod - fsod) <= _cfg.HINT_WINDOW_S:
                    c_hints.add(bib)
        c._cluster_hints = c_hints

    # ── Deductive Elimination (Solution B) ────────────────────────
    #   For each unclaimed hint that already has a confirmed identity
    #   in the DB, try to biometrically match it to a remaining bio
    #   cluster.  This resolves "phantom" clusters/hints from runners
    #   identified in other bursts.
    if project_id and bio_clusters:
        global_avail = burst_hints - claimed_hints
        known = db.load_identity_centroids(project_id, global_avail)
        for bib, (face_c, reid_c) in known.items():
            for c in list(bio_clusters):
                if bib not in (c._cluster_hints - claimed_hints):
                    continue
                face_sim = _reid_cosine(c.best_face_vec, face_c)
                reid_sim = _reid_cosine(c.blended_reid_vec, reid_c)
                if (face_sim >= _cfg.CASCADE_FACE_STRICT
                        or (reid_sim >= _cfg.CASCADE_REID_STRONG
                            and face_sim >= _cfg.CASCADE_FACE_SOFT)):
                    c.assign(bib, "deductive_known")
                    claimed_hints.add(bib)
                    unassigned.remove(c)
                    bio_clusters.remove(c)
                    break

    # ── Rules 4 & 5: Per-cluster hint remainder ───────────────────
    #   Iterative: claiming one hint may free another cluster's pool
    #   to a single candidate.
    changed = True
    while changed:
        changed = False
        for c in list(bio_clusters):
            c_avail = c._cluster_hints - claimed_hints
            if len(c_avail) != 1:
                continue
            hint = next(iter(c_avail))
            # If another bio cluster also has this as its sole hint, skip
            if any(x is not c
                   and (x._cluster_hints - claimed_hints) == {hint}
                   for x in bio_clusters):
                continue
            mtype = ("blind_trust"
                     if not claimed_hints and len(burst_hints) == 1
                     else "hint_remainder")
            c.assign(hint, mtype)
            claimed_hints.add(hint)
            unassigned.remove(c)
            bio_clusters.remove(c)
            changed = True
            break  # restart scan after mutation

    # ── Rules 7 & 8: Spectator Veto / Ambiguous Pack → ghost ────
    for c in list(unassigned):
        c.assign(None, "ghost")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Intent Payload Construction
# ═══════════════════════════════════════════════════════════════════

def _build_scribe_task(
    payload: dict,
    clusters: List[IdentityCluster],
    project_id: str,
    is_fl: bool,
    t0: float,
) -> dict:
    """Build the scribe_tasks V3 payload from resolved clusters."""
    burst_id = payload.get("burst_id", "?")

    # Build photo map: original photo_id → (deterministic UUID, file_path)
    photo_uuid_map: Dict[str, Tuple[str, str]] = {}
    for img in payload.get("images", []):
        if not img.get("success", True):
            continue
        file_path = (
            img.get("r2_key") or img.get("filename")
            or img.get("file_path") or img.get("path", "")
        )
        p_uuid = deterministic_photo_uuid(project_id, file_path)
        photo_uuid_map[str(img["photo_id"])] = (p_uuid, file_path)

    # Build photos array
    scribe_photos = []
    for img in payload.get("images", []):
        entry = photo_uuid_map.get(str(img["photo_id"]))
        if not entry:
            continue
        p_uuid, file_path = entry
        scribe_photos.append({
            "uuid": p_uuid,
            "project_id": project_id,
            "photo_id": img.get("photo_id"),
            "file_path": file_path,
            "corrected_time": _parse_corrected_time(img.get("corrected_time")),
            "is_finish_line": is_fl,
        })

    # Build intents array from resolved clusters
    intents = []
    for c in clusters:
        det_list = []
        for d in c.detections:
            entry = photo_uuid_map.get(str(d.photo_id))
            if not entry:
                continue
            p_uuid, _ = entry
            x1, y1, x2, y2 = d.bbox
            w = d.img_width if d.img_width > 0 else 1920
            h = d.img_height if d.img_height > 0 else 1280
            bib, bib_conf = d.best_bib
            crop_area = max(0, x2 - x1) * max(0, y2 - y1)
            img_area = w * h
            area_pct = round(crop_area / img_area, 6) if img_area > 0 else 0.0
            det_list.append({
                "photo_id": p_uuid,
                "bbox": d.bbox,
                "bbox_x": round(x1 / w, 4),
                "bbox_y": round(y1 / h, 4),
                "bbox_w": round((x2 - x1) / w, 4),
                "bbox_h": round((y2 - y1) / h, 4),
                "area_pct": area_pct,
                "confidence": d.confidence,
                "face_quality": d.face_quality,
                "face_enc": d.face_vector_b64,
                "reid_enc": d.reid_vector_b64,
                "ocr_bib": bib,
                "ocr_confidence": bib_conf if bib else None,
            })

        intents.append({
            "cluster_id": c.id,
            "assigned_bib": c.assigned_bib,
            "match_type": c.match_type,
            "consensus_bib": c.consensus_bib,
            "consensus_conf": round(c.consensus_conf, 4),
            "best_face_quality": round(c.best_face_quality, 4),
            "best_face_enc": _encrypt_vec_b64(c.best_face_vec),
            "blended_reid_enc": _encrypt_vec_b64(c.blended_reid_vec),
            "detections": det_list,
        })

    # Build photo status
    photo_status = []
    for img in payload.get("images", []):
        entry = photo_uuid_map.get(str(img["photo_id"]))
        if not entry:
            continue
        p_uuid, _ = entry
        img_pid = str(img["photo_id"])
        photo_status.append({
            "photo_uuid": p_uuid,
            "status": "completed",
            "subject_count": sum(
                1 for c in clusters for d in c.detections
                if str(d.photo_id) == img_pid
            ),
            "matched_count": sum(
                1 for c in clusters for d in c.detections
                if str(d.photo_id) == img_pid and c.assigned_bib is not None
            ),
            "inference_ms": payload.get("inference_ms", 0),
        })

    # Summary stats
    matched = [c for c in clusters if c.assigned_bib]
    ghosts = sum(1 for c in clusters if c.match_type and "ghost" in c.match_type)
    match_types: dict = {}
    for c in clusters:
        if c.match_type:
            match_types[c.match_type] = match_types.get(c.match_type, 0) + 1

    return {
        "schema_version": 3,
        "burst_id": burst_id,
        "project_id": project_id,
        "job_id": payload.get("job_id"),
        "priority": payload.get("priority", 5),
        "photo_ids": payload.get("photo_ids", []),
        "photos": scribe_photos,
        "intents": intents,
        "photo_status": photo_status,
        "summary": {
            "cluster_count": len(clusters),
            "matched_count": len(matched),
            "ghost_count": ghosts,
            "match_types": match_types,
            "inference_ms": payload.get("inference_ms", 0),
            "cluster_ms": round((time.time() - t0) * 1000, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Main Processing Flow
# ═══════════════════════════════════════════════════════════════════

def process_payload(payload: dict) -> bool:
    """Main ID Cluster flow.  Returns True on success."""
    t0 = time.time()
    project_id = str(payload["project_id"])
    burst_id = payload.get("burst_id", "?")
    short_tag = burst_id[-16:] if len(str(burst_id)) > 16 else burst_id
    priority = int(payload.get("priority", 5))
    is_fl = priority == 9

    n_images = len(payload.get("images", []))
    log = logger.bind(
        burst_id=burst_id, project_id=project_id,
        priority=priority, task_type="bib_detection",
    )
    _status(f"[IDC:{short_tag}] Processing {n_images} images | P{priority} "
            f"{'FL' if is_fl else 'COURSE'}")

    # ── Course passthrough (V3 Phase 1: FL only) ─────────────────
    if not is_fl:
        _status(f"[IDC:{short_tag}] Course burst — passthrough (V3 Phase 1)")
        scribe_task = {
            "schema_version": 3,
            "burst_id": burst_id,
            "project_id": project_id,
            "photo_ids": payload.get("photo_ids", []),
            "job_id": payload.get("job_id"),
            "priority": priority,
            "photos": [],
            "intents": [],
            "photo_status": [],
            "summary": {
                "cluster_count": 0, "matched_count": 0, "ghost_count": 0,
                "match_types": {},
                "inference_ms": payload.get("inference_ms", 0),
                "cluster_ms": 0,
            },
        }
        return publish_scribe_task(scribe_task)

    # ── Load reference data (cached per project) ─────────────────
    valid_bibs = _ref_cache.get_valid_bibs(project_id)
    timed_participants = _ref_cache.get_timed_participants(project_id)

    # ── Calculate burst hints ─────────────────────────────────────
    burst_hints = calculate_burst_hints(
        payload.get("images", []), timed_participants, _cfg.HINT_WINDOW_S,
    )
    if burst_hints:
        log.info("timing_hint_pool",
                 hint_count=len(burst_hints), hints=sorted(burst_hints))

    # ── Phase 1: Feature-Space Clustering ─────────────────────────
    clusters = cluster_burst_detections(payload.get("images", []), burst_hints)
    log.info("clusters_built",
        cluster_count=len(clusters),
        with_ocr=sum(1 for c in clusters if c.consensus_bib),
        without_ocr=sum(1 for c in clusters if not c.consensus_bib),
    )
    _status(f"[IDC:{short_tag}] {len(clusters)} clusters built")

    # ── Compute representative burst SOD for Rule 12 time bound ──
    burst_sod = None
    for _img in payload.get("images", []):
        _ct = _parse_corrected_time(_img.get("corrected_time"))
        if _ct is not None:
            burst_sod = _photo_epoch_to_sod(_ct)
            break  # all images in a burst share the same timestamp

    # ── Phase 2: Deterministic Cascade ────────────────────────────
    registered_bibs = _ref_cache.get_registered_bibs(project_id)
    run_cascade(clusters, burst_hints, valid_bibs, timed_participants, burst_sod, registered_bibs, project_id)

    matched = [c for c in clusters if c.assigned_bib]
    ghosts = sum(1 for c in clusters if c.match_type and "ghost" in c.match_type)
    match_types: dict = {}
    for c in clusters:
        if c.match_type:
            match_types[c.match_type] = match_types.get(c.match_type, 0) + 1
    log.info("cascade_complete",
        cluster_count=len(clusters), matched=len(matched),
        ghosts=ghosts, match_types=match_types,
    )

    # ── Phase 3: Build & publish scribe task ──────────────────────
    scribe_task = _build_scribe_task(payload, clusters, project_id, is_fl, t0)
    success = publish_scribe_task(scribe_task)

    if success:
        elapsed = (time.time() - t0) * 1000
        _status(
            f"[IDC:{short_tag}] Done: {len(clusters)} clusters, "
            f"{len(matched)} matched, {ghosts} ghosts | {elapsed:.0f}ms",
            GREEN,
        )
        log_burst_metrics(
            burst_id=burst_id,
            project_id=project_id,
            latency_total_burst_ms=elapsed,
            latency_ocr_pipeline_ms=float(payload.get("inference_ms", 0)),
            batch_size_images=n_images,
            batch_size_persons=sum(
                len(img.get("persons", []))
                for img in payload.get("images", []) if img.get("success")
            ),
            clusters_total=len(clusters),
            clusters_matched=len(matched),
            clusters_ghosts=ghosts,
            match_type_distribution=match_types,
            priority=priority,
            is_finish_line=is_fl,
        )
    else:
        log.error("scribe_publish_failed")

    return success


# ═══════════════════════════════════════════════════════════════════
# RabbitMQ Consumer
# ═══════════════════════════════════════════════════════════════════

def start_worker():
    """Blocking RabbitMQ consume loop with reconnect."""

    def _callback(ch, method, properties, body):
        payload = None
        try:
            payload = json.loads(body)
            success = process_payload(payload)
            if success:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            else:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        except Exception as e:
            logger.exception("id_cluster_fatal", error=str(e))
            log_exception(
                burst_id=payload.get("burst_id") if isinstance(payload, dict) else None,
                project_id=str(payload.get("project_id", "")) if isinstance(payload, dict) else None,
                error_category="id_cluster_fatal",
                error_message=str(e),
                error_type=type(e).__name__,
                task_type="bib_detection",
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _signal_handler(signum, _frame):
        logger.info("shutdown_signal", signal=signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while True:
        try:
            conn = create_local_connection()
            channel = conn.channel()
            channel.queue_declare(queue=QUEUE_IN, durable=True)
            channel.basic_qos(prefetch_count=4)
            channel.basic_consume(queue=QUEUE_IN, on_message_callback=_callback)
            _status(f"ID Cluster Worker listening on '{QUEUE_IN}' (local RabbitMQ)")
            channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("worker_shutting_down")
            sys.exit(0)
        except Exception as e:
            logger.error("connection_error", error=str(e), retry_delay_s=5)
            time.sleep(5)


# ═══════════════════════════════════════════════════════════════════
# Banner & Entry Point
# ═══════════════════════════════════════════════════════════════════

BANNER = """
\033[36m\033[1m
 ═══════════════════════════════════════════════════════════════
   pixxEngine ID Cluster Worker — Tier 2: Identity Clustering
   Pipeline V3 — 13-Rule Deterministic Cascade
 ═══════════════════════════════════════════════════════════════
   Input:   raw_inference_results  (local RabbitMQ)
   Output:  scribe_tasks           (local RabbitMQ)
   DB:      read-only cache        (participant data)
 ═══════════════════════════════════════════════════════════════
\033[0m"""


def main():
    print(BANNER, flush=True)
    logger.info("worker_starting")
    start_worker()


if __name__ == "__main__":
    main()