"""
pixxEngine Configuration
========================
Centralized configuration for the pixxEngine GPU pipeline.

  Section A - Infrastructure (Settings):  DB, RabbitMQ, paths, models, encryption.
  Section B - Detection Logic (DetectionConfig):  filtering, exclusion zones,
              OCR validation, and identity-resolution thresholds.

Every worker imports from this single file:
    from src.detection_config import settings           # infrastructure
    from src.detection_config import detection_settings  # detection logic
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# A.  INFRASTRUCTURE  SETTINGS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # ==========================================
    # PostgreSQL Database
    # ==========================================
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "192.168.1.100")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "pixxengine")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "pixxadmin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")   

    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ==========================================
    # RabbitMQ Message Broker
    # ==========================================
    RABBITMQ_HOST: str = os.getenv("RABBITMQ_HOST", "192.168.1.100")
    RABBITMQ_PORT: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    RABBITMQ_USER: str = os.getenv("RABBITMQ_USER", "pixxGPU")
    RABBITMQ_PASSWORD: str = os.getenv("RABBITMQ_PASSWORD", "")
    RABBITMQ_VHOST: str = os.getenv("RABBITMQ_VHOST", "pixxengine")

    @property
    def RABBITMQ_URL(self) -> str:
        """Construct RabbitMQ AMQP URL."""
        return (
            f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}"
            f"@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{self.RABBITMQ_VHOST}"
        )

    # ==========================================
    # VPS RabbitMQ (remote, via Tailscale)
    # ==========================================
    VPS_RABBITMQ_HOST: str = os.getenv("VPS_RABBITMQ_HOST", "100.95.216.33")
    VPS_RABBITMQ_PORT: int = int(os.getenv("VPS_RABBITMQ_PORT", "5672"))
    VPS_RABBITMQ_USER: str = os.getenv("VPS_RABBITMQ_USER", "pixxGPU")
    VPS_RABBITMQ_PASSWORD: str = os.getenv("VPS_RABBITMQ_PASSWORD", "")
    VPS_RABBITMQ_VHOST: str = os.getenv("VPS_RABBITMQ_VHOST", "/")

    # ==========================================
    # Application Paths
    # ==========================================
    BASE_PATH: Path = Path(os.getenv("BASE_PATH", "/opt/pixxEngine"))
    MEDIA_PATH: Path = Path(os.getenv("MEDIA_PATH", "/opt/pixxEngine/media"))
    WEIGHTS_PATH: Path = Path(os.getenv("WEIGHTS_PATH", "/opt/pixxEngine/weights"))
    LOGS_PATH: Path = Path(os.getenv("LOGS_PATH", "/opt/pixxEngine/logs"))

    # ==========================================
    # GPU Worker Settings
    # ==========================================
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    INFERENCE_DEVICE: str = os.getenv("INFERENCE_DEVICE", "cuda:0")

    # ==========================================
    # Model Weights Paths
    # ==========================================
    def _get_model_path(self, base_name: str) -> str:
        """Get model path, preferring TensorRT engine if available."""
        engine_weights = self.WEIGHTS_PATH / f"{base_name}.engine"
        pt_weights = self.WEIGHTS_PATH / f"{base_name}.pt"
        engine_root = self.BASE_PATH / f"{base_name}.engine"
        pt_root = self.BASE_PATH / f"{base_name}.pt"

        if engine_weights.exists():
            return str(engine_weights)
        if pt_weights.exists():
            return str(pt_weights)
        if engine_root.exists():
            return str(engine_root)
        if pt_root.exists():
            return str(pt_root)

        return str(pt_weights)

    @property
    def YOLO_PERSON_MODEL(self) -> str:
        return self._get_model_path("yolo26m-seg")

    @property
    def YOLO_BIB_MODEL(self) -> str:
        return self._get_model_path("yolo26_bib_v2")

    @property
    def YOLO_TEXT_MODEL(self) -> str:
        return self._get_model_path("yolo26_numbers_v2")

    @property
    def DBNET_ENGINE(self) -> str:
        """DBNet TRT engine used for text-region detection (replaces YOLO_text)."""
        engine = self.WEIGHTS_PATH / "dbnet_resnet50_256.engine"
        if engine.exists():
            return str(engine)
        raise FileNotFoundError(f"DBNet engine not found: {engine}")

    # ==========================================
    # Encryption (Fernet AES-128)
    # ==========================================
    FERNET_KEY: str = os.getenv("FERNET_KEY", "")

    def validate(self) -> bool:
        """Validate that all required settings are present."""
        required = [
            ("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD),
            ("RABBITMQ_PASSWORD", self.RABBITMQ_PASSWORD),
            ("FERNET_KEY", self.FERNET_KEY),
        ]

        missing = [name for name, value in required if not value]

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        return True


# Global infrastructure singleton
settings = Settings()


# ══════════════════════════════════════════════════════════════════════
# B.  DETECTION  LOGIC  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════


class DetectionConfig:
    # ==========================================
    # 1. HARD FILTERS
    # ==========================================
    # The pipeline uses hard-rejection gates in the inference engine:
    #   F0. Edge-zone mask  — left/right pixel strips blacked out BEFORE
    #       YOLO-seg runs, so no detections can originate there at all.
    #   F1. YOLO confidence — SEG_CONF threshold in the seg detector.
    #   F2. Anchor-relative — ANCHOR_AREA_MIN_PCT, applied after YOLO-seg.
    #   F3. Has face        — faceless persons dropped after InsightFace
    #       extraction, BEFORE ReID / bib / OCR to save GPU work.
    #   F4. Per-image cap   — MAX_PERSONS_PER_IMAGE, top-N by conf × area.

    # YOLO-seg person detection confidence.  Detections below this
    # threshold are discarded before any downstream processing.
    # DB analysis (32 628 subjects): 81.4% of identified subjects
    # score ≥ 0.90, only 2.6% score < 0.50.  0.80 is user-tuned.
    SEG_CONF: float = 0.80

    # Burst-relative size filter: reject any person whose crop area is
    # smaller than this fraction of the largest non-border person seen
    # anywhere in the burst (the "anchor subject").  Eliminates distant
    # spectators who are dwarfed by foreground athletes.  The anchor is
    # computed once per burst (from non-border-touching boxes only) so
    # the threshold stays stable and does not cause cluster fragmentation.
    # History: 0.20 → 0.30 → 0.22 (0.30 too aggressive for finish-line
    # perspective depth where foreground runners dominate at 4-5% area
    # and mid-distance runners sit at 0.5-1.5%).
    ANCHOR_AREA_MIN_PCT: float = 0.22

    # Rescue multiplier for zero-subject images.  When faceless rejection
    # eliminates every person in an image (typically a dominant back-facing
    # runner inflating the anchor), re-scan raw detections with a relaxed
    # anchor of ANCHOR_AREA_MIN_PCT × RESCUE_ANCHOR_FACTOR and keep any
    # that have a detectable face.  Only fires on zero-survivor images so
    # it cannot degrade existing results.
    # Derivation: the two motivating photos (d111/a451) need ≤0.36 to
    # rescue a faced runner at 8-10% of anchor area.  0.25 gives margin.
    RESCUE_ANCHOR_FACTOR: float = 0.25

    # Maximum person detections to keep per image after anchor + face
    # filtering.  Survivors are ranked by confidence × area and the
    # top-N are kept.  Prevents pathological crowd scenes from
    # starving the GPU with 20+ crops through ReID / bib / OCR.
    # Set to 0 to disable.
    MAX_PERSONS_PER_IMAGE: int = 7

    # ==========================================
    # 2. EDGE EXCLUSION ZONES (letterbox mask)
    # ==========================================
    # Fraction of image width to physically black-out on each side
    # BEFORE YOLO-seg detection.  YOLO literally cannot see pixels in
    # these strips, so no person detections can originate there.
    # This eliminates edge spectators, partial-body close-passers,
    # and lens-distortion artifacts without any post-filter leaks.
    # History: 0.15 → 0.08 → removed → restored at 0.10 (Feb 2026).
    ZONE_LEFT_PCT: float = 0.10
    ZONE_RIGHT_PCT: float = 0.10

    # ==========================================
    # 3. SOFT QUALITY MARKERS
    # ==========================================

    # Laplacian variance threshold for blur flagging.  Crops below this
    # value are soft-flagged (PersonDetection.is_blurry = True) and
    # excluded from OCR voting and biometric blending while still
    # contributing to cluster continuity.  Not a hard filter.
    # Sharp race subjects typically score 150+; motion-blurred or
    # defocused close-passers fall below 70.  Set to 0.0 to disable.
    BLUR_THRESHOLD: float = 70.0

    # Minimum face-quality score required to trust biometric vectors
    # (face_vector and reid_vector) from a detection.  Below this
    # threshold, vectors are nulled — the detection keeps its OCR,
    # bbox, and cluster-membership contribution but cannot be used
    # for face/ReID matching.  This prevents noisy embeddings from
    # low-quality crops (distant, blurred, side-profile) from causing
    # false merges during identity clustering.
    # Analysis on project 159: FQ<0.80 is 66% ghosts, 4× the rate of
    # FQ>=0.80 (17%).  Only 9 runners out of 1 810 identities would
    # lose all biometric coverage — all 9 retain their OCR-based bib.
    # Set to 0.0 to disable.
    FQ_MIN_BIOMETRIC: float = 0.70

    # ── Hint Window ──────────────────────────────────────────────
    # Maximum delta (seconds) between a photo's corrected_time and a
    # participant's finish_time to include them as a timing hint.
    # Probe-calibrated photos land within ~1 s of finish_time;
    # ±2 s covers jitter without pulling in adjacent pack finishers.
    HINT_WINDOW_S: float = 2.0

    # Strict window used ONLY for Rule 5 (hint_remainder / deductive
    # remainder).  Tighter than HINT_WINDOW_S to avoid pulling in
    # adjacent-pack finishers whose timing falls between 1-2 s of the
    # photo.  Empirically resolves +54 ghosts vs baseline with only
    # 3 true-cost regressions (all genuinely ambiguous).  Rule 4
    # (blind_trust) keeps the wider HINT_WINDOW_S because its solo-
    # hint constraint already prevents contamination.
    HINT_REMAINDER_STRICT_WINDOW_S: float = 1.0

    # Minimum separation (seconds) between the nearest and second-nearest
    # hint finish times for the proximity tiebreaker in blind trust.
    # When a single-cluster burst has multiple unclaimed hints and no OCR,
    # we pick the nearest hint IFF it is at least this many seconds closer
    # than the runner-up.  At 0.5 s the timing mat resolution is enough to
    # distinguish adjacent finishers in almost all real packs.
    BLIND_TRUST_MIN_GAP_S: float = 0.5

    # ==========================================
    # 4. BIB NUMBER VALIDATION (OCR Hallucination Prevention)
    # ==========================================

    # Minimum OCR confidence to accept a bib reading.
    # Previous: 0.85.  Ground truth showed 354 valid-zone persons had
    # bib=NULL — many were correctly-read bibs rejected by the strict
    # confidence gate.  Lowered to 0.60; the bib range check + identity
    # resolution already guard against hallucinated numbers.
    MIN_OCR_CONF: float = 0.60

    # Minimum number of digits for a bib number to be accepted.
    # Ground truth shows the race uses 2-4 digit bibs (201-1211 main range).
    # Single-digit OCR reads (e.g. "1", "4", "5") are always fragments
    # of partially-occluded multi-digit bibs.  Reject them here to
    # prevent 126 phantom bib identities that pollute the runner space.
    # Analysis: bib "10" appeared 40 times as FP, bib "1" 11 times, etc.
    # Raised from 2→3: 2-char partials like "17" or "16" match too many
    # 4-digit hints via substring, causing wrong bib assignment.
    MIN_BIB_DIGITS: int = 3

    # ==========================================
    # 5. IDENTITY CLUSTERING (Cross-Frame Association)
    # ==========================================
    #
    # Identity clustering merges person crops across burst frames
    # into IdentityClusters using a 5-step greedy algorithm in
    # cluster_burst_detections() (id_cluster.py):
    #
    #   Step 1 — Ambiguous Partial Pre-Filter (multi-hint substring)
    #   Step 2 — Hard Veto (conflicting OCR → skip)
    #   Step 3 — Hint Disambiguation Veto (co-present bibs)
    #   Step 4 — Biometric Gravitation (face ≥ 0.40 AND reid ≥ 0.80)
    #   Step 5 — OCR Anchor fallback (compatible OCR, hostile-bio veto)
    #
    # Rule 13 (Concurrent Frame Veto): crops from the same photo_id
    # can never merge into the same cluster.

    # ── OCR tier ─────────────────────────────────────────────────
    # Minimum OCR confidence for a bib reading to participate in the
    # linking score at all.  Below this, the bib is treated as absent.
    OCR_LOCK_CONF: float = 0.40

    # Score bonus when both detections read the same bib with
    # confidence ≥ OCR_LOCK_CONF.  Set high enough (3.0) to
    # mathematically overwhelm any combination of lower-tier scores,
    # guaranteeing the Hungarian algorithm links these detections.
    OCR_EXACT_BONUS: float = 3.0

    # Score bonus for substring / partial bib match (e.g. "348" and
    # "1348" when a hand covers the leading digit).  Strong but
    # weaker than an exact match.
    OCR_PARTIAL_BONUS: float = 2.0

    # Hard conflict veto: different bibs on different people.
    # Extremely negative value forces a split regardless of any
    # biometric similarity.
    OCR_CONFLICT_VETO: float = -1e9

    # ── Face tier ────────────────────────────────────────────────
    # Cosine similarity threshold for a "strong" face match.
    # Above this, the face is highly likely the same person across
    # frames — awards FACE_STRONG_BONUS.
    FACE_STRONG_SIM: float = 0.60
    FACE_STRONG_BONUS: float = 1.5

    # Moderate face match: same person probable but not certain.
    # Awards FACE_MODERATE_BONUS (< strong, > linear fallback).
    FACE_MODERATE_SIM: float = 0.40
    FACE_MODERATE_BONUS: float = 0.7

    # ── In-Burst ReID Gate ────────────────────────────────────────
    # Minimum ReID cosine similarity for biometric gravitation in
    # in-burst clustering.  Step 4 now requires BOTH face >= 0.40
    # AND reid >= 0.70 to merge — neither alone is sufficient.
    CLUSTER_REID_MIN: float = 0.70

    # ── OCR Anchor Hostile-Biometric Cross-Check ──────────────────
    # When biometric gravitation (step 4) fails but OCR is compatible
    # (step 5), block the OCR merge if EITHER face OR reid is below
    # these floors — indicating different people sharing a partial
    # bib reading (e.g. "368" matching 3681 via subsequence).
    # The OR logic ensures one weak biometric is enough to veto;
    # legitimate same-person merges pass Step 4's dual-gate instead.
    OCR_ANCHOR_HOSTILE_FACE: float = 0.30
    OCR_ANCHOR_HOSTILE_REID: float = 0.65

    # Below FACE_CONFLICT_SIM the faces are clearly different people.
    # Applies FACE_CONFLICT_PENALTY to discourage linking.
    FACE_CONFLICT_SIM: float = 0.20
    FACE_CONFLICT_PENALTY: float = -1.0

    # ==========================================
    # 6. CASCADE MATCH (Global Biometric Fallback)
    # ==========================================
    #
    # When hint-guided and OCR-based steps fail to identify a cluster,
    # the cascade match scans ALL enrolled identities in the project
    # and accepts the first candidate exceeding one of these threshold
    # paths (evaluated in order):
    #
    #   1. face_strict    — face alone is decisive
    #   2. reid_soft_face — strong ReID confirmed by moderate face
    #   3. solo_face      — only face available, must be strong
    #   4. solo_reid      — only ReID available, must be very strong
    #
    # These are deliberately tight to prevent false positives in the
    # unguided global search.  The identity_db.py and
    # identity_matcher.py modules share these same thresholds.

    CASCADE_FACE_STRICT: float = 0.75
    CASCADE_REID_STRONG: float = 0.85
    CASCADE_FACE_SOFT: float = 0.60
    CASCADE_SOLO_FACE: float = 0.60
    CASCADE_SOLO_REID: float = 0.88

    # Path 3: strong ReID confirmed by moderate face — catches
    # ghosts where face-strict fails due to cross-angle variance.
    CASCADE_REID_SOLO: float = 0.88
    CASCADE_REID_SOLO_FACE: float = 0.60

    # Hint-biometric guardrails — require BOTH face AND reid to agree
    # before attributing a bibless cluster to a timing hint.  The old
    # OR gate let a single weak signal (face 0.42 alone) assign the
    # wrong runner when multiple finishers cross in the same 2-second
    # window.  AND + best-hint-wins eliminates first-match ordering
    # bias and forces both modalities to confirm.
    HINT_BIO_MIN_FACE: float = 0.50
    HINT_BIO_MIN_REID: float = 0.75

    # FL Step 5 guardrails — prevent the global cascade from acting as
    # a vacuum cleaner that absorbs spectators / distant runners.
    # Gate A: reject cascade hits whose identity's finish_time is more
    #         than this many seconds from the photo's corrected_time.
    FL_CASCADE_TIMING_GATE_S: float = 10.0
    # Gate B: skip cascade entirely when the cluster's best face
    #         quality is below this floor (low-quality = unreliable).
    FL_CASCADE_MIN_FACE_QUALITY: float = 0.80

    # Gate C: use bipartite graph matching (Hungarian algorithm) instead
    # of the greedy sequential cascade for finish-line identity resolution.
    # When True, all cluster-to-hint assignments are optimised globally
    # via a cost matrix, eliminating first-claim ordering bias.
    FL_USE_BIPARTITE: bool = True

    # ── Enrollment Similarity Gate ─────────────────────────────────
    # Minimum face cosine similarity between an incoming vector and the
    # existing centroid before the blend is allowed.  Only applies when
    # the identity already has ≥ 2 sightings (centroid is stable).
    # Prevents a wrong person's vectors from poisoning the centroid
    # (e.g. bib 3333 contaminated by a golden_delayed photo of a
    # different runner 34 minutes before the real finisher).
    # 0.30 is deliberately low — only blocks clearly different people.
    ENROLLMENT_MIN_SIM: float = 0.40

    # ── Golden-Delayed Time Bound ────────────────────────────────
    # Maximum allowed delta (seconds) between a burst's photo time and
    # the participant's finish time for Rule 12 (golden_delayed) to
    # fire.  Without this limit, any photo captured hours before or
    # after a finisher's crossing can claim that bib based on OCR alone.
    # 120 s (2 min) is generous enough for finish-line loitering but
    # tight enough to reject photos from mid-course (e.g. 34 min gap).
    DELAYED_MAX_DELTA_S: float = 20.0

    # ── Bipartite Cost-Matrix Weights ──────────────────────────────
    # Tuneable without touching worker source.  Changing W_TIME is the
    # most common need (e.g. when the finish-line clock drifts).
    BIPARTITE_W_OCR:  float = 1.0   # OCR agreement weight
    BIPARTITE_W_BIO:  float = 0.8   # biometric distance weight
    BIPARTITE_W_TIME: float = 0.5   # timing proximity weight
    BIPARTITE_MAX_COST: float = 1.8 # reject threshold (above → unassigned)

    # ==========================================
    # 7. OCR ERROR MAP (Known Misread Patterns)
    # ==========================================
    #
    # Constrained lookup table for known OCR hallucinations specific to
    # this race's bib font/number set.  When the consensus OCR string
    # matches a key, the candidates are checked with biometric
    # disambiguation (face + ReID) to pick the correct bib.
    #
    # Format:  { "misread_string": ["candidate_bib_1", ...] }
    #
    # Only add entries with empirical evidence from race-day data.
    # General OCR correction (Hamming-1, substring) is already handled
    # by bib_is_compatible() — this map is for stubborn misreads that
    # don't fit those patterns.
    OCR_ERROR_MAP: Dict[str, List[str]] = None  # type: ignore[assignment]

    def __init__(self):
        # Mutable default — must be set in __init__ to avoid shared state
        if self.OCR_ERROR_MAP is None:
            self.OCR_ERROR_MAP = {
                "39": ["394", "139"],
                "12": ["72", "02"],
                "185": ["785", "1B5"],
            }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_bib(
        bib_number: Optional[str],
        ranges: Optional[List[Union[dict, tuple]]] = None,
    ) -> bool:
        """
        Validate whether *bib_number* falls within allowed race ranges.

        Args:
            bib_number: Raw string from OCR (e.g. "194").
            ranges:     List of range dicts ``{'min': 1, 'max': 999}``
                        **or** tuples ``(1, 999)``.  ``None`` → accept all.

        Returns:
            ``True`` if the bib is numeric, meets minimum digit length,
            and is inside at least one range (or no ranges are defined).
        """
        if not bib_number or not bib_number.isdigit():
            return False

        # Reject OCR fragments: single-digit reads are always partial
        # occlusions of real multi-digit bibs (e.g. "1" from "1129").
        if len(bib_number) < DetectionConfig.MIN_BIB_DIGITS:
            return False

        val = int(bib_number)

        if not ranges:
            return True

        for r in ranges:
            if isinstance(r, dict):
                r_min = r.get('min', 0)
                r_max = r.get('max', 0)
            else:
                # tuple / list
                r_min, r_max = r[0], r[1]

            if r_min <= val <= r_max:
                return True

        return False

    @staticmethod
    def bib_is_compatible(bib_a: Optional[str], bib_b: Optional[str]) -> bool:
        """
        Check whether two bib strings are "compatible" — meaning one
        could be an occluded, extended, or misread version of the other.

        Covered OCR failure modes:
          Leading occlusion      416 → 16   (hand covers first digit(s))
          Trailing occlusion     416 → 41   (hand covers last digit(s))
          Extra char prepend     416 → 7416 (OCR hallucinates a leading digit)
          Extra char append      416 → 4167 (OCR hallucinates a trailing digit)
          Single-char misread    416 → 476  (OCR confuses one digit, e.g. 1→7)

        The first four are handled by suffix/prefix checks.
        The fifth is handled by a Hamming-distance-1 check on same-length bibs.

        Safety guards:
          - Either bib is None / empty → not compatible (no info)
          - Identical → compatible (trivially)
          - Length difference must be 0–2 digits
          - For different-length: short bib must be ≥ 2 digits
          - For same-length: exactly 1 character may differ
          - Bibs must be ≥ 2 digits (prevents '4' matching '416')
        """
        if not bib_a or not bib_b:
            return False
        if bib_a == bib_b:
            return True

        len_a, len_b = len(bib_a), len(bib_b)
        len_diff = abs(len_a - len_b)

        # ── Same-length: single-character substitution (Hamming = 1) ──
        # e.g. 416 → 476, 218 → 228
        if len_diff == 0:
            if len_a < 2:
                return False  # single-digit bibs are too ambiguous
            mismatches = sum(1 for ca, cb in zip(bib_a, bib_b) if ca != cb)
            return mismatches == 1

        # ── Different-length: occlusion / extra-character ──
        if len_diff > 2:
            return False  # more than 2-digit difference is too loose

        long_b = bib_a if len_a >= len_b else bib_b
        short_b = bib_b if len_a >= len_b else bib_a

        # Short bib must be at least 2 digits (prevents '4' matching '416')
        if len(short_b) < 2:
            return False

        return short_b in long_b


# Global singleton — import this in workers
detection_settings = DetectionConfig()
