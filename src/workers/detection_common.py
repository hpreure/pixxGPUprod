"""
Detection Common — Shared Utilities for Detection Pipelines
=============================================================
Provides constants, logging helpers, path resolution, encryption,
subject-row building, and spatial filtering used by both the
GPU worker and CPU worker pipeline stages.

Import from here instead of duplicating across detection handlers.
"""

import logging
import re
import uuid
import time
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.detection_config import settings, detection_settings
from src.workers.inference_engine import (
    InferenceResult, PersonDetection, BatchTimingBreakdown,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

VPS_PATH_PREFIX = "/app/proxies/"
LOCAL_PATH_PREFIX = "/opt/pixxEngine/media/proxies/"


# ═══════════════════════════════════════════════════════════════════
# Terminal Status Logging
# ═══════════════════════════════════════════════════════════════════

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def status(msg: str, color: str = CYAN):
    """Print a colored status line and mirror to the logger."""
    print(f"{color}{BOLD}▶ {msg}{RESET}", flush=True)
    logger.info(msg)


def ok(msg: str):
    status(msg, GREEN)


def warn(msg: str):
    status(msg, YELLOW)


def err(msg: str):
    status(msg, RED)


# ═══════════════════════════════════════════════════════════════════
# Path Resolution
# ═══════════════════════════════════════════════════════════════════

def resolve_path(raw_path: str) -> str:
    """Map VPS/relative paths to local absolute paths."""
    base = str(settings.BASE_PATH)
    if raw_path.startswith(VPS_PATH_PREFIX):
        return LOCAL_PATH_PREFIX + raw_path[len(VPS_PATH_PREFIX):]
    if not raw_path.startswith("/"):
        return f"{base}/{raw_path}"
    return raw_path


# ═══════════════════════════════════════════════════════════════════
# Deterministic Photo UUIDs
# ═══════════════════════════════════════════════════════════════════

def deterministic_photo_uuid(project_id: str, file_path: str) -> str:
    """Generate a deterministic UUIDv5 for a photo.

    Eliminates the N ``upsert_photo`` DB round-trips previously needed
    to look up or create a ``pipeline.photos`` UUID.  The scribe worker
    later reconciles this with any pre-existing uuid4 row in the database.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{project_id}_{file_path}"))


# ═══════════════════════════════════════════════════════════════════
# Encryption
# ═══════════════════════════════════════════════════════════════════

def encrypt_vec(vec) -> Optional[bytes]:
    """Encrypt a numpy vector for DB storage."""
    if vec is None:
        return None
    try:
        from src.encryption import get_encryptor
        return get_encryptor().encrypt_vector(vec)
    except Exception:
        if hasattr(vec, "tobytes"):
            return vec.astype(np.float32).tobytes()
        return None


# ═══════════════════════════════════════════════════════════════════
# Message Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_corrected_time(value) -> Optional[float]:
    """
    Convert corrected_time to epoch seconds (float).
    Accepts:  float/int (epoch seconds), string "YYYY-MM-DD HH:MM:SS", or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


# ═══════════════════════════════════════════════════════════════════
# Spatial Filtering  (shared person crop validation)
# ═══════════════════════════════════════════════════════════════════

# NMS thresholds — two detections of the same person are suppressed
# when their IoU is high OR the smaller box is mostly inside the larger.
# YOLO frequently emits a tight torso box and a full-body box for the
# same runner; IoU alone misses this because the size difference
# dilutes the score, but containment catches it clearly.
NMS_IOU_THRESH         = 0.60   # IoU above this -> same person
NMS_CONTAINMENT_THRESH = 0.65   # smaller-inside-larger above this -> same person


def _box_iou_and_containment(a_bbox, b_bbox):
    """Return (IoU, containment) between two (x1,y1,x2,y2) boxes.

    Containment = intersection / area_of_smaller_box.  A value of 1.0
    means the smaller box is completely inside the larger one.
    """
    ax1, ay1, ax2, ay2 = a_bbox
    bx1, by1, bx2, by2 = b_bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return iou, containment


def suppress_overlapping_persons(persons, iou_thresh=NMS_IOU_THRESH,
                                  contain_thresh=NMS_CONTAINMENT_THRESH):
    """Containment-aware NMS: suppress duplicate person detections.

    Two boxes are considered the same person when:
      * IoU >= iou_thresh, OR
      * containment >= contain_thresh (smaller box nested inside larger).

    The lower-confidence detection is suppressed.  Works on any object
    with ``.bbox`` and ``.confidence`` attributes (PersonDetection, etc.).

    Returns a new list with duplicates removed.
    """
    if len(persons) <= 1:
        return list(persons)

    keep = [True] * len(persons)
    for i in range(len(persons)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(persons)):
            if not keep[j]:
                continue
            iou, containment = _box_iou_and_containment(
                persons[i].bbox, persons[j].bbox
            )
            if iou >= iou_thresh or containment >= contain_thresh:
                # Suppress the lower-confidence duplicate
                if persons[i].confidence >= persons[j].confidence:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is suppressed, stop comparing it
    return [p for p, k in zip(persons, keep) if k]
