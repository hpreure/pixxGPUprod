"""
Asymmetric GPU Worker — Raw Inference Pipeline
================================================
Tier 1 of the pixxEngine Asymmetric Detection Pipeline.

This worker is strictly responsible for AI tensor math and nothing else.
It consumes *tickets* from the local ``gpu_tasks`` queue (published by
the Image Feeder), reads pre-fetched images from ``/dev/shm``, runs
GPU inference (YOLO, PARSeq, TransReID, InsightFace), publishes raw
results, and **deletes the images from RAM** when done.

Claim-Check Architecture::

    Image Feeder  ──►  gpu_tasks (local)  ──►  GPU Worker
                        (ticket with            │
                         /dev/shm paths)        ├──► raw_inference_results  (bib detection)
                                                ├──► scribe_tasks           (calibration)
                                                └──► rm /dev/shm/…

Strict Separation of Concerns
------------------------------
 GPU Worker (this file):
   - Consumes tickets from the **local** ``gpu_tasks`` queue.
   - Reads images from ``/dev/shm`` (RAM) — zero network I/O.
   - Runs YOLO person detection, bib/text YOLO, PARSeq OCR, TransReID ReID,
     and InsightFace facial recognition (profile-dependent).
   - Handles both ``bib_detection`` and ``probe_calibration`` task types.
   - Fernet-encrypts biometric vectors for safe queue transport.
   - Publishes ``raw_inference_results`` for bib-detection bursts (→ CPU worker).
   - Publishes ``scribe_tasks`` for calibration results (→ DB scribe, bypasses CPU).
   - **Deletes images from /dev/shm** after inference (claim-check cleanup).
   - NEVER performs identity clustering, IoU matching, resolves identity, or
     touches PostgreSQL.

 CPU Worker (``cpu_worker.py``):
   - Consumes ``raw_inference_results``.
   - Runs identity clustering, consensus OCR, and identity resolution.
   - Publishes to ``scribe_tasks`` for async DB writes.

Priority Rules
--------------
 ``priority == 9``  →  Finish-line burst → ``PROFILE_FULL``
   Pipeline: YOLO26 Person → YOLO26 Bib → YOLO26 Numbers → PARSeq OCR
             → TransReID ReID → InsightFace Face

 ``priority == 10`` →  Calibration shot → routed to ``scribe_tasks`` (bypass CPU).

 ``priority < 9``   →  Course-side burst → ``PROFILE_PROBE``
   Pipeline: YOLO26 Person → YOLO26 Bib → YOLO26 Numbers → PARSeq OCR

``raw_inference_results`` Payload Schema
-----------------------------------------
::

    {
      "task_type"          : "raw_inference_results",
      "schema_version"     : 1,

      "project_id"         : str,
      "job_id"             : str | null,
      "burst_id"           : str,
      "photo_ids"          : [int, ...],

      "priority"           : int,
      "camera_id"          : str | null,
      "camera_serial"      : str | null,
      "analysis_mode"      : "smart" | "standard",
      "inference_profile"  : "full" | "probe",

      "submitted_at"       : float,
      "processed_at"       : float,
      "inference_ms"       : float,

      "timing": {
          "load_ms"  : float,
          "det_ms"   : float,
          "reid_ms"  : float,
          "face_ms"  : float,
          "bib_ms"   : float
      },

      "images": [
        {
          "photo_id"       : str,
          "path"           : str,
          "burst_seq"      : int,
          "corrected_time" : float | null,
          "hints"          : [str, ...],
          "img_width"      : int,
          "img_height"     : int,
          "inference_ms"   : float,
          "success"        : bool,
          "error"          : str | null,
          "persons": [
            {
              "bbox"            : [x1, y1, x2, y2],
              "confidence"      : float,
              "blur_score"      : float,
              "is_blurry"       : bool,
              "face_quality"    : float,
              "reid_vector_b64" : str | null,
              "face_vector_b64" : str | null,
              "bib_number"      : str | null,
              "ocr_confidence"  : float | null,
              "bibs": [
                {
                  "bib_number"     : str,
                  "ocr_confidence" : float,
                  "bbox"           : [x1, y1, x2, y2]
                }
              ]
            }
          ]
        }
      ]
    }

Launch::

    source pixxEngine_venv/bin/activate
    python -m src.workers.asymmetric_gpu_worker
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

from src.detection_config import settings, detection_settings
from src.encryption import get_encryptor
from src.workers.inference_engine import (
    InferenceEngine, InferenceResult, PersonDetection,
    PROFILE_FULL, PROFILE_PROBE,
)
from src.workers.probe_calibration import run_probe_calibration
from src.metrics.log_config import configure as _configure_structlog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_configure_structlog()
logger = structlog.get_logger("gpu_worker")


# ──────────────────────────────────────────────────────────────────────────
# Queue names
# ──────────────────────────────────────────────────────────────────────────

QUEUE_GPU_TASKS          = "gpu_tasks"              # inbound from Image Feeder (local)
QUEUE_RAW_INFERENCE      = "raw_inference_results"  # outbound to CPU worker (local)
QUEUE_SCRIBE_TASKS       = "scribe_tasks"            # outbound calibration → DB scribe (local)

# Priority threshold above which we run the full biometric pipeline
FINISH_LINE_PRIORITY = 9

# Payload schema version — increment when the shape changes
SCHEMA_VERSION = 1

from src.workers.detection_common import (
    status as _status, CYAN, GREEN, YELLOW, BOLD, RESET,
)


# ──────────────────────────────────────────────────────────────────────────
# Biometric vector encryption
# ──────────────────────────────────────────────────────────────────────────

def _encrypt_biometric_vector(vec: Optional[np.ndarray]) -> Optional[str]:
    """
    Encrypt a biometric vector (ReID or face) for safe queue transport.

    Uses Fernet (AES-128-CBC + HMAC-SHA256) so the CPU worker can verify
    the ciphertext has not been tampered with before writing to the DB.
    Returns ``None`` when *vec* is ``None`` (probe-profile persons).

    The CPU worker decrypts with::

        vec = get_encryptor().decrypt_vector_b64(b64, shape=(dim,))
    """
    if vec is None:
        return None
    return get_encryptor().encrypt_vector_b64(vec)


def serialize_vector(vec: Optional[np.ndarray]) -> Optional[str]:
    """
    Encode a NumPy float32 array as a base64 string for JSON transport.
    Returns ``None`` when *vec* is ``None``.
    """
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def deserialize_vector(b64: Optional[str], shape: Optional[tuple] = None) -> Optional[np.ndarray]:
    """
    Inverse of :func:`serialize_vector`.  Reconstructs the float32 array.

    Args:
        b64:   base64-encoded string produced by :func:`serialize_vector`.
        shape: optional tuple to reshape the flat vector (e.g. ``(768,)``).

    Returns:
        NumPy float32 array, or ``None`` when *b64* is ``None``.
    """
    if b64 is None:
        return None
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


# ──────────────────────────────────────────────────────────────────────────
# Payload assembly
# ──────────────────────────────────────────────────────────────────────────

def _serialise_person(person: PersonDetection) -> dict:
    """Convert a :class:`PersonDetection` into a JSON-safe dict."""
    best_bib = None
    best_conf: Optional[float] = None
    if person.bibs:
        best = max(person.bibs, key=lambda b: b.confidence)
        best_bib = best.bib_number
        best_conf = round(best.confidence, 4)

    return {
        "bbox":             list(person.bbox),
        "confidence":       round(float(person.confidence), 4),
        "blur_score":       round(float(person.blur_score), 2),
        "is_blurry":        bool(person.is_blurry),
        "face_quality":     round(float(person.face_quality), 4),
        "reid_vector_b64":  _encrypt_biometric_vector(person.reid_vector),
        "face_vector_b64":  _encrypt_biometric_vector(person.face_vector),
        "bib_number":       best_bib,
        "ocr_confidence":   best_conf,
        "bibs": [
            {
                "bib_number":     b.bib_number,
                "ocr_confidence": round(float(b.confidence), 4),
                "bbox":           list(b.bbox),
            }
            for b in (person.bibs or [])
        ],
    }


def build_raw_inference_payload(
    message:         Dict,
    results:         List[InferenceResult],
    inference_profile: str,
    processed_at:    float,
    inference_ms:    float,
    timing:          Optional[Dict] = None,
) -> Dict:
    """
    Assemble the ``raw_inference_results`` payload from a burst message
    and the corresponding list of :class:`InferenceResult` objects.
    """
    images_meta: List[Dict] = message.get("images", [])

    if len(images_meta) != len(results):
        raise ValueError(
            f"build_raw_inference_payload: images_meta length ({len(images_meta)}) "
            f"does not match results length ({len(results)})"
        )

    image_payloads = []
    for img_meta, result in zip(images_meta, results):
        persons_payload = []
        if result.success:
            for person in result.persons:
                persons_payload.append(_serialise_person(person))

        image_payloads.append({
            "photo_id":       str(img_meta.get("photo_id", "")),
            "path":           img_meta.get("path", ""),
            "r2_key":         img_meta.get("r2_key", ""),
            "burst_seq":      int(img_meta.get("burst_seq", 0)),
            "corrected_time": img_meta.get("corrected_time"),
            "hints":          img_meta.get("hints", []),
            "img_width":      result.img_width,
            "img_height":     result.img_height,
            "inference_ms":   round(result.inference_time_ms, 2),
            "success":        result.success,
            "error":          result.error,
            "persons":        persons_payload,
        })

    payload = {
        "task_type":        "raw_inference_results",
        "schema_version":   SCHEMA_VERSION,

        # ── Identifiers ──────────────────────────────────────────────
        "project_id":       str(message.get("project_id", "")),
        "job_id":           message.get("job_id"),
        "burst_id":         message.get("burst_id"),
        "photo_ids":        message.get("photo_ids", []),

        # ── Context ───────────────────────────────────────────────────
        "priority":         int(message.get("priority", 5)),
        "camera_id":        message.get("camera_id"),
        "camera_serial":    message.get("camera_serial"),
        "analysis_mode":    message.get("analysis_mode", "smart"),
        "inference_profile": inference_profile,

        # ── Timing ────────────────────────────────────────────────────
        "submitted_at":     message.get("submitted_at"),
        "processed_at":     round(processed_at, 3),
        "inference_ms":     round(inference_ms, 2),

        "timing": timing or {},

        # ── Per-image results ─────────────────────────────────────────
        "images":           image_payloads,
    }

    return payload


# ──────────────────────────────────────────────────────────────────────────
# Core processing function (engine-agnostic, testable)
# ──────────────────────────────────────────────────────────────────────────

def process_burst(
    message: Dict,
    engine: InferenceEngine,
) -> Dict:
    """
    Run inference on a burst message and return the serialised payload.

    This function contains all business logic and is fully testable without
    a RabbitMQ connection — pass a mock message dict and an engine instance.

    Priority logic:
      Both finish-line (priority 9) and course-side (priority < 9)
      use ``PROFILE_FULL`` so that all models (YOLO, TransReID ReID,
      InsightFace, PARSeq OCR) run on every bib_detection burst.
      ``PROFILE_PROBE`` is reserved for ``probe_calibration`` tasks.

    Args:
        message: Burst payload dict (see module docstring for schema).
        engine:  Pre-loaded :class:`InferenceEngine` instance.

    Returns:
        JSON-serialisable ``raw_inference_results`` payload dict.
    """
    priority = int(message.get("priority", 5))
    is_finish_line = priority == FINISH_LINE_PRIORITY

    required_profile = PROFILE_FULL
    engine.load_models(profile=required_profile)

    log = logger.bind(
        burst_id=message.get("burst_id"),
        project_id=str(message.get("project_id", "")),
        priority=priority,
        task_type="bib_detection",
        analysis_mode=message.get("analysis_mode", "smart"),
    )
    log.info("burst_start",
        profile=required_profile,
        image_count=len(message.get("images", [])),
    )

    # ── Image paths (pre-fetched to /dev/shm by Image Feeder) ────
    images_meta = message.get("images", [])
    photo_paths: List[str] = []
    fetch_failed: set = set()          # indices of images that never downloaded
    for i, img in enumerate(images_meta):
        if img.get("_fetch_failed"):
            fetch_failed.add(i)
        else:
            photo_paths.append(img["path"])

    _fail_result = lambda path: InferenceResult(
        photo_path=path, persons=[], inference_time_ms=0,
        success=False, error="Image fetch failed (R2 download)",
    )

    if not photo_paths:
        log.warning("burst_empty_no_images")
        return build_raw_inference_payload(
            message=message,
            results=[_fail_result(img.get("path", "")) for img in images_meta],
            inference_profile=required_profile,
            processed_at=time.time(),
            inference_ms=0.0,
        )

    # ── GPU inference ─────────────────────────────────────────────
    t0 = time.perf_counter()
    engine_results = engine.process_photos(photo_paths)
    inference_ms = (time.perf_counter() - t0) * 1000
    processed_at = time.time()

    # Merge engine results with fetch-failed placeholders so the
    # results list is 1:1 with images_meta.
    if fetch_failed:
        results: List[InferenceResult] = []
        engine_iter = iter(engine_results)
        for i, img in enumerate(images_meta):
            if i in fetch_failed:
                results.append(_fail_result(img.get("path", "")))
            else:
                results.append(next(engine_iter))
    else:
        results = engine_results

    # ── Expose timing breakdown ───────────────────────────────────
    td = engine.last_timing
    timing = {
        "load_ms": round(td.load_ms, 2),
        "det_ms":  round(td.det_ms, 2),
        "reid_ms": round(td.reid_ms, 2),
        "face_ms": round(td.face_ms, 2),
        "bib_ms":  round(td.bib_ms, 2),
    }

    success_count = sum(1 for r in results if r.success)
    person_count  = sum(len(r.persons) for r in results if r.success)
    _status(
        f"[AsymGPU] done — {success_count}/{len(results)} images OK, "
        f"{person_count} persons, {inference_ms:.0f}ms"
    )

    # ── Structured telemetry events ───────────────────────────────
    log.info("inference_timing",
        load_ms=timing["load_ms"], det_ms=timing["det_ms"],
        reid_ms=timing["reid_ms"], face_ms=timing["face_ms"],
        bib_ms=timing["bib_ms"], total_ms=round(inference_ms, 2),
    )

    total_bib_boxes = 0
    blurry_crops_dropped = 0
    faceless_crops_dropped = 0
    person_bboxes = []
    bib_bboxes = []
    for r in results:
        if not r.success:
            continue
        for p in r.persons:
            if p.blur_score < 70.0:
                blurry_crops_dropped += 1
            if p.face_quality < 0.80:
                faceless_crops_dropped += 1
            person_bboxes.append(list(p.bbox))
            for b in (p.bibs or []):
                total_bib_boxes += 1
                bib_bboxes.append(list(b.bbox))

    log.info("inference_counts",
        total_persons=person_count, total_bib_boxes=total_bib_boxes,
    )
    log.info("inference_quality_drops",
        blurry_crops_dropped=blurry_crops_dropped,
        faceless_crops_dropped=faceless_crops_dropped,
    )
    log.debug("bbox_coords",
        person_bboxes=person_bboxes, bib_bboxes=bib_bboxes,
    )

    payload = build_raw_inference_payload(
        message=message,
        results=results,
        inference_profile=required_profile,
        processed_at=processed_at,
        inference_ms=inference_ms,
        timing=timing,
    )
    return payload


# ──────────────────────────────────────────────────────────────────────────
# Probe Calibration Handler
# ──────────────────────────────────────────────────────────────────────────

def _handle_probe_calibration(message: Dict) -> Dict:
    """
    Handle probe_calibration task.

    Photos have already been downloaded to /dev/shm by the Image Feeder.
    Groups by camera_serial and runs probe calibration for each.
    Returns the calibration result dict.
    """
    project_id = str(message['project_id'])
    payload = message.get('payload', {})
    photos = payload.get('photos', [])
    if not photos:
        logger.warning("calibration_no_photos")
        return {
            "task_type": "probe_calibration_result",
            "project_id": int(project_id),
            "status": "failed",
            "error": "No photos provided",
        }

    # Filter out fetch failures
    valid_photos = [p for p in photos if not p.get('_fetch_failed')]
    if not valid_photos:
        return {
            "task_type": "probe_calibration_result",
            "project_id": int(project_id),
            "status": "failed",
            "error": "All photo downloads failed",
        }

    # Group photos by camera
    cameras: Dict[str, list] = {}
    for photo in valid_photos:
        camera_serial = photo.get('camera_serial', 'unknown')
        cameras.setdefault(camera_serial, []).append(photo)

    results = []
    for camera_serial, camera_photos in cameras.items():
        logger.info("calibration_camera_processing",
            camera_serial=camera_serial, photo_count=len(camera_photos),
        )
        result = run_probe_calibration(
            project_id=project_id,
            photos=camera_photos,
            camera_serial=camera_serial,
        )
        results.append(result)

    return results[0] if results else {
        "task_type": "probe_calibration_result",
        "project_id": int(project_id),
        "status": "failed",
        "error": "No results",
    }


# ──────────────────────────────────────────────────────────────────────────
# Claim-Check Cleanup — delete images from /dev/shm after processing
# ──────────────────────────────────────────────────────────────────────────

def _cleanup_shm(message: Dict) -> int:
    """
    Delete the claim-check directory from /dev/shm after inference.

    Reads ``_claim_check.shm_dir`` from the message.  If absent (legacy
    message), does nothing.  Returns the number of files removed.
    """
    claim = message.get("_claim_check", {})
    shm_dir_str = claim.get("shm_dir")
    if not shm_dir_str:
        return 0
    shm_dir = Path(shm_dir_str)
    if not shm_dir.exists():
        return 0
    try:
        count = sum(1 for f in shm_dir.iterdir() if f.is_file())
        shutil.rmtree(shm_dir, ignore_errors=True)
        logger.debug("shm_cleanup_done", path=str(shm_dir), files=count)
        # Also clean up empty parent (project dir) if possible
        parent = shm_dir.parent
        if parent.exists() and parent != Path("/dev/shm/pixxengine"):
            try:
                parent.rmdir()  # only succeeds if empty
            except OSError:
                pass
        return count
    except Exception as exc:
        logger.warning("shm_cleanup_failed", path=str(shm_dir), error=str(exc))
        return 0


# ──────────────────────────────────────────────────────────────────────────
# RabbitMQ Worker Loop
# ──────────────────────────────────────────────────────────────────────────

def run_worker(device: str = "cuda:0") -> None:
    """
    Start the asymmetric GPU worker.

    Consumes tickets from the **local** ``gpu_tasks`` queue (published
    by the Image Feeder), runs inference on images pre-fetched to
    ``/dev/shm``, publishes results, and deletes the images from RAM.

    Routing:
      - ``bib_detection``      → ``raw_inference_results`` (CPU worker refines)
      - ``probe_calibration``  → ``scribe_tasks`` (final result, bypass CPU)
    """
    import pika as _pika
    from src.messaging import create_mq_client, create_local_connection
    from src.metrics.burst_logger import log_exception

    engine = InferenceEngine(device=device)
    engine.load_models(profile=PROFILE_PROBE)

    # ── Local broker connections ──────────────────────────────────
    local_mq = create_mq_client()
    local_mq.connect()
    local_mq.declare_queue(QUEUE_RAW_INFERENCE)
    local_mq.declare_queue(QUEUE_SCRIBE_TASKS)

    def _make_local_connection() -> _pika.BlockingConnection:
        return create_local_connection()

    def _handle_message(ch, method, properties, body):
        delivery_tag = method.delivery_tag
        try:
            message = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.error("invalid_json", error=str(exc))
            ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
            return

        task_type = message.get("task_type", "bib_detection")
        claim = message.get("_claim_check", {})
        shm_dir = claim.get("shm_dir", "?")
        log = logger.bind(
            burst_id=message.get("burst_id"),
            project_id=str(message.get("project_id", "")),
            task_type=task_type,
            priority=message.get("priority"),
        )

        # ── Probe Calibration (needs GPU for OCR) ─────────────────
        if task_type == "probe_calibration":
            cal_requeued = False
            try:
                _status(
                    f"[AsymGPU] probe_calibration — "
                    f"{len(message.get('payload', {}).get('photos', []))} photos "
                    f"from {shm_dir}"
                )
                result = _handle_probe_calibration(message)

                # Publish calibration result directly to scribe_tasks
                # (bypasses CPU worker — calibration results are final,
                #  no raw inference data to refine)
                published = local_mq.publish_json(QUEUE_SCRIBE_TASKS, result)
                if published:
                    ch.basic_ack(delivery_tag=delivery_tag)
                    status = result.get("status", "?")
                    offset = result.get("offset_seconds", "?")
                    _status(
                        f"[AsymGPU] Calibration {status}: offset={offset}s",
                        GREEN if status == "completed" else YELLOW,
                    )
                else:
                    log.error("calibration_publish_failed")
                    ch.basic_nack(delivery_tag=delivery_tag, requeue=True)
                    cal_requeued = True

            except Exception as exc:
                log.exception("calibration_failed", error=str(exc))
                ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
            finally:
                if not cal_requeued:
                    _cleanup_shm(message)
            return

        # ── Bib Detection (primary path) ──────────────────────────
        requeued = False
        try:
            payload = process_burst(message, engine)
            published = local_mq.publish_json(QUEUE_RAW_INFERENCE, payload)
            if published:
                ch.basic_ack(delivery_tag=delivery_tag)
                log.info("burst_published",
                    queue=QUEUE_RAW_INFERENCE,
                )
            else:
                log.error("publish_failed_requeue")
                ch.basic_nack(delivery_tag=delivery_tag, requeue=True)
                requeued = True

        except Exception as exc:
            log.exception("burst_fatal_error", error=str(exc))
            log_exception(
                burst_id=message.get("burst_id") if isinstance(message, dict) else None,
                project_id=str(message.get("project_id", "")) if isinstance(message, dict) else None,
                error_category="gpu_worker_fatal",
                error_message=str(exc),
                error_type=type(exc).__name__,
                task_type=message.get("task_type", "bib_detection") if isinstance(message, dict) else "unknown",
            )
            # Requeue so the message can be retried, but keep SHM
            # intact — deleting it here caused an infinite failure
            # loop where the retried message found no images on disk.
            ch.basic_nack(delivery_tag=delivery_tag, requeue=True)
            requeued = True
        finally:
            if not requeued:
                _cleanup_shm(message)

    # ── Reconnect loop with exponential backoff ───────────────────
    _RECONNECT_DELAYS = [5, 10, 30, 60, 120]
    attempt = 0
    while True:
        try:
            conn = _make_local_connection()
            ch = conn.channel()
            ch.basic_qos(prefetch_count=1)
            ch.queue_declare(queue=QUEUE_GPU_TASKS, durable=True)
            ch.basic_consume(
                queue=QUEUE_GPU_TASKS,
                on_message_callback=_handle_message,
            )
            _status(
                f"[AsymGPU] Worker started on {device}. "
                f"Consuming {QUEUE_GPU_TASKS!r} → bib→{QUEUE_RAW_INFERENCE!r} | cal→{QUEUE_SCRIBE_TASKS!r}"
            )
            attempt = 0
            ch.start_consuming()

        except Exception as exc:
            import pika as _pika_check
            if isinstance(exc, (
                _pika_check.exceptions.AMQPConnectionError,
                _pika_check.exceptions.StreamLostError,
                _pika_check.exceptions.ChannelClosedByBroker,
            )):
                delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
                logger.warning("connection_lost",
                    error_type=exc.__class__.__name__, error=str(exc),
                    retry_delay_s=delay, attempt=attempt + 1,
                )
                attempt += 1
                time.sleep(delay)
            elif isinstance(exc, KeyboardInterrupt):
                logger.info("worker_shutting_down")
                break
            else:
                delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
                logger.exception("unexpected_error",
                    error=str(exc), retry_delay_s=delay,
                )
                attempt += 1
                time.sleep(delay)


# ──────────────────────────────────────────────────────────────────────────
# Payload validation helper
# ──────────────────────────────────────────────────────────────────────────

class PayloadValidationError(ValueError):
    """Raised when a ``raw_inference_results`` payload fails validation."""


def validate_payload(payload: Dict) -> None:
    """
    Validate the structure and content of a ``raw_inference_results`` payload.

    Raises :class:`PayloadValidationError` on the first violation found.
    """
    required_top = [
        "task_type", "schema_version", "project_id", "burst_id",
        "priority", "inference_profile", "processed_at",
        "inference_ms", "images",
    ]
    for field in required_top:
        if field not in payload:
            raise PayloadValidationError(f"Missing top-level field: {field!r}")

    if payload["task_type"] != "raw_inference_results":
        raise PayloadValidationError(
            f"Unexpected task_type: {payload['task_type']!r}"
        )

    if payload["inference_profile"] not in (PROFILE_FULL, PROFILE_PROBE):
        raise PayloadValidationError(
            f"Unknown inference_profile: {payload['inference_profile']!r}"
        )

    images = payload["images"]
    if not isinstance(images, list):
        raise PayloadValidationError("'images' must be a list")
    if len(images) == 0:
        raise PayloadValidationError("'images' must be a non-empty list")

    required_img = [
        "photo_id", "path", "burst_seq", "img_width",
        "img_height", "success", "persons",
    ]
    required_person = [
        "bbox", "confidence", "blur_score", "is_blurry",
        "face_quality", "reid_vector_b64", "face_vector_b64",
        "bib_number", "ocr_confidence", "bibs",
    ]

    has_full_profile = payload["inference_profile"] == PROFILE_FULL
    finish_line = int(payload.get("priority", 0)) == FINISH_LINE_PRIORITY
    biometric_required = has_full_profile and finish_line

    for img_i, img in enumerate(images):
        for field in required_img:
            if field not in img:
                raise PayloadValidationError(
                    f"images[{img_i}] missing field {field!r}"
                )
        if not img["success"]:
            continue

        for p_i, person in enumerate(img["persons"]):
            for field in required_person:
                if field not in person:
                    raise PayloadValidationError(
                        f"images[{img_i}].persons[{p_i}] missing field {field!r}"
                    )

            for vec_field in ("reid_vector_b64", "face_vector_b64"):
                b64_val = person.get(vec_field)
                if b64_val is not None:
                    try:
                        from cryptography.fernet import InvalidToken
                        arr = get_encryptor().decrypt_vector_b64(b64_val)
                        if arr.size == 0:
                            raise PayloadValidationError(
                                f"images[{img_i}].persons[{p_i}].{vec_field} "
                                f"decrypted to empty array"
                            )
                    except PayloadValidationError:
                        raise
                    except Exception as exc:
                        raise PayloadValidationError(
                            f"images[{img_i}].persons[{p_i}].{vec_field} "
                            f"decrypt failed: {exc}"
                        ) from exc

            bbox = person["bbox"]
            if len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
                raise PayloadValidationError(
                    f"images[{img_i}].persons[{p_i}].bbox must be [x1,y1,x2,y2]"
                )

        if biometric_required and img["persons"]:
            sharp_persons = [p for p in img["persons"] if not p.get("is_blurry", False)]
            if sharp_persons:
                has_bio = any(
                    p.get("reid_vector_b64") is not None or p.get("face_vector_b64") is not None
                    for p in sharp_persons
                )
                if not has_bio:
                    raise PayloadValidationError(
                        f"images[{img_i}]: finish-line burst (priority==9, profile=full) "
                        f"has {len(sharp_persons)} sharp person(s) but zero biometric vectors."
                    )


# ──────────────────────────────────────────────────────────────────────────
# Banner & Entry Point
# ──────────────────────────────────────────────────────────────────────────

BANNER = """
\033[36m\033[1m
 ════════════════════════════════════════════════════════════════
   pixxEngine Asymmetric GPU Worker v2.0 — Claim-Check Consumer
 ════════════════════════════════════════════════════════════════
   Input:   gpu_tasks              (local RabbitMQ, /dev/shm paths)
   Output:  raw_inference_results  (bib detection → CPU worker)
           scribe_tasks           (calibration  → DB scribe → VPS)
   RAM:     /dev/shm/pixxengine    (auto-cleanup after inference)
   DB:      NONE — GPU never touches PostgreSQL
 ════════════════════════════════════════════════════════════════
\033[0m"""


def main():
    print(BANNER, flush=True)
    logger.info("worker_starting")
    run_worker()


if __name__ == "__main__":
    main()
