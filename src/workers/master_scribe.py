"""
Master Scribe — Identity Resolver, Database Writer & VPS Notifier (Pipeline V3)
================================================================================
Tier 3 of the pixxEngine Inference Pipeline V3.

Consumes ``scribe_tasks`` from the **local** RabbitMQ broker.

Three task paths:

1. **Bib-detection intents** (from id_cluster, schema_version=3) — resolves
   identity IDs by upserting into ``pipeline.identities`` with EMA centroid
   blending, bulk-inserts subjects, runs the ghost adoption sweep, then
   publishes a ``batch_notifications`` summary to the VPS RabbitMQ broker.

2. **Probe-calibration results** (from GPU worker, bypass id_cluster) —
   forwards directly to VPS ``batch_notifications``.  No DB writes.

3. **Course stubs** (from id_cluster, priority < 9) — publishes an empty
   ``batch_notifications`` notification.  Course processing deferred to
   V3 Phase 2.

Atomic Transaction Model:
  Every bib-detection burst is processed inside a single PostgreSQL
  transaction.  The identity ON CONFLICT + RETURNING + UPDATE pattern
  provides atomic upsert semantics with implicit row locking during the
  DO UPDATE clause, preventing concurrent scribe workers from corrupting
  centroid blends.

Identity Resolution Flow (per intent):
  A. Ghost intents (assigned_bib=NULL): INSERT a new identity row.
  B. Known-bib intents:
     1. INSERT … ON CONFLICT (project_id, bib) DO UPDATE → returns existing row
     2. EMA-blend face/ReID centroids (CENTROID_MOMENTUM = 0.8)
     3. Check shard split threshold → create appearance variant if needed
     4. UPDATE identity row with blended vectors

Security:
  Biometric vectors arrive Fernet-encrypted (AES-128-CBC + HMAC-SHA256).
  This worker decrypts them in memory, then stores raw bytes to PostgreSQL
  BYTEA columns.  Ciphertext never touches disk.

Launch::

    source pixxEngine_venv/bin/activate
    python -m src.workers.master_scribe
"""

import base64
import json
import logging
import signal
import sys
import time
import uuid
from typing import Dict, Optional

import pika
import structlog

from src.detection_config import settings
from src.encryption import get_encryptor
from src.messaging import create_local_connection, create_vps_connection
from src.workers import identity_db as db
from src.metrics.log_config import configure as _configure_structlog

# ── Intent processing priority (lower = processed first) ─────────
# Ensures highest-confidence match_type wins the first INSERT into
# pipeline.identities, setting the centroid before weaker types arrive.
_MATCH_TYPE_PRIORITY = {
    "golden_sample":           0,
    "golden_partial":          1,
    "golden_delayed":          2,
    "error_map_timing":        3,
    "ocr_unvalidated":         4,
    "blind_trust":             5,
    "hint_remainder":          6,
    "ghost_adopted":           7,
    "ghost":                   8,
    "ghost_multi_bib":         9,
    "ghost_ambiguous_partial": 9,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_configure_structlog()
logger = structlog.get_logger("master_scribe")

SCRIBE_QUEUE = "scribe_tasks"
VPS_NOTIFY_QUEUE = "batch_notifications"

from src.workers.detection_common import (
    status as _status, CYAN, GREEN, YELLOW, BOLD, RESET,
)


# ═══════════════════════════════════════════════════════════════════
# Biometric Decryption Helpers
# ═══════════════════════════════════════════════════════════════════

def _decrypt_vector(b64_str: Optional[str]):
    """Decrypt a base64-encoded Fernet-encrypted vector → numpy array.

    Used for intent-level biometrics (best_face_enc, blended_reid_enc)
    that feed into identity centroid blending via enroll_identity.
    """
    if not b64_str:
        return None
    try:
        encrypted_bytes = base64.b64decode(b64_str)
        return get_encryptor().decrypt_vector(encrypted_bytes)
    except Exception as exc:
        logger.warning("vector_decrypt_failed", error=str(exc))
        return None


def _decrypt_to_bytes(b64_str: Optional[str]) -> Optional[bytes]:
    """Decrypt a base64-encoded Fernet-encrypted vector → raw float32 bytes.

    Used for subject-level biometrics (face_enc, reid_enc) stored as
    PostgreSQL BYTEA columns.
    """
    if not b64_str:
        return None
    try:
        encrypted_bytes = base64.b64decode(b64_str)
        vec = get_encryptor().decrypt_vector(encrypted_bytes)
        return vec.tobytes()
    except Exception as exc:
        logger.warning("vector_decrypt_to_bytes_failed", error=str(exc))
        return None


# ═══════════════════════════════════════════════════════════════════
# Master Scribe Worker
# ═══════════════════════════════════════════════════════════════════

class MasterScribe:
    """Consumes scribe_tasks, executes atomic DB writes, notifies VPS.

    After a successful DB commit the worker publishes a lightweight
    notification to the VPS ``batch_notifications`` queue so the web
    dashboard can update in near real-time.
    """

    def __init__(self):
        self.connection = None
        self.channel = None
        self.vps_connection = None
        self.vps_channel = None
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame):
        logger.info("shutdown_signal", signal=signum)
        self.stop()
        sys.exit(0)

    # ── Local RabbitMQ connection ─────────────────────────────────

    def connect(self):
        """Connect to the **local** RabbitMQ broker."""
        self.connection = create_local_connection()
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=SCRIBE_QUEUE, durable=True)
        self.channel.basic_qos(prefetch_count=10)
        _status(f"Master Scribe connected to local RabbitMQ "
                f"({settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT})")

    # ── VPS RabbitMQ connection ───────────────────────────────────

    def _connect_vps(self):
        """Connect to the **VPS** RabbitMQ broker (Tailscale).
        Non-fatal — if the VPS link is down we still commit to the local DB.
        """
        try:
            self.vps_connection = create_vps_connection(
                blocked_connection_timeout=10,
                connection_attempts=1,
                retry_delay=0,
            )
            self.vps_channel = self.vps_connection.channel()
            self.vps_channel.queue_declare(queue=VPS_NOTIFY_QUEUE, durable=True)
            _status(f"Master Scribe connected to VPS RabbitMQ "
                    f"({settings.VPS_RABBITMQ_HOST}:{settings.VPS_RABBITMQ_PORT})")
        except Exception as e:
            logger.warning("vps_rabbitmq_unavailable", error=str(e))
            self.vps_connection = None
            self.vps_channel = None

    def _publish_to_vps(self, payload: dict):
        """Publish an arbitrary JSON dict to the VPS ``batch_notifications`` queue.

        Handles VPS connection lifecycle (connect/reconnect).
        Failures are non-fatal — logged and the VPS connection is reset.
        """
        if self.vps_channel is None or (
                self.vps_connection and self.vps_connection.is_closed):
            self._connect_vps()
        if self.vps_channel is None:
            return

        try:
            self.vps_channel.basic_publish(
                exchange="",
                routing_key=VPS_NOTIFY_QUEUE,
                body=json.dumps(payload, ensure_ascii=False),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                ),
            )
            logger.debug("vps_notification_sent", queue=VPS_NOTIFY_QUEUE)
        except Exception as e:
            logger.warning("vps_publish_failed", error=str(e))
            self.vps_channel = None
            self.vps_connection = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        """Blocking consume loop with auto-reconnect."""
        while True:
            try:
                self.connect()
                self._connect_vps()
                _status(f"Master Scribe waiting for tasks on '{SCRIBE_QUEUE}'")
                self.channel.basic_consume(
                    queue=SCRIBE_QUEUE,
                    on_message_callback=self._on_message,
                    auto_ack=False,
                )
                self.channel.start_consuming()
            except KeyboardInterrupt:
                logger.info("worker_shutting_down")
                self.stop()
                sys.exit(0)
            except Exception as e:
                logger.exception("connection_error", error=str(e), retry_delay_s=5)
                time.sleep(5)

    def stop(self):
        """Graceful shutdown of both local and VPS connections."""
        if self.channel and self.channel.is_open:
            try:
                self.channel.stop_consuming()
            except Exception:
                pass
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        if self.vps_connection and not self.vps_connection.is_closed:
            try:
                self.vps_connection.close()
            except Exception:
                pass
        logger.info("worker_stopped")

    # ── Message Router ────────────────────────────────────────────

    def _on_message(self, ch, method, _properties, body):
        t0 = time.time()
        burst_id = "?"
        try:
            task = json.loads(body.decode("utf-8"))
            task_type = task.get("task_type", "")

            # ── Calibration fast-path (no DB writes) ──────────────
            if task_type == "probe_calibration_result":
                self._handle_calibration_result(ch, method, task, t0)
                return

            # ── V3 bib-detection / course stub ────────────────────
            burst_id = str(task.get("burst_id") or "?")
            short = burst_id[-16:] if len(burst_id) > 16 else burst_id
            log = logger.bind(
                burst_id=burst_id,
                project_id=str(task.get("project_id", "")),
            )

            self._process_bib_detection(task, log, short, t0)

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.exception("scribe_message_failed",
                burst_id=burst_id, elapsed_ms=round(elapsed, 1),
                error=str(e),
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    # ── Bib Detection Processing ──────────────────────────────────

    def _process_bib_detection(self, task: dict, log, short: str, t0: float):
        """Execute atomic DB writes for a bib-detection burst.

        Atomic sequence per burst:
          1. Upsert photo rows
          2. Delete stale subjects (prevents reprocessing duplicates)
          3. Process identity intents (upsert identities + EMA blend)
          4. Bulk insert subject rows
          5. Update photo status
          6. Ghost adoption sweep
        """
        project_id = str(task.get("project_id", ""))
        intents = task.get("intents", [])
        photos = task.get("photos", [])
        photo_status_data = task.get("photo_status", [])

        # Empty burst (course stub or no detections) — notify VPS only
        if not intents:
            elapsed = (time.time() - t0) * 1000
            _status(f"[scribe:{short}] Empty burst (no intents) | {elapsed:.0f}ms",
                    YELLOW)
            self._publish_vps_notification(task)
            return

        subjects_to_insert = []
        bibs_to_adopt: set = set()

        with db.get_cursor() as cur:
            # 1. Upsert photos → build uuid5-to-actual mapping
            uuid5_to_actual: Dict[str, str] = {}
            for p in photos:
                actual_id = db.scribe_upsert_photo(cur, p)
                uuid5_to_actual[p["uuid"]] = actual_id

            # 2. Delete stale subjects for these photos
            actual_uuids = list(set(uuid5_to_actual.values()))
            if actual_uuids:
                db.delete_subjects_for_photos(cur, actual_uuids)

            # 3. Process identity intents (sorted by match-type confidence
            #    so the strongest evidence sets the centroid first)
            sorted_intents = sorted(
                intents,
                key=lambda i: _MATCH_TYPE_PRIORITY.get(
                    i.get("match_type", "ghost"), 8),
            )
            for intent in sorted_intents:
                assigned_bib = intent.get("assigned_bib")
                match_type = intent.get("match_type", "ghost")

                # Decrypt intent-level biometric vectors for identity upsert
                face_vec = _decrypt_vector(intent.get("best_face_enc"))
                reid_vec = _decrypt_vector(intent.get("blended_reid_enc"))

                if assigned_bib is not None:
                    # Known-bib identity: upsert + EMA blend centroids
                    bibs_to_adopt.add(assigned_bib)
                    identity_id, released_ghost = db.enroll_identity(
                        cur, project_id, assigned_bib,
                        face_vec=face_vec, reid_vec=reid_vec,
                        enrollment_type=match_type,
                    )
                    if released_ghost is not None:
                        # A rank override released old subjects to a ghost.
                        # The adoption sweep (step 6) will attempt to match
                        # the released ghost to a different confirmed bib.
                        logger.info("rank_override_ghost_released",
                            bib=assigned_bib, ghost_id=released_ghost,
                            incoming_type=match_type,
                        )
                else:
                    # Ghost identity: no bib, always creates a new row
                    identity_id = db.ensure_identity(
                        cur, project_id, bib=None,
                        face_vec=face_vec, reid_vec=reid_vec,
                    )

                # Map detections to subject rows
                for det in intent.get("detections", []):
                    real_photo_id = uuid5_to_actual.get(
                        det["photo_id"], det["photo_id"],
                    )
                    bbox = det.get("bbox", [0, 0, 0, 0])
                    subjects_to_insert.append({
                        "id": str(uuid.uuid4()),
                        "photo_id": real_photo_id,
                        "identity_id": identity_id,
                        "bbox_x": det.get("bbox_x", 0.0),
                        "bbox_y": det.get("bbox_y", 0.0),
                        "bbox_w": det.get("bbox_w", 0.0),
                        "bbox_h": det.get("bbox_h", 0.0),
                        "px1": bbox[0] if len(bbox) > 0 else 0,
                        "py1": bbox[1] if len(bbox) > 1 else 0,
                        "px2": bbox[2] if len(bbox) > 2 else 0,
                        "py2": bbox[3] if len(bbox) > 3 else 0,
                        "confidence": det.get("confidence", 0.0),
                        "area_pct": det.get("area_pct", 0.0),
                        "face_quality": det.get("face_quality", 0.0),
                        "face_enc": _decrypt_to_bytes(det.get("face_enc")),
                        "reid_enc": _decrypt_to_bytes(det.get("reid_enc")),
                        "ocr_bib": det.get("ocr_bib"),
                        "ocr_confidence": det.get("ocr_confidence"),
                        "assigned_bib": assigned_bib,
                        "match_type": match_type,
                    })

            # 4. Bulk insert subjects
            if subjects_to_insert:
                db.record_subjects_batch(cur, subjects_to_insert)

            # 5. Update photo status
            for ps in photo_status_data:
                actual_uuid = uuid5_to_actual.get(
                    ps["photo_uuid"], ps["photo_uuid"],
                )
                db.update_photo_status(
                    cur, actual_uuid, ps["status"],
                    subject_count=ps.get("subject_count", 0),
                    matched_count=ps.get("matched_count", 0),
                    inference_ms=ps.get("inference_ms", 0),
                )

            # 6. Ghost adoption sweep
            for bib in bibs_to_adopt:
                db.adopt_ghosts_for_bib(cur, project_id, bib)

        # ── Transaction committed — log and notify VPS ────────────
        elapsed = (time.time() - t0) * 1000
        n_photos = len(photos)
        n_subjects = len(subjects_to_insert)

        bibs_detected = sorted({
            i.get("assigned_bib") or i.get("consensus_bib")
            for i in intents
            if i.get("assigned_bib") or i.get("consensus_bib")
        })

        log.info("db_transaction_complete",
            db_transaction_ms=round(elapsed, 1),
            photos_upserted=n_photos, subjects_inserted=n_subjects,
            bibs_detected=bibs_detected,
        )
        _status(
            f"[scribe:{short}] {n_photos} photos, "
            f"{n_subjects} subjects | {elapsed:.0f}ms",
            GREEN,
        )

        # VPS notification (non-fatal if VPS link is down)
        vps_t0 = time.time()
        self._publish_vps_notification(task)
        log.info("vps_notification_sent",
            vps_notify_ms=round((time.time() - vps_t0) * 1000, 1),
        )

    # ── VPS Notification Builder ──────────────────────────────────

    def _publish_vps_notification(self, task: dict):
        """Build and publish a ``bib_detection`` notification to VPS.

        Reshapes the internal scribe payload into the format expected by
        the VPS consumer::

            {
              "task_type":  "bib_detection",
              "burst_id":   "p12_c292027000688_b0042",
              "photo_ids":  ["235367", "235368"],
              "job_id":     86,
              "status":     "completed",
              "runners":    [{"bib_number": "821", "confidence": 0.95}, ...],
              "per_image":  {"235367": {"runners": [...]}, ...}
            }
        """
        # Map deterministic UUIDs → original photo IDs
        uuid_to_photo_id: dict = {}
        for p in task.get("photos", []):
            uuid_to_photo_id[p["uuid"]] = str(p["photo_id"])

        photo_ids = [str(pid) for pid in task.get("photo_ids", [])]
        per_image: dict = {pid: {"runners": []} for pid in photo_ids}
        burst_bibs: dict = {}  # bib → max confidence

        for intent in task.get("intents", []):
            bib = intent.get("assigned_bib") or intent.get("consensus_bib")
            if not bib:
                continue
            intent_conf = intent.get("consensus_conf", 0.0)

            for det in intent.get("detections", []):
                photo_uuid = det.get("photo_id")
                orig_pid = uuid_to_photo_id.get(photo_uuid)
                if not orig_pid:
                    continue

                # Convert px1/py1/px2/py2 bbox → [x, y, w, h]
                bbox = det.get("bbox", [0, 0, 0, 0])
                px1 = bbox[0] if len(bbox) > 0 else 0
                py1 = bbox[1] if len(bbox) > 1 else 0
                px2 = bbox[2] if len(bbox) > 2 else 0
                py2 = bbox[3] if len(bbox) > 3 else 0

                det_conf = det.get("ocr_confidence") or intent_conf
                runner_entry = {
                    "bib_number": str(bib),
                    "confidence": round(float(det_conf), 4),
                    "bbox": [round(px1), round(py1),
                             round(px2 - px1), round(py2 - py1)],
                }

                if orig_pid in per_image:
                    per_image[orig_pid]["runners"].append(runner_entry)
                else:
                    per_image[orig_pid] = {"runners": [runner_entry]}

                # Track burst-level max confidence per bib
                bib_str = str(bib)
                if bib_str not in burst_bibs or float(det_conf) > burst_bibs[bib_str]:
                    burst_bibs[bib_str] = float(det_conf)

        # Top-level runners (unique bibs, max confidence)
        runners = [
            {"bib_number": b, "confidence": round(c, 4)}
            for b, c in sorted(burst_bibs.items())
        ]

        notification = {
            "task_type": "bib_detection",
            "burst_id": task.get("burst_id"),
            "photo_ids": photo_ids,
            "job_id": task.get("job_id"),
            "status": "completed",
            "runners": runners,
            "per_image": per_image,
        }
        self._publish_to_vps(notification)

    # ── Calibration Result Handler ────────────────────────────────

    def _handle_calibration_result(self, ch, method, task: dict, t0: float):
        """Forward a ``probe_calibration_result`` to VPS ``batch_notifications``.

        No DB writes — calibration results are final as produced by the GPU
        worker.  The VPS web app stores the camera offset and updates the
        dashboard.

        Unlike bib-detection notifications (which are reshaped into a
        lightweight summary), calibration results are published **as-is**
        so the VPS receives the full payload.
        """
        project_id = task.get("project_id", "?")
        camera = task.get("camera_serial", "?")
        status = task.get("status", "?")
        offset = task.get("offset_seconds", "?")
        try:
            self._publish_to_vps(task)
            elapsed = (time.time() - t0) * 1000
            _status(
                f"[scribe:cal] project={project_id} camera={camera} "
                f"offset={offset}s status={status} | {elapsed:.0f}ms",
                GREEN if status == "completed" else YELLOW,
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.exception("calibration_forward_failed",
                project_id=project_id, camera=camera,
                error=str(e), elapsed_ms=round(elapsed, 1),
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# ═══════════════════════════════════════════════════════════════════
# Banner & Entry Point
# ═══════════════════════════════════════════════════════════════════

BANNER = """
\033[36m\033[1m
 ═══════════════════════════════════════════════════════════════
   pixxEngine Master Scribe — Tier 3: Identity DB Writer
   Pipeline V3 — Atomic Intent Resolution
 ═══════════════════════════════════════════════════════════════
   Input:   scribe_tasks           (local RabbitMQ)
   Output:  batch_notifications    (VPS  RabbitMQ)
   DB:      pipeline.*             (PostgreSQL — identity + subjects)
   Cal:     probe_calibration_result → VPS (no DB writes)
 ═══════════════════════════════════════════════════════════════
\033[0m"""


def main():
    print(BANNER, flush=True)
    logger.info("worker_starting")
    worker = MasterScribe()
    worker.start()


if __name__ == "__main__":
    main()
