"""
Scribe Task Publisher
======================
Publishes DB-write tasks to the local ``scribe_tasks`` queue on the
local RabbitMQ broker.  Binary fields (face_enc, reid_enc) are
base64-encoded automatically for JSON transport.

Usage::

    from src.workers.scribe_publisher import publish_scribe_task

    ok = publish_scribe_task({
        'burst_id':     '...',
        'project_id':   '...',
        'photos':       [...],     # photo upsert data
        'subjects':     [...],     # subject rows (may contain bytes)
        'photo_status': [...],     # status update data
    })

Returns True on success.  If False, the caller should fall back to
``db.execute_scribe_writes(task)`` for inline synchronous writes.
"""

import base64
import json
import logging

import pika

from src.detection_config import settings
from src.messaging import create_local_connection

logger = logging.getLogger(__name__)

SCRIBE_QUEUE = "scribe_tasks"

# ── Lazy-init singleton connection ────────────────────────────────
_connection = None
_channel = None


def _ensure_connection():
    """Establish (or re-establish) connection to local RabbitMQ."""
    global _connection, _channel
    if _connection is not None and not _connection.is_closed:
        if _channel is not None and _channel.is_open:
            return
    _connection = create_local_connection()
    _channel = _connection.channel()
    _channel.queue_declare(queue=SCRIBE_QUEUE, durable=True)
    logger.info(
        "Scribe publisher: connected to local RabbitMQ (%s:%d/%s)",
        settings.RABBITMQ_HOST, settings.RABBITMQ_PORT, settings.RABBITMQ_VHOST,
    )


def _serialize_task(task: dict) -> dict:
    """Deep-copy task, encoding binary face/reid fields to base64.

    Handles both V1.5 ``subjects`` and V3 ``intents`` payload formats.
    """
    out = dict(task)

    # ── V1.5 subjects path ────────────────────────────────────────
    serialized = []
    for s in out.get("subjects", []):
        row = dict(s)
        if isinstance(row.get("face_enc"), (bytes, bytearray)):
            row["face_enc"] = base64.b64encode(row["face_enc"]).decode("ascii")
        if isinstance(row.get("reid_enc"), (bytes, bytearray)):
            row["reid_enc"] = base64.b64encode(row["reid_enc"]).decode("ascii")
        serialized.append(row)
    out["subjects"] = serialized

    # ── V3 intents path ───────────────────────────────────────────
    serialized_intents = []
    for intent in out.get("intents", []):
        row = dict(intent)
        for key in ("best_face_enc", "blended_reid_enc"):
            if isinstance(row.get(key), (bytes, bytearray)):
                row[key] = base64.b64encode(row[key]).decode("ascii")
        if "detections" in row:
            dets = []
            for d in row["detections"]:
                d = dict(d)
                if isinstance(d.get("face_enc"), (bytes, bytearray)):
                    d["face_enc"] = base64.b64encode(d["face_enc"]).decode("ascii")
                if isinstance(d.get("reid_enc"), (bytes, bytearray)):
                    d["reid_enc"] = base64.b64encode(d["reid_enc"]).decode("ascii")
                dets.append(d)
            row["detections"] = dets
        serialized_intents.append(row)
    out["intents"] = serialized_intents

    return out


def publish_scribe_task(task: dict) -> bool:
    """Publish a scribe task to the local ``scribe_tasks`` queue.

    Binary fields in ``task['subjects']`` (V1.5) or ``task['intents']``
    (V3) are automatically base64-encoded for JSON transport.

    Returns ``True`` on success, ``False`` on failure (caller should
    NACK the upstream message for redelivery).
    """
    global _connection, _channel
    try:
        _ensure_connection()
        body = json.dumps(_serialize_task(task), ensure_ascii=False)
        _channel.basic_publish(
            exchange="",
            routing_key=SCRIBE_QUEUE,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,          # persistent
                content_type="application/json",
            ),
        )
        logger.debug("Published scribe task: burst=%s", task.get("burst_id"))
        return True
    except Exception as e:
        logger.error("Failed to publish scribe task: %s", e, exc_info=True)
        try:
            if _connection and not _connection.is_closed:
                _connection.close()
        except Exception:
            pass
        _connection = None
        _channel = None
        return False
