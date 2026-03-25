"""
Image Feeder — Claim-Check Pipeline Entry Point
=================================================
The single entry point for all GPU workloads.  Consumes tasks from the
VPS ``bib_detection_tasks`` queue, downloads images from R2 via the
pixx-signer → imgproxy CDN stack, drops them into **shared memory**
(``/dev/shm``), and publishes a lightweight *ticket* to the local
``gpu_tasks`` queue.  GPU workers pick up tickets, read images from RAM,
run inference, and delete the files when done.

Architecture (Claim-Check Pattern)::

    VPS bib_detection_tasks
            │
            ▼
    ┌───────────────┐         /dev/shm/pixxengine/{pid}/{tid}/*.jpg
    │  Image Feeder │──────►  (RAM drop)
    └──────┬────────┘
           │  ticket (local paths + metadata)
           ▼
    local gpu_tasks  ──►  GPU Workers  ──►  raw_inference_results
                           │
                           └──►  rm /dev/shm/pixxengine/{pid}/{tid}/

Backpressure:
  The feeder **pauses** consuming when ``/dev/shm`` has less than 4 GB
  free, polling every 2 s until space is reclaimed by GPU worker cleanup.

Task routing — all tasks get the same treatment:

  1. Download images from R2 to ``/dev/shm``
  2. Rewrite paths to RAM locations
  3. Publish ticket to local ``gpu_tasks``

The GPU worker is responsible for:
  - Routing by ``task_type`` (``bib_detection`` vs ``probe_calibration``)
  - Running inference
  - Publishing results
  - Deleting images from RAM after processing

Launch::

    source pixxEngine_venv/bin/activate
    python -m src.workers.image_feeder
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import pika
import requests
import structlog

from src.detection_config import settings
from src.messaging import create_local_connection, create_vps_connection
from src.metrics.log_config import configure as _configure_structlog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_configure_structlog()
logger = structlog.get_logger("image_feeder")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

SIGNER_URL = os.getenv(
    "PXX_SIGNER_API_URL", "http://100.110.159.48:3000/sign"
).rstrip("/")
SIGNER_BASE = SIGNER_URL.rsplit("/sign", 1)[0]
SIGN_BATCH_ENDPOINT = f"{SIGNER_BASE}/sign-batch"
HEALTH_ENDPOINT = f"{SIGNER_BASE}/health"

BUCKET = os.getenv("S3_BUCKET", "pixxengine-prod")

# ── RAM staging (claim-check drop zone) ───────────────────────────
SHM_BASE = Path("/dev/shm/pixxengine")

# ── Backpressure ──────────────────────────────────────────────────
BACKPRESSURE_THRESHOLD_BYTES = int(
    os.getenv("SHM_BACKPRESSURE_GB", "4")
) * (1024 ** 3)                                   # default 4 GB free
BACKPRESSURE_POLL_S = 2.0                          # poll interval while paused

# ── CDN tuning ────────────────────────────────────────────────────
CDN_CONCURRENCY = int(os.getenv("CDN_CONCURRENCY", "32"))
CDN_TIMEOUT = int(os.getenv("CDN_TIMEOUT", "30"))
CDN_FETCH_RETRIES = int(os.getenv("CDN_FETCH_RETRIES", "2"))
CDN_RETRY_TIMEOUT = int(os.getenv("CDN_RETRY_TIMEOUT", "10"))

# ── Signing retries ──────────────────────────────────────────────
SIGN_MAX_RETRIES = 3
SIGN_BACKOFF_S = 2.0
# ── Batch signing across bursts ──────────────────────────────
BATCH_DRAIN_S = float(os.getenv("BATCH_DRAIN_S", "0.05"))  # 50ms drain window
BATCH_MAX_IMAGES = int(os.getenv("BATCH_MAX_IMAGES", "128"))  # signer hard limit
# ── Queue names ───────────────────────────────────────────────────
QUEUE_IN  = "bib_detection_tasks"      # VPS → Feeder  (R2 keys)
QUEUE_OUT = "gpu_tasks"                # Feeder → GPU   (RAM paths)

# ── imgproxy presets ──────────────────────────────────────────────
PRESET_FULL  = "rs:fill:1920:0/q:88/f:jpg"   # finish-line (person+bib+face)
PRESET_PROBE = "rs:fill:640:0/q:80/f:jpg"    # probe / calibration

# ── Priority threshold ────────────────────────────────────────────
FINISH_LINE_PRIORITY = 9

# ── SHM reaper (orphan cleanup) ───────────────────────────────────
SHM_MAX_AGE_S = int(os.getenv("SHM_MAX_AGE_S", "10800"))       # 3 hours
SHM_REAPER_INTERVAL_S = int(os.getenv("SHM_REAPER_INTERVAL_S", "3600"))  # 1 hour

# ── R2 key pattern ────────────────────────────────────────────────
_R2_KEY_RE = re.compile(r"^users/\d+/projects/\d+/[a-f0-9]+$")

# ── Terminal colours ──────────────────────────────────────────────
from src.workers.detection_common import (
    status as _status, CYAN, GREEN, YELLOW, RED, BOLD, RESET,
)


# ══════════════════════════════════════════════════════════════════════
# SHM Reaper — background orphan cleanup
# ══════════════════════════════════════════════════════════════════════

class ShmReaper(threading.Thread):
    """
    Background daemon that deletes orphaned claim-check directories
    from ``/dev/shm/pixxengine`` when their mtime exceeds a TTL.

    Catches the case where the GPU worker is SIGKILL'd, OOM-killed,
    or crashes without executing its ``finally`` cleanup block.
    """

    def __init__(
        self,
        base: Path = SHM_BASE,
        max_age_s: int = SHM_MAX_AGE_S,
        interval_s: int = SHM_REAPER_INTERVAL_S,
    ):
        super().__init__(daemon=True, name="shm-reaper")
        self._base = base
        self._max_age_s = max_age_s
        self._interval_s = interval_s

    def run(self) -> None:
        logger.info("shm_reaper_started",
            interval_s=self._interval_s, max_age_s=self._max_age_s,
            base=str(self._base),
        )
        while True:
            time.sleep(self._interval_s)
            try:
                self._sweep()
            except Exception as exc:
                logger.warning("shm_reaper_sweep_error", error=str(exc))

    def _sweep(self) -> None:
        if not self._base.exists():
            return
        now = time.time()
        reaped_dirs = 0
        reaped_files = 0
        reaped_bytes = 0

        # Task dirs are /dev/shm/pixxengine/{project_id}/{task_id}/
        for project_dir in self._base.iterdir():
            if not project_dir.is_dir():
                continue
            for task_dir in project_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                try:
                    age = now - task_dir.stat().st_mtime
                except OSError:
                    continue
                if age < self._max_age_s:
                    continue
                # Tally before removal
                files = list(task_dir.iterdir())
                n_files = len(files)
                n_bytes = sum(f.stat().st_size for f in files if f.is_file())
                shutil.rmtree(task_dir, ignore_errors=True)
                reaped_dirs += 1
                reaped_files += n_files
                reaped_bytes += n_bytes
                logger.warning("shm_reaper_reaped_orphan",
                    path=str(task_dir), age_s=round(age),
                    files=n_files, size_mb=round(n_bytes / (1024 * 1024), 1),
                )
            # Remove empty project dir
            try:
                project_dir.rmdir()
            except OSError:
                pass

        if reaped_dirs:
            _status(
                f"[ShmReaper] Sweep done — reaped {reaped_dirs} orphan dir(s), "
                f"{reaped_files} files, {reaped_bytes / (1024 * 1024):.1f} MB",
                YELLOW,
            )


# ══════════════════════════════════════════════════════════════════════
# Backpressure — /dev/shm free-space gate
# ══════════════════════════════════════════════════════════════════════

def shm_free_bytes() -> int:
    """Return available bytes on /dev/shm."""
    st = os.statvfs("/dev/shm")
    return st.f_bavail * st.f_frsize


def shm_has_capacity() -> bool:
    """True when /dev/shm has >= BACKPRESSURE_THRESHOLD_BYTES free."""
    return shm_free_bytes() >= BACKPRESSURE_THRESHOLD_BYTES


def wait_for_capacity() -> None:
    """Block until /dev/shm has enough free space (backpressure gate)."""
    if shm_has_capacity():
        return
    free_gb = shm_free_bytes() / (1024 ** 3)
    threshold_gb = BACKPRESSURE_THRESHOLD_BYTES / (1024 ** 3)
    logger.info("backpressure_status", shm_free_gb=round(free_gb, 2),
        threshold_gb=round(threshold_gb), paused=True)
    _status(
        f"[Backpressure] /dev/shm {free_gb:.1f} GB free < "
        f"{threshold_gb:.0f} GB threshold — PAUSING",
        YELLOW,
    )
    while not shm_has_capacity():
        time.sleep(BACKPRESSURE_POLL_S)
    free_gb = shm_free_bytes() / (1024 ** 3)
    _status(
        f"[Backpressure] /dev/shm {free_gb:.1f} GB free — RESUMING",
        GREEN,
    )


# ══════════════════════════════════════════════════════════════════════
# Path Validation
# ══════════════════════════════════════════════════════════════════════

def is_r2_key(path: str) -> bool:
    """Return True if *path* matches the expected R2 object key pattern."""
    return bool(_R2_KEY_RE.match(path))


# ══════════════════════════════════════════════════════════════════════
# pixx-signer API
# ══════════════════════════════════════════════════════════════════════

def health_check() -> bool:
    """Verify pixx-signer is reachable.  Non-fatal on failure."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            logger.info("signer_health_ok", uptime_s=data.get("uptime", 0))
            return True
        logger.warning("signer_health_fail", http_status=resp.status_code)
        return False
    except Exception as exc:
        logger.warning("signer_unreachable", error=str(exc))
        return False


def sign_batch(
    keys: List[str],
    options: str = PRESET_FULL,
) -> List[str]:
    """
    Sign up to 128 R2 object keys in one API call.

    Returns a list of signed HTTPS URLs in the same order as *keys*.
    Raises on persistent failure after retries.
    """
    sources = [f"s3://{BUCKET}/{k}" for k in keys]
    payload = {"sources": sources, "options": options}

    last_exc: Optional[Exception] = None
    for attempt in range(SIGN_MAX_RETRIES):
        try:
            resp = requests.post(
                SIGN_BATCH_ENDPOINT, json=payload, timeout=10
            )
            resp.raise_for_status()
            urls = resp.json()["urls"]
            if len(urls) != len(keys):
                raise ValueError(
                    f"Signer returned {len(urls)} URLs for {len(keys)} keys"
                )
            return urls
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < SIGN_MAX_RETRIES - 1:
                delay = SIGN_BACKOFF_S * (attempt + 1)
                logger.warning("sign_batch_retry",
                    attempt=attempt + 1, max_retries=SIGN_MAX_RETRIES,
                    error=str(exc), retry_delay_s=delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"sign_batch failed after {SIGN_MAX_RETRIES} attempts"
    ) from last_exc


# ══════════════════════════════════════════════════════════════════════
# Async Image Download → /dev/shm
# ══════════════════════════════════════════════════════════════════════

async def _fetch_one(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
) -> Dict:
    """Download a single signed URL to *dest* with retry.

    First attempt uses the shorter CDN_RETRY_TIMEOUT so transient CDN
    stalls recover quickly.  Final attempt uses the full CDN_TIMEOUT.
    """
    t0 = time.perf_counter()
    last_err: Dict = {}
    for attempt in range(CDN_FETCH_RETRIES):
        is_last = attempt == CDN_FETCH_RETRIES - 1
        attempt_timeout = CDN_TIMEOUT if is_last else CDN_RETRY_TIMEOUT
        try:
            timeout = aiohttp.ClientTimeout(total=attempt_timeout)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    last_err = {
                        "ok": False, "status": resp.status,
                        "error": body[:200],
                        "elapsed": time.perf_counter() - t0, "size": 0,
                    }
                    if is_last or resp.status < 500:
                        return last_err
                    continue
                data = await resp.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return {
                    "ok": True, "status": 200, "size": len(data),
                    "elapsed": time.perf_counter() - t0,
                    "cache": resp.headers.get("X-Cache-Status", "?"),
                }
        except Exception as exc:
            err_msg = str(exc)[:200] or type(exc).__name__
            last_err = {
                "ok": False, "status": 0, "error": err_msg,
                "elapsed": time.perf_counter() - t0, "size": 0,
            }
            if is_last:
                return last_err
    return last_err


async def fetch_images(
    urls: List[str],
    dest_paths: List[Path],
    concurrency: int = CDN_CONCURRENCY,
) -> List[Dict]:
    """Download all signed URLs in parallel to their dest paths."""
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _fetch_one(session, url, dest)
            for url, dest in zip(urls, dest_paths)
        ]
        return await asyncio.gather(*tasks)


SIGN_BATCH_MAX = 128  # signer hard limit per request


def sign_and_fetch(
    keys: List[str],
    dest_paths: List[Path],
    options: str = PRESET_FULL,
    concurrency: int = CDN_CONCURRENCY,
) -> List[Dict]:
    """Sign a batch of R2 keys, download in parallel to /dev/shm.

    Automatically chunks into groups of up to 128 (signer hard limit).
    """
    all_urls: List[str] = []
    for i in range(0, len(keys), SIGN_BATCH_MAX):
        chunk = keys[i : i + SIGN_BATCH_MAX]
        all_urls.extend(sign_batch(chunk, options))
    return asyncio.run(fetch_images(all_urls, dest_paths, concurrency))


# ══════════════════════════════════════════════════════════════════════
# Claim-Check: RAM Drop + Ticket Build
# ══════════════════════════════════════════════════════════════════════

def _pick_preset(message: Dict) -> str:
    """Choose imgproxy preset based on task type.

    All ``bib_detection`` tasks (course *and* finish-line) require
    1920 px images because the GPU worker now runs ``PROFILE_FULL``
    for every bib burst.  Only ``probe_calibration`` uses the smaller
    640 px probe preset.
    """
    task_type = message.get("task_type", "bib_detection")
    if task_type == "probe_calibration":
        return PRESET_PROBE
    return PRESET_FULL


def _extract_photo_list(message: Dict) -> List[Dict]:
    """
    Return the list of photo dicts from the message, regardless of
    whether they live under ``images[]`` or ``payload.photos[]``.
    """
    images = message.get("images")
    if images:
        return images
    return message.get("payload", {}).get("photos", [])


def _task_id(message: Dict) -> str:
    """Return a unique ID for the task (burst_id or fallback)."""
    return str(
        message.get("burst_id")
        or message.get("payload", {}).get("sub_event_id")
        or f"task_{int(time.time()*1000)}"
    )


def download_to_shm(message: Dict) -> Dict:
    """
    Download all images to /dev/shm and build a ticket message.

    1. Extracts R2 keys from the photo list.
    2. Signs + downloads to ``/dev/shm/pixxengine/{project_id}/{task_id}/``.
    3. Rewrites ``path`` fields to the RAM file locations.
    4. Injects ``_claim_check`` metadata for the GPU worker.

    Returns the rewritten message (the *ticket*).
    """
    photos = _extract_photo_list(message)
    if not photos:
        logger.warning("message_no_photos")
        return message

    project_id = str(message.get("project_id", "unknown"))
    tid = _task_id(message)
    short_tag = tid[-16:] if len(tid) > 16 else tid
    preset = _pick_preset(message)

    _status(
        f"[Feeder:{short_tag}] {len(photos)} images | "
        f"preset={'FULL' if 'fill:1920' in preset else 'PROBE'}"
    )

    # ── Build R2 keys and destination paths in /dev/shm ───────────
    shm_dir = SHM_BASE / project_id / tid
    r2_keys: List[str] = []
    dest_paths: List[Path] = []

    for i, photo in enumerate(photos):
        key = photo.get("path", "")
        if not is_r2_key(key):
            logger.warning("invalid_r2_key", index=i, key=key)
        r2_keys.append(key)
        photo_id = str(photo.get("photo_id", key.rsplit("/", 1)[-1]))
        dest_paths.append(shm_dir / f"{photo_id}.jpg")

    # ── Sign + download to RAM ────────────────────────────────────
    t0 = time.perf_counter()
    results = sign_and_fetch(r2_keys, dest_paths, options=preset)
    fetch_ms = (time.perf_counter() - t0) * 1000

    # ── Rewrite paths + tally ─────────────────────────────────────
    ok_count = 0
    fail_count = 0
    total_bytes = 0

    for photo, dest, result in zip(photos, dest_paths, results):
        if result["ok"]:
            photo["r2_key"] = photo.get("path", "")
            photo["path"] = str(dest)
            ok_count += 1
            total_bytes += result["size"]
        else:
            logger.warning("image_fetch_failed",
                path=photo.get("path"), http_status=result.get("status"),
                error=result.get("error", "")[:100],
            )
            photo["_fetch_failed"] = True
            fail_count += 1

    total_mb = total_bytes / (1024 * 1024)
    throughput = total_mb / (fetch_ms / 1000) if fetch_ms > 0 else 0

    color = GREEN if fail_count == 0 else YELLOW
    _status(
        f"[Feeder:{short_tag}] Done: {ok_count} OK, {fail_count} fail | "
        f"{total_mb:.1f} MB in {fetch_ms:.0f}ms ({throughput:.1f} MB/s)",
        color,
    )

    logger.info("download_complete",
        burst_id=short_tag, project_id=project_id,
        download_ms=round(fetch_ms, 1), ok_count=ok_count,
        fail_count=fail_count, total_mb=round(total_mb, 1),
    )

    # ── Inject claim-check metadata ───────────────────────────────
    message["_claim_check"] = {
        "shm_dir": str(shm_dir),
        "fetch_ms": round(fetch_ms, 1),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "total_bytes": total_bytes,
        "preset": preset,
    }

    return message


# ══════════════════════════════════════════════════════════════════════
# RabbitMQ Consumer
# ══════════════════════════════════════════════════════════════════════

class ImageFeederWorker:
    """
    Claim-check pipeline entry point.

    Consumes tasks from VPS ``bib_detection_tasks``, downloads images
    to ``/dev/shm``, and publishes tickets to local ``gpu_tasks``.
    Pauses automatically when shared memory is low (backpressure).
    """

    def __init__(self):
        self.vps_connection: Optional[pika.BlockingConnection] = None
        self.vps_channel = None
        self.local_connection: Optional[pika.BlockingConnection] = None
        self.local_channel = None
        self._task_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame):
        logger.info("shutdown_signal", signal=signum)
        self.stop()
        sys.exit(0)

    # ── VPS RabbitMQ (inbound) ────────────────────────────────────

    def _connect_vps(self):
        self.vps_connection = create_vps_connection()
        self.vps_channel = self.vps_connection.channel()
        self.vps_channel.basic_qos(prefetch_count=BATCH_MAX_IMAGES)
        _status(
            f"Connected to VPS RabbitMQ "
            f"({settings.VPS_RABBITMQ_HOST}:{settings.VPS_RABBITMQ_PORT})"
        )

    # ── Local RabbitMQ (outbound) ─────────────────────────────────

    def _connect_local(self):
        self.local_connection = create_local_connection()
        self.local_channel = self.local_connection.channel()
        self.local_channel.queue_declare(queue=QUEUE_OUT, durable=True)
        _status(
            f"Connected to local RabbitMQ "
            f"({settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT})"
        )

    # ── Publish ticket to local gpu_tasks ─────────────────────────

    def _publish_ticket(self, ticket: Dict) -> bool:
        """Publish ticket to local ``gpu_tasks``.

        On channel/connection failure, reconnects once and retries.
        Returns *True* on success, *False* on permanent failure.
        """
        body = json.dumps(ticket, ensure_ascii=False)
        props = pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
        )
        for attempt in range(2):
            try:
                self.local_channel.basic_publish(
                    exchange="",
                    routing_key=QUEUE_OUT,
                    body=body,
                    properties=props,
                )
                return True
            except Exception as exc:
                if attempt == 0:
                    logger.warning("local_channel_reconnecting", error=str(exc))
                    try:
                        self._connect_local()
                    except Exception as reconn_exc:
                        logger.error("local_reconnect_failed", error=str(reconn_exc))
                        return False
                else:
                    logger.error("publish_failed_after_reconnect",
                        queue=QUEUE_OUT, error=str(exc),
                    )
                    return False
        return False  # unreachable but defensive

    # ── Batch message handler ────────────────────────────────────

    def _accumulate(self, _ch, method, _properties, body):
        """Callback: buffer inbound messages for batch processing."""
        try:
            message = json.loads(body.decode("utf-8"))
        except Exception as exc:
            logger.exception("message_parse_error", error=str(exc))
            self.vps_channel.basic_nack(
                delivery_tag=method.delivery_tag, requeue=True)
            return
        self._pending.append((method, message))

    def _batch_loop(self):
        """Drain-and-flush loop: collect prefetched messages,
        batch-sign their R2 keys, download all in parallel, publish."""
        self._pending: List[tuple] = []

        self.vps_channel.basic_consume(
            queue=QUEUE_IN,
            on_message_callback=self._accumulate,
            auto_ack=False,
        )

        while True:
            # Block until at least one message arrives (or timeout)
            self.vps_connection.process_data_events(
                time_limit=BATCH_DRAIN_S)

            if not self._pending:
                continue

            # Keep draining: RabbitMQ pushes prefetched messages in
            # bursts; give it short micro-polls to deliver more before
            # we flush.  Stop when no new messages arrive or we've
            # accumulated enough images.
            pending_images = sum(
                len(_extract_photo_list(m)) for _, m in self._pending)
            while pending_images < BATCH_MAX_IMAGES:
                before = len(self._pending)
                self.vps_connection.process_data_events(
                    time_limit=0.01)       # 10ms micro-poll
                if len(self._pending) == before:
                    break                  # nothing new, stop waiting
                pending_images = sum(
                    len(_extract_photo_list(m))
                    for _, m in self._pending)

            # Peel messages until we hit the image-count cap.
            # Remaining messages carry over to the next iteration.
            batch: List[tuple] = []
            img_count = 0
            while self._pending:
                method, msg = self._pending[0]
                n = len(_extract_photo_list(msg))
                if batch and img_count + n > BATCH_MAX_IMAGES:
                    break                  # over budget; leave for next loop
                batch.append(self._pending.pop(0))
                img_count += n

            try:
                wait_for_capacity()
                self._process_batch(batch)
            except ConnectionError:
                for method, _msg in batch:
                    try:
                        self.vps_channel.basic_nack(
                            delivery_tag=method.delivery_tag, requeue=True)
                    except Exception:
                        pass
                raise
            except Exception as exc:
                logger.exception("batch_fatal",
                    error=str(exc), batch_size=len(batch))
                for method, _msg in batch:
                    try:
                        self.vps_channel.basic_nack(
                            delivery_tag=method.delivery_tag, requeue=True)
                    except Exception:
                        pass

    def _process_batch(self, batch: List[tuple]):
        """Sign + download + publish a batch of messages in minimal
        signer round-trips."""
        t0_batch = time.perf_counter()

        # ── 1. Collect R2 keys / dest paths across all messages ───
        all_keys: List[str] = []
        all_dests: List[Path] = []
        key_presets: List[str] = []      # preset per key (for grouping)
        entries: list = []               # per-message bookkeeping

        for method, msg in batch:
            photos = _extract_photo_list(msg)
            project_id = str(msg.get("project_id", "unknown"))
            tid = _task_id(msg)
            short_tag = tid[-16:] if len(tid) > 16 else tid
            shm_dir = SHM_BASE / project_id / tid
            preset = _pick_preset(msg)

            start_idx = len(all_keys)
            for i, photo in enumerate(photos):
                key = photo.get("path", "")
                if not is_r2_key(key):
                    logger.warning("invalid_r2_key",
                        burst_id=short_tag, index=i, key=key)
                all_keys.append(key)
                photo_id = str(
                    photo.get("photo_id", key.rsplit("/", 1)[-1]))
                all_dests.append(shm_dir / f"{photo_id}.jpg")
                key_presets.append(preset)
            end_idx = len(all_keys)

            entries.append((
                method, msg, photos, project_id,
                tid, short_tag, shm_dir, preset,
                start_idx, end_idx,
            ))

        total_keys = len(all_keys)

        # ── 2. Batch-sign: group by preset, chunk at 128 ─────────
        all_urls: List[Optional[str]] = [None] * total_keys
        by_preset: Dict[str, List[int]] = {}
        for idx, preset in enumerate(key_presets):
            by_preset.setdefault(preset, []).append(idx)

        sign_calls = 0
        for preset, indices in by_preset.items():
            preset_keys = [all_keys[i] for i in indices]
            for chunk_off in range(0, len(preset_keys), SIGN_BATCH_MAX):
                chunk = preset_keys[chunk_off:chunk_off + SIGN_BATCH_MAX]
                chunk_idx = indices[chunk_off:chunk_off + SIGN_BATCH_MAX]
                urls = sign_batch(chunk, preset)
                for i, url in zip(chunk_idx, urls):
                    all_urls[i] = url
                sign_calls += 1

        # ── 3. Download all images in parallel ────────────────────
        t0_fetch = time.perf_counter()
        results = asyncio.run(fetch_images(all_urls, all_dests))
        fetch_ms_total = (time.perf_counter() - t0_fetch) * 1000

        # ── 4. Batch-level summary (log first, before per-msg) ─────
        batch_ok = sum(1 for r in results if r["ok"])
        batch_fail = total_keys - batch_ok
        batch_bytes = sum(r.get("size", 0) for r in results if r["ok"])
        batch_mb = batch_bytes / (1024 * 1024)
        batch_throughput = (
            batch_mb / (fetch_ms_total / 1000)
            if fetch_ms_total > 0 else 0)

        _status(
            f"[Batch] {len(batch)} msgs, {total_keys} images, "
            f"{sign_calls} sign call(s) | {batch_ok} OK, "
            f"{batch_fail} fail | {batch_mb:.1f} MB in "
            f"{fetch_ms_total:.0f}ms ({batch_throughput:.1f} MB/s)",
            GREEN if batch_fail == 0 else YELLOW,
        )
        logger.info("batch_complete",
            messages=len(batch), total_keys=total_keys,
            sign_calls=sign_calls,
            download_ms=round(fetch_ms_total, 1),
            ok=batch_ok, fail=batch_fail,
            total_mb=round(batch_mb, 1),
        )

        # ── 5. Distribute results → per-message finalization ──────
        for (
            method, msg, photos, project_id,
            tid, short_tag, shm_dir, preset,
            start, end,
        ) in entries:
            msg_results = results[start:end]
            msg_dests = all_dests[start:end]

            ok_count = 0
            fail_count = 0
            total_bytes = 0
            max_elapsed = 0.0

            for photo, dest, result in zip(photos, msg_dests, msg_results):
                if result["ok"]:
                    photo["r2_key"] = photo.get("path", "")
                    photo["path"] = str(dest)
                    ok_count += 1
                    total_bytes += result["size"]
                else:
                    logger.warning("image_fetch_failed",
                        path=photo.get("path"),
                        http_status=result.get("status"),
                        error=result.get("error", "")[:100],
                    )
                    photo["_fetch_failed"] = True
                    fail_count += 1
                max_elapsed = max(
                    max_elapsed, result.get("elapsed", 0))

            fetch_ms = max_elapsed * 1000

            logger.debug("message_done",
                burst_id=short_tag, ok=ok_count, fail=fail_count)

            msg["_claim_check"] = {
                "shm_dir": str(shm_dir),
                "fetch_ms": round(fetch_ms, 1),
                "ok_count": ok_count,
                "fail_count": fail_count,
                "total_bytes": total_bytes,
                "preset": preset,
            }

            if self._publish_ticket(msg):
                self.vps_channel.basic_ack(
                    delivery_tag=method.delivery_tag)
                self._task_count += 1
            else:
                self.vps_channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=True)
                raise ConnectionError(
                    f"Cannot publish to {QUEUE_OUT} — local channel dead")

        batch_ms = (time.perf_counter() - t0_batch) * 1000
        logger.info("batch_signed",
            messages=len(batch), total_keys=total_keys,
            sign_calls=sign_calls,
            download_ms=round(fetch_ms_total, 1),
            batch_ms=round(batch_ms, 1),
        )

    # ── Main loop ─────────────────────────────────────────────────

    def start(self):
        """Blocking consume loop with auto-reconnect."""
        health_check()
        SHM_BASE.mkdir(parents=True, exist_ok=True)

        # ── Start background SHM orphan reaper ────────────────────
        ShmReaper().start()

        while True:
            try:
                self._connect_vps()
                self._connect_local()

                free_gb = shm_free_bytes() / (1024 ** 3)
                _status(
                    f"Image Feeder listening on '{QUEUE_IN}' (VPS) "
                    f"-> '{QUEUE_OUT}' (local) | "
                    f"/dev/shm {free_gb:.1f} GB free | "
                    f"max_images={BATCH_MAX_IMAGES}"
                )
                self._batch_loop()

            except KeyboardInterrupt:
                logger.info("worker_shutting_down")
                self.stop()
                sys.exit(0)
            except Exception as exc:
                logger.exception("connection_error", error=str(exc), retry_delay_s=5)
                time.sleep(5)

    def stop(self):
        """Graceful shutdown."""
        for conn in (self.vps_connection, self.local_connection):
            if conn and not conn.is_closed:
                try:
                    conn.close()
                except Exception:
                    pass
        logger.info("worker_stopped")


# ══════════════════════════════════════════════════════════════════════
# Banner & Entry Point
# ══════════════════════════════════════════════════════════════════════

BANNER = """
\033[36m\033[1m
 ════════════════════════════════════════════════════════════════
   pixxEngine Image Feeder v3.0 — Claim-Check Pipeline
 ════════════════════════════════════════════════════════════════
   Input:   bib_detection_tasks    (VPS   RabbitMQ)
   Output:  gpu_tasks              (local RabbitMQ)
   RAM:     /dev/shm/pixxengine    (claim-check drop zone)
   CDN:     pixx-signer → imgproxy → Cloudflare R2
   Gate:    Backpressure at < {bp_gb} GB /dev/shm free
 ════════════════════════════════════════════════════════════════
\033[0m"""


def main():
    bp_gb = BACKPRESSURE_THRESHOLD_BYTES / (1024 ** 3)
    print(BANNER.format(bp_gb=int(bp_gb)), flush=True)
    logger.info("worker_starting",
        signer=SIGN_BATCH_ENDPOINT, bucket=BUCKET,
        ram_drop=str(SHM_BASE), backpressure_gb=round(bp_gb),
        concurrency=CDN_CONCURRENCY,
    )

    worker = ImageFeederWorker()
    worker.start()


if __name__ == "__main__":
    main()
