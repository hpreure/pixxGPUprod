"""
Enterprise Structured Logger — pixxEngine Burst Metrics
========================================================
Emits a **single JSON dictionary per burst** to stdout using structlog.
A log-scraper (Promtail / Fluent Bit) picks these up and feeds them into
a time-series database (Loki + Prometheus).

Usage:
    from src.metrics.burst_logger import log_burst_metrics, log_system_metrics

    log_burst_metrics(
        burst_id="abc123",
        project_id="42",
        latency_io_load_ms=120.3,
        ...
    )

All fields are optional except burst_id and project_id.  Unknown keyword
arguments are silently included in the JSON output.
"""

from __future__ import annotations

import os
import socket
import structlog
import logging
import sys
from typing import Any, Dict, Optional

from src.metrics.log_config import configure as _configure_structlog

_configure_structlog()

# Named loggers for different metric domains
_burst_log = structlog.get_logger("pixx.burst")
_system_log = structlog.get_logger("pixx.system")
_error_log = structlog.get_logger("pixx.error")

# Static context: hostname + worker PID (stamped on every line)
_HOST = socket.gethostname()
_PID = os.getpid()


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def log_burst_metrics(
    *,
    burst_id: str,
    project_id: str,
    # ── Latency (ms) ──────────────────────────────────────────
    latency_io_load_ms: float = 0.0,
    latency_yolo_person_ms: float = 0.0,
    latency_reid_ms: float = 0.0,
    latency_face_ms: float = 0.0,
    latency_ocr_pipeline_ms: float = 0.0,
    latency_total_burst_ms: float = 0.0,
    # ── Throughput ────────────────────────────────────────────
    batch_size_images: int = 0,
    batch_size_persons: int = 0,
    images_per_second: float = 0.0,
    # ── Bib Tagging / Accuracy ────────────────────────────────
    tracklets_total: int = 0,
    tracklets_matched: int = 0,
    tracklets_ghosts: int = 0,
    match_type_distribution: Optional[Dict[str, int]] = None,
    avg_ocr_confidence: float = 0.0,
    # ── Stability ─────────────────────────────────────────────
    failed_frames_count: int = 0,
    timing_gate_rejections: int = 0,
    exception_count: int = 0,
    # ── Context ───────────────────────────────────────────────
    priority: int = 0,
    is_finish_line: bool = False,
    hints: Optional[list] = None,
    # ── Overflow bucket ───────────────────────────────────────
    **extra: Any,
) -> None:
    """Emit a single structured JSON line with all burst-level metrics."""
    _burst_log.info(
        "burst_metrics",
        host=_HOST,
        pid=_PID,
        burst_id=burst_id,
        project_id=project_id,
        # Latency
        latency_io_load_ms=round(latency_io_load_ms, 1),
        latency_yolo_person_ms=round(latency_yolo_person_ms, 1),
        latency_reid_ms=round(latency_reid_ms, 1),
        latency_face_ms=round(latency_face_ms, 1),
        latency_ocr_pipeline_ms=round(latency_ocr_pipeline_ms, 1),
        latency_total_burst_ms=round(latency_total_burst_ms, 1),
        # Throughput
        batch_size_images=batch_size_images,
        batch_size_persons=batch_size_persons,
        images_per_second=round(images_per_second, 2),
        # Accuracy
        tracklets_total=tracklets_total,
        tracklets_matched=tracklets_matched,
        tracklets_ghosts=tracklets_ghosts,
        match_type_distribution=match_type_distribution or {},
        avg_ocr_confidence=round(avg_ocr_confidence, 4),
        # Stability
        failed_frames_count=failed_frames_count,
        timing_gate_rejections=timing_gate_rejections,
        exception_count=exception_count,
        # Context
        priority=priority,
        is_finish_line=is_finish_line,
        hints=hints or [],
        **extra,
    )


def log_system_metrics(
    *,
    gpu_utilization_pct: float = 0.0,
    gpu_vram_used_mb: float = 0.0,
    gpu_vram_total_mb: float = 0.0,
    gpu_temp_c: float = 0.0,
    cpu_utilization_pct: float = 0.0,
    ram_used_mb: float = 0.0,
    ram_total_mb: float = 0.0,
    rabbitmq_queue_depth: int = -1,
    **extra: Any,
) -> None:
    """Emit a periodic system-health JSON line (called by the resource monitor)."""
    _system_log.info(
        "system_metrics",
        host=_HOST,
        pid=_PID,
        gpu_utilization_pct=round(gpu_utilization_pct, 1),
        gpu_vram_used_mb=round(gpu_vram_used_mb, 1),
        gpu_vram_total_mb=round(gpu_vram_total_mb, 1),
        gpu_temp_c=round(gpu_temp_c, 1),
        cpu_utilization_pct=round(cpu_utilization_pct, 1),
        ram_used_mb=round(ram_used_mb, 1),
        ram_total_mb=round(ram_total_mb, 1),
        rabbitmq_queue_depth=rabbitmq_queue_depth,
        **extra,
    )


def log_exception(
    *,
    burst_id: Optional[str] = None,
    project_id: Optional[str] = None,
    error_category: str = "unknown",
    error_message: str = "",
    error_type: str = "",
    task_type: str = "",
    **extra: Any,
) -> None:
    """Emit a structured exception event for categorized error tracking."""
    _error_log.error(
        "pipeline_exception",
        host=_HOST,
        pid=_PID,
        burst_id=burst_id,
        project_id=project_id,
        error_category=error_category,
        error_message=error_message,
        error_type=error_type,
        task_type=task_type,
        **extra,
    )
