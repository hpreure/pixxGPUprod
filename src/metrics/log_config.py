"""
Shared structlog configuration for all pixxEngine workers.
===========================================================
Import ``configure`` at the top of each worker module to ensure structlog
is initialised before any ``structlog.get_logger()`` call::

    from src.metrics.log_config import configure as _configure_structlog
    _configure_structlog()

All workers share the same processor chain so that JSON output is
consistent across the pipeline (Feeder → GPU → CPU → Scribe).

Log output is written directly to a file (``logs/<worker>.log``) so that
terminal pipe backpressure from ``| tee`` cannot stall the workers.
The log-file path is derived from ``LOG_FILE`` env-var; when unset the
default ``logs/worker.log`` is used.
"""

import logging
import os
import sys
from pathlib import Path

import structlog

_configured = False


def configure() -> None:
    """Idempotent structlog configuration — safe to call from every worker."""
    global _configured
    if _configured:
        return

    log_file = os.environ.get("LOG_FILE", "")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _file = open(log_file, "a", buffering=1)  # line-buffered
    else:
        _file = sys.stdout

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=_file),
        cache_logger_on_first_use=True,
    )
    _configured = True
