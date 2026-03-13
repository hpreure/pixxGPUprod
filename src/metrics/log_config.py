"""
Shared structlog configuration for all pixxEngine workers.
===========================================================
Import ``configure`` at the top of each worker module to ensure structlog
is initialised before any ``structlog.get_logger()`` call::

    from src.metrics.log_config import configure as _configure_structlog
    _configure_structlog()

All workers share the same processor chain so that JSON output is
consistent across the pipeline (Feeder → GPU → CPU → Scribe).
"""

import logging
import sys

import structlog

_configured = False


def configure() -> None:
    """Idempotent structlog configuration — safe to call from every worker."""
    global _configured
    if _configured:
        return
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
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True
