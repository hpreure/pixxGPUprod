"""
Workers Module — GPU Inference Workers
=======================================
Contains all worker modules for the pixxEngine detection pipeline.

Modules:
- detection_common:         Shared utilities for detection pipelines
- inference_engine:         Core GPU inference for person/bib/text/OCR/ReID/Face
- identity_db:              PostgreSQL helpers for pipeline schema
- scribe_publisher:         Async DB-write queue publisher
- probe_calibration:        Camera time offset calculation
- asymmetric_gpu_worker:    Tier 1 — raw GPU inference, no DB
- cpu_worker:               Tier 2 — identity clustering, identity resolution
- db_scribe:                Tier 3 — async DB writer + VPS notifier
"""

# Lazy imports — avoid pulling torch/tensorrt/cv2 at package-load time.
# The heavy modules are imported only when their symbols are first accessed.

__all__ = [
    'get_engine',
    'InferenceEngine',
    'InferenceResult',
    'ProbeCalibration',
    'run_probe_calibration',
]

_LAZY_MAP = {
    'get_engine':             'src.workers.inference_engine',
    'InferenceEngine':        'src.workers.inference_engine',
    'InferenceResult':        'src.workers.inference_engine',
    'ProbeCalibration':       'src.workers.probe_calibration',
    'run_probe_calibration':  'src.workers.probe_calibration',
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        import importlib
        mod = importlib.import_module(_LAZY_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
