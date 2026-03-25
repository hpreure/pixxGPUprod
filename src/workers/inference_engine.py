"""
GPU Inference Engine — Batched Detection Processor
====================================================
Performs person detection, bib detection, text detection, OCR,
ReID, and face recognition on race photos.

Performance-critical design:
  - **Bulk image loading** via thread pool
  - **Batched YOLO person detection** across all burst images at once
  - **Batched ReID** — all person crops from all images in one GPU call
  - **Batched Face** — all person crops in one InsightFace call
  - **Batched bib+text+OCR** — all bib crops grouped and OCR'd together
"""

import cv2
import re
import time
import torch
import tensorrt as trt
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import structlog

from src.detection_config import settings, detection_settings
from src.models import InsightFaceWrapper, ParseqWrapper
from src.reid_wrapper import ReIDWrapper
from src.metrics.log_config import configure as _configure_structlog

_configure_structlog()
logger = structlog.get_logger("inference_engine")

# Thread pool for parallel image loading
_io_pool = ThreadPoolExecutor(max_workers=8)

# Bib crop padding: expand detected bib box by this fraction on each side
# to capture leading/trailing digits that YOLO may clip.
BIB_CROP_PAD_FRAC = 0.10


@dataclass
class BibDetection:
    """Single bib number detection result."""
    bib_number: Optional[str]
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class PersonDetection:
    """Single person detection with biometrics and bib."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    reid_vector: Optional[np.ndarray] = None
    face_vector: Optional[np.ndarray] = None
    face_quality: float = 0.0
    face_yaw: float = 0.0
    # Laplacian variance of the grayscale crop — proxy for sharpness.
    # Computed in the engine immediately after crop extraction (pass 2).
    # 0.0 is a valid score meaning the crop is perfectly flat/uniform.
    blur_score: float = 0.0
    # True when blur_score is below BLUR_THRESHOLD.  Soft quality flag:
    # excluded from OCR voting and biometric blending in the clustering
    # pipeline, but the detection is kept for cluster continuity and
    # still goes through the full GPU pipeline (face, ReID, bib).
    is_blurry: bool = False
    bibs: List[BibDetection] = None

    def __post_init__(self):
        if self.bibs is None:
            self.bibs = []


@dataclass
class InferenceResult:
    """Result of inference on a single photo."""
    photo_path: str
    persons: List[PersonDetection]
    inference_time_ms: float
    img_width: int = 0
    img_height: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class BatchTimingBreakdown:
    """Timing breakdown for a single process_photos() call.

    Attached to the InferenceEngine after every batch so the caller
    (asymmetric_gpu_worker) can forward it to the structured logger.
    """
    load_ms: float = 0.0
    det_ms: float = 0.0
    reid_ms: float = 0.0
    face_ms: float = 0.0
    bib_ms: float = 0.0
    total_ms: float = 0.0
    n_images: int = 0
    n_persons: int = 0
    n_failed: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Seg-based person detector  (yolo26m-seg — bboxes + masks in one pass)
# ──────────────────────────────────────────────────────────────────────────

class _SegPersonDetector:
    """
    Native TRT wrapper around yolo26m-seg.engine.

    A single forward pass per full-frame image returns both person bounding
    boxes and per-person binary segmentation masks in original image coords.

    This replaces two previously separate GPU passes:
      1. yolo26l.engine   — detect-only person bbox extraction
      2. SegInferencer    — separate seg pass for ReID background removal

    Output tensors from the engine (dynamic batch B):
      output0  (B, 300, 38) — NMS-free [x1,y1,x2,y2,conf,cls, 32 mask coefs]
      output1  (B, 32, 160, 160) — prototype mask features

    The TRT engine is built with dynamic batch 1-16 so we can process
    multiple burst images in a single GPU kernel launch.
    """

    IMGSZ       = 640
    PERSON_CLS  = 0
    MASK_THRESH = 0.45    # sigmoid threshold for binary mask
    MAX_BATCH   = 12      # upper bound per inference call

    def __init__(self, engine_path: Path, device: str = "cuda:0"):
        self.device = device
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        if not engine_path.exists():
            raise FileNotFoundError(
                f"Seg person engine not found: {engine_path}\n"
                f"Run scripts/rebuild_all_trt_engines.py to build it."
            )

        logger.info("seg_detector_loading", engine=engine_path.name)
        runtime = trt.Runtime(_TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        self._dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8:  torch.int8,
            trt.DataType.BOOL:  torch.bool,
        }

        self._stream = torch.cuda.current_stream().cuda_stream
        logger.info("seg_detector_ready", engine=engine_path.name)

    def _allocate_buffers(self, batch: int) -> dict:
        """Allocate GPU buffers for a given batch size and bind them."""
        self._ctx.set_input_shape("images", (batch, 3, self.IMGSZ, self.IMGSZ))
        buffers: dict = {}
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = self._ctx.get_tensor_shape(name)
            dtype = self._dtype_map.get(
                self._engine.get_tensor_dtype(name), torch.float32
            )
            buf = torch.zeros(*shape, dtype=dtype, device=self.device)
            buffers[name] = buf
            self._ctx.set_tensor_address(name, buf.data_ptr())
        return buffers

    def _preprocess(self, img_bgr: np.ndarray):
        """Letterbox + normalise one image. Returns (tensor_chw, scale, pad_t, pad_l, h0, w0)."""
        h0, w0 = img_bgr.shape[:2]
        scale  = self.IMGSZ / max(h0, w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        pad_t  = (self.IMGSZ - nh) // 2
        pad_l  = (self.IMGSZ - nw) // 2

        canvas = np.full((self.IMGSZ, self.IMGSZ, 3), 114, dtype=np.uint8)
        canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = cv2.resize(img_bgr, (nw, nh))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_chw = torch.from_numpy(rgb).permute(2, 0, 1)   # (3, 640, 640)
        return t_chw, scale, pad_t, pad_l, h0, w0

    def _decode_one(
        self, dets: torch.Tensor, proto: torch.Tensor,
        scale: float, pad_t: int, pad_l: int, h0: int, w0: int,
        nh: int, nw: int,
    ) -> List[Tuple[int, int, int, int, float, np.ndarray]]:
        """Decode detections + masks for one image in the batch."""
        _seg_conf = detection_settings.SEG_CONF
        results: List[Tuple[int, int, int, int, float, np.ndarray]] = []
        for i in range(dets.shape[0]):
            conf = float(dets[i, 4])
            cls  = int(dets[i, 5])
            if cls != self.PERSON_CLS or conf < _seg_conf:
                continue

            bx1, by1, bx2, by2 = dets[i, :4].cpu().numpy()
            x1 = max(0,  int((bx1 - pad_l) / scale))
            y1 = max(0,  int((by1 - pad_t) / scale))
            x2 = min(w0, int((bx2 - pad_l) / scale))
            y2 = min(h0, int((by2 - pad_t) / scale))
            if x2 <= x1 or y2 <= y1:
                continue

            mc       = dets[i, 6:].unsqueeze(0)
            mask_160 = torch.sigmoid(
                (mc @ proto.reshape(32, -1)).reshape(160, 160)
            ).cpu().numpy()
            mask_640  = cv2.resize(mask_160, (self.IMGSZ, self.IMGSZ),
                                   interpolation=cv2.INTER_LINEAR)
            region    = mask_640[pad_t:pad_t + nh, pad_l:pad_l + nw]
            mask_orig = cv2.resize(region, (w0, h0), interpolation=cv2.INTER_LINEAR)
            mask_bin  = (mask_orig > self.MASK_THRESH).astype(np.uint8)

            results.append((x1, y1, x2, y2, conf, mask_bin))
        return results

    # ── Public API ────────────────────────────────────────────────

    def detect(
        self, img_bgr: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, np.ndarray]]:
        """Single-image convenience wrapper."""
        return self.detect_batch([img_bgr])[0]

    def detect_batch(
        self, images_bgr: List[np.ndarray],
    ) -> List[List[Tuple[int, int, int, int, float, np.ndarray]]]:
        """
        Run batched seg inference on multiple full-frame BGR images.

        Returns one result list per image: ``[(x1, y1, x2, y2, conf, mask), ...]``
        Automatically chunks into MAX_BATCH-sized sub-batches.
        """
        all_results: List[List[Tuple[int, int, int, int, float, np.ndarray]]] = []

        for start in range(0, len(images_bgr), self.MAX_BATCH):
            chunk = images_bgr[start:start + self.MAX_BATCH]
            B = len(chunk)

            # Preprocess all images in this chunk
            preprocessed = [self._preprocess(img) for img in chunk]
            t_batch = torch.stack([p[0] for p in preprocessed]).to(self.device)  # (B,3,640,640)

            buffers = self._allocate_buffers(B)
            buffers["images"].copy_(t_batch.to(buffers["images"].dtype))
            self._ctx.execute_async_v3(self._stream)
            torch.cuda.synchronize()

            out0 = buffers["output0"].float()   # (B, 300, 38)
            out1 = buffers["output1"].float()   # (B, 32, 160, 160)

            for bi in range(B):
                _, scale, pad_t, pad_l, h0, w0 = preprocessed[bi]
                nh = int(h0 * scale)
                nw = int(w0 * scale)
                all_results.append(
                    self._decode_one(out0[bi], out1[bi], scale, pad_t, pad_l, h0, w0, nh, nw)
                )

        return all_results


# ──────────────────────────────────────────────────────────────────────────
# Native TRT YOLO detect-only wrapper  (bib / text — pre-allocated buffers)
# ──────────────────────────────────────────────────────────────────────────

class _YOLODetector:
    """
    Native TRT wrapper for detect-only YOLO engines (bib, text).

    Pre-allocates max-size input + output GPU buffers at __init__ and
    reuses them every call via slicing + .copy_().  This eliminates the
    per-call CUDA tensor creation/destruction that Ultralytics .predict()
    performs — the root cause of 30-50 s VRAM fragmentation stalls.

    Engine IO (NMS already baked in at export):
      input   "images"  (B, 3, imgsz, imgsz)  FP32
      output  "output0" (B, 300, 6)            FP32  [x1,y1,x2,y2,conf,cls]
    """

    MAX_BATCH = 32

    def __init__(self, engine_path: str, imgsz: int, device: str = "cuda:0"):
        self.imgsz = imgsz
        self.device = device

        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"YOLO detect engine not found: {engine_path}\n"
                f"Run scripts/build_yolo26_engines.py to build it."
            )

        logger.info("yolo_detector_loading", engine=engine_path.name, imgsz=imgsz)
        runtime = trt.Runtime(_TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()
        self._stream = torch.cuda.current_stream().cuda_stream

        # ── Pre-allocate max-size buffers (pinned for entire process) ──
        self._inp_buf = torch.zeros(
            self.MAX_BATCH, 3, imgsz, imgsz,
            dtype=torch.float32, device=device,
        )
        self._out_buf = torch.zeros(
            self.MAX_BATCH, 300, 6,
            dtype=torch.float32, device=device,
        )
        logger.info(
            "yolo_detector_ready",
            engine=engine_path.name,
            inp_MB=round(self._inp_buf.nelement() * 4 / 1e6, 1),
            out_MB=round(self._out_buf.nelement() * 4 / 1e6, 1),
        )

    # ── Preprocessing ─────────────────────────────────────────────

    @staticmethod
    def _preprocess_crop(img_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
        """BGR HWC uint8 → RGB CHW float32 [0,1].  Caller already resized."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)          # (3, H, W)

    # ── Public API ────────────────────────────────────────────────

    def detect_batch(
        self,
        images_bgr: List[np.ndarray],
        conf: float = 0.25,
    ) -> List[np.ndarray]:
        """
        Run batched detection on pre-resized BGR crops.

        Args:
            images_bgr: list of BGR uint8 numpy arrays, already resized to
                        (imgsz × imgsz).
            conf:       minimum confidence threshold.

        Returns:
            List (one per image) of Nx6 float32 numpy arrays
            ``[x1, y1, x2, y2, conf, cls]``.  Empty (0,6) array if no dets.
        """
        n = len(images_bgr)
        if n == 0:
            return []

        all_results: List[np.ndarray] = []

        for start in range(0, n, self.MAX_BATCH):
            chunk = images_bgr[start:start + self.MAX_BATCH]
            B = len(chunk)

            # Build batch tensor on CPU, copy into pre-allocated GPU buffer
            batch_cpu = torch.stack([
                self._preprocess_crop(img, self.imgsz) for img in chunk
            ])                                                   # (B, 3, H, W)
            self._inp_buf[:B].copy_(batch_cpu)

            # Bind sliced views and execute
            self._ctx.set_input_shape("images", (B, 3, self.imgsz, self.imgsz))
            self._ctx.set_tensor_address("images", self._inp_buf.data_ptr())
            self._ctx.set_tensor_address("output0", self._out_buf.data_ptr())
            self._ctx.execute_async_v3(self._stream)
            torch.cuda.synchronize()

            # Read results and apply confidence filter
            raw = self._out_buf[:B].cpu().numpy()                # (B, 300, 6)
            for bi in range(B):
                dets = raw[bi]                                   # (300, 6)
                mask = dets[:, 4] >= conf
                all_results.append(dets[mask].copy())            # (K, 6)

        return all_results


# Model loading profiles
PROFILE_PROBE = "probe"          # YOLO person + bib + text + OCR only
PROFILE_FULL  = "full"           # Everything: YOLO + OCR + ReID + Face


# How often to flush the CUDA caching allocator's free-block pool.
# After this many process_photos() calls, torch.cuda.empty_cache()
# is invoked proactively.  Reduced from 200 → 50 on 2026-03-20 as
# additional safety net alongside the _YOLODetector rewrite.
_CUDA_CACHE_FLUSH_INTERVAL = 50

# Log a warning when any pipeline phase exceeds this threshold.
_STALL_WARN_MS = 5_000


class InferenceEngine:
    """GPU inference engine with full batching across burst images."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._loaded_profile = None

        # Model references
        self.seg_detector = None    # _SegPersonDetector (yolo26m-seg: bboxes + masks)
        self.yolo_bib = None
        self.yolo_text = None
        self.reid_model = None
        self.face_model = None
        self.ocr_model = None

        # Timing breakdown from the most recent process_photos() call.
        # Populated after every batch so callers can forward metrics.
        self.last_timing: BatchTimingBreakdown = BatchTimingBreakdown()

        # Counter for periodic CUDA cache flush.
        self._batch_counter: int = 0

        logger.info("engine_init", device=device)

    @property
    def models_loaded(self) -> bool:
        return self._loaded_profile is not None

    def load_models(self, profile: str = PROFILE_FULL):
        """
        Load AI models for the given profile.

        Profiles:
          - 'probe': YOLO person/bib/text + PARSeq OCR  (probe calibration)
          - 'full':  All of the above + TransReID ReID + InsightFace  (bib_detection)
        """
        if self._loaded_profile == PROFILE_FULL:
            return
        if self._loaded_profile == profile:
            return

        if self.seg_detector is None:
            logger.info("models_loading", profile=profile)
            self.seg_detector = _SegPersonDetector(
                engine_path=Path(settings.YOLO_PERSON_MODEL), device=self.device
            )

            self.yolo_bib = _YOLODetector(
                engine_path=settings.YOLO_BIB_MODEL,
                imgsz=640, device=self.device,
            )
            self.yolo_text = _YOLODetector(
                engine_path=settings.YOLO_TEXT_MODEL,
                imgsz=320, device=self.device,
            )

        if self.ocr_model is None:
            logger.info("model_loading", model="PARSeq")
            self.ocr_model = ParseqWrapper("parseq", self.device)
            if "cuda" in self.device:
                self.ocr_model.half()

        if profile == PROFILE_FULL:
            if self.reid_model is None:
                logger.info("model_loading", model="TransReID")
                self.reid_model = ReIDWrapper(device=self.device)
                if "cuda" in self.device:
                    self.reid_model.half()

            if self.face_model is None:
                logger.info("model_loading", model="InsightFace")
                self.face_model = InsightFaceWrapper("buffalo_l", self.device)
                if "cuda" in self.device:
                    self.face_model.half()

        self._loaded_profile = profile
        logger.info("models_loaded", profile=profile)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def process_photos(self, photo_paths: List[str]) -> List[InferenceResult]:
        """
        Process a batch of photos with full pipeline batching.

        Pipeline:
          1. Parallel image loading (ThreadPool)
          2. Batched person detection (all images → one YOLO-seg call)
          3. Per-image cap (MAX_PERSONS_PER_IMAGE, by conf × area)
          4. Batched Face extraction (capped survivors only)
          5. Faceless rejection
          6. Batched ReID extraction (survivors only → one TransReID call)
          7. Batched bib detection + text + OCR (survivors only)
        """
        if not self.models_loaded:
            self.load_models()

        t0 = time.time()
        n = len(photo_paths)

        # ── 1. Parallel image loading ─────────────────────────────
        t_load = time.time()
        futures = [_io_pool.submit(_load_image, p) for p in photo_paths]
        loaded = [f.result() for f in futures]
        load_ms = (time.time() - t_load) * 1000

        # Separate successful loads from failures
        images = []          # (idx, img_bgr, h, w)
        results = [None] * n  # pre-fill with errors for failed loads

        for idx, (path, load_result) in enumerate(zip(photo_paths, loaded)):
            if load_result is None:
                results[idx] = InferenceResult(
                    photo_path=path, persons=[], inference_time_ms=0,
                    success=False, error=f"Could not load image: {path}"
                )
            else:
                img_bgr, h, w = load_result
                images.append((idx, img_bgr, h, w))

        if not images:
            return results

        # ── 2. Edge-zone mask + Person detection (yolo26m-seg) ────────────
        # Black-out the left and right edge strips BEFORE YOLO-seg sees
        # the image.  This is a letterbox-style hard mask: YOLO literally
        # cannot produce detections in blacked-out regions, so edge
        # spectators, partial-body close-passers, and lens-distortion
        # artefacts are eliminated with zero post-filter leaks.
        # Crops from valid-zone detections whose bbox extends slightly
        # into the masked strip still get full pixels from the ORIGINAL
        # image (kept separately).
        _zone_l = detection_settings.ZONE_LEFT_PCT
        _zone_r = detection_settings.ZONE_RIGHT_PCT

        t_det = time.time()
        masked_images = []
        for _, img_bgr, h_img, w_img in images:
            # Build masked copy — black out edge zones
            masked = img_bgr
            if _zone_l > 0 or _zone_r > 0:
                masked = img_bgr.copy()
                left_px  = int(w_img * _zone_l)
                right_px = int(w_img * _zone_r)
                if left_px > 0:
                    masked[:, :left_px] = 0
                if right_px > 0:
                    masked[:, w_img - right_px:] = 0
            masked_images.append(masked)
        person_results_batch = self.seg_detector.detect_batch(masked_images)
        det_ms = (time.time() - t_det) * 1000

        # ── Collect raw boxes from seg output ──────────────────────────────────
        raw_boxes = []  # (img_i, x1, y1, x2, y2, conf, h, w, seg_mask)
        for img_i, (idx, img_bgr, h, w) in enumerate(images):
            for x1, y1, x2, y2, conf, seg_mask in person_results_batch[img_i]:
                raw_boxes.append((img_i, x1, y1, x2, y2, conf, h, w, seg_mask))

        # ── Compute per-image anchor from non-border boxes ────────────────────
        # Each image gets its OWN anchor — the largest non-border person in
        # that specific frame.  A cross-image anchor is dangerous because a
        # single close-up in one burst photo inflates the threshold and kills
        # every foreground runner in every other photo
        # (root-cause of P159 batch-2 zero-subject failures, 2026-02-28).
        _per_image_anchor: Dict[int, float] = {}
        for img_i_a, x1, y1, x2, y2, _, h, w, _ in raw_boxes:
            is_border = (x1 <= 5 or y1 <= 5 or x2 >= w - 5 or y2 >= h - 5)
            if not is_border:
                area = float((x2 - x1) * (y2 - y1))
                if area > _per_image_anchor.get(img_i_a, 0.0):
                    _per_image_anchor[img_i_a] = area

        # Fallback: if ALL detections in an image touch the border, use the
        # largest detection in that image so the anchor is never zero.
        for img_i_a, x1, y1, x2, y2, _, h, w, _ in raw_boxes:
            if img_i_a not in _per_image_anchor:
                area = float((x2 - x1) * (y2 - y1))
                if area > _per_image_anchor.get(img_i_a, 0.0):
                    _per_image_anchor[img_i_a] = area

        # ── Pre-filter + crop extraction ──────────────────────────────────────
        # Hard gates applied in order:
        #   1. Anchor-relative size  — reject tiny background persons.
        #   2. Per-image cap         — keep top-N by confidence × area.
        #   3. Face extraction       — batched InsightFace on capped survivors.
        #   4. Faceless rejection    — drop persons with no detected face.
        # After these gates, only survivors go through ReID, bib, text, OCR.
        _blur_thresh = detection_settings.BLUR_THRESHOLD
        _anchor_pct  = detection_settings.ANCHOR_AREA_MIN_PCT
        _max_per_img = detection_settings.MAX_PERSONS_PER_IMAGE

        # ── Stage A: anchor filter + crop / blur ──────────────────
        # anchor_survivors: list of (img_i, x1, y1, x2, y2, conf,
        #                            raw_crop, seg_mask, blur_score, is_blurry)
        anchor_survivors = []

        for img_i, x1, y1, x2, y2, conf, h, w, seg_mask in raw_boxes:
            person_area = (x2 - x1) * (y2 - y1)
            _anchor_min = _per_image_anchor.get(img_i, 1.0) * _anchor_pct
            if person_area < _anchor_min:
                continue

            raw_crop = images[img_i][1][y1:y2, x1:x2].copy()

            # Blur score — CPU only, ~0.1 ms/crop.  Soft flag only.
            if raw_crop.size > 0:
                gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            else:
                blur_score = 0.0
            is_blurry = _blur_thresh > 0.0 and blur_score < _blur_thresh

            anchor_survivors.append(
                (img_i, x1, y1, x2, y2, conf, raw_crop, seg_mask, blur_score, is_blurry)
            )

        # ── Stage B: per-image cap BEFORE face extraction ─────────
        # Cap anchor survivors to MAX_PERSONS_PER_IMAGE by conf × area
        # so InsightFace never processes more than N crops per image.
        _img_buckets: Dict[int, list] = {}
        for surv in anchor_survivors:
            _img_buckets.setdefault(surv[0], []).append(surv)

        capped_survivors = []
        for img_i, entries in _img_buckets.items():
            if _max_per_img > 0 and len(entries) > _max_per_img:
                entries.sort(
                    key=lambda t: t[5] * float((t[3] - t[1]) * (t[4] - t[2])),
                    reverse=True,
                )
                entries = entries[:_max_per_img]
            capped_survivors.extend(entries)

        # ── Stage C: batched Face extraction on capped survivors ──
        t_face = time.time()
        all_face_pre = [None] * len(capped_survivors)
        if self.face_model is not None and capped_survivors:
            try:
                face_crops = [s[6] for s in capped_survivors]  # raw_crop
                # Build per-crop seg masks so InsightFace can use mask-overlap
                # to pick the correct face when multiple faces are detected.
                face_seg_masks: List[Optional[np.ndarray]] = []
                for s in capped_survivors:
                    img_i, x1, y1, x2, y2 = s[0], s[1], s[2], s[3], s[4]
                    full_mask = s[7]  # seg_mask in full-image coords
                    if full_mask is not None:
                        crop_mask = full_mask[y1:y2, x1:x2]
                        face_seg_masks.append(crop_mask)
                    else:
                        face_seg_masks.append(None)
                # Build photo hints for diagnostic logging inside InsightFaceWrapper
                _photo_hints = []
                for s in capped_survivors:
                    _ii = s[0]
                    _pp = photo_paths[images[_ii][0]] if _ii < len(images) else "?"
                    _photo_hints.append(_pp.rsplit("/", 1)[-1] if "/" in str(_pp) else str(_pp))
                face_results = self.face_model.extract(face_crops, seg_masks=face_seg_masks, photo_hints=_photo_hints)
                for i, fr in enumerate(face_results):
                    if fr and fr.get('embedding') is not None:
                        all_face_pre[i] = fr
            except Exception as e:
                logger.warning("batched_face_failed", error=str(e))
        face_ms = (time.time() - t_face) * 1000

        # ── Stage D: faceless rejection + build downstream crops ──
        # NOTE: faceless rejection only applies when face_model is loaded
        # (PROFILE_FULL).  Probe calibration uses PROFILE_PROBE which has
        # no face model — all persons must survive for OCR-based calibration.
        _has_face_model = self.face_model is not None

        per_image_person_boxes = [[] for _ in range(len(images))]
        crop_map: List[Tuple[int, int, float, bool]] = []
        gpu_crops_reid: List[np.ndarray] = []
        gpu_crops_raw:  List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        face_kept: List[Optional[dict]] = []

        _faceless_dropped = []  # collect details for logging
        for si, surv in enumerate(capped_survivors):
            # Drop faceless only when the face model actually ran
            if _has_face_model and all_face_pre[si] is None:
                _img_i_d = surv[0]
                _x1d, _y1d, _x2d, _y2d, _confd = surv[1], surv[2], surv[3], surv[4], surv[5]
                _photo_path_d = photo_paths[images[_img_i_d][0]] if _img_i_d < len(images) else "?"
                _faceless_dropped.append({
                    "photo": _photo_path_d.rsplit("/", 1)[-1] if "/" in str(_photo_path_d) else str(_photo_path_d),
                    "bbox": [_x1d, _y1d, _x2d, _y2d],
                    "conf": round(_confd, 3),
                    "crop_hw": [_y2d - _y1d, _x2d - _x1d],
                })
                continue

            img_i, x1, y1, x2, y2, conf, raw_crop, seg_mask, blur_score, is_blurry = surv
            person_idx = len(per_image_person_boxes[img_i])
            per_image_person_boxes[img_i].append((x1, y1, x2, y2, conf))
            crop_map.append((img_i, person_idx, blur_score, is_blurry))
            face_kept.append(all_face_pre[si])

            # Build masked crop (gray background) for ReID
            masked_crop = raw_crop.copy()
            crop_mask = None
            if seg_mask is not None:
                crop_mask = seg_mask[y1:y2, x1:x2]
                if crop_mask.shape == masked_crop.shape[:2]:
                    masked_crop = np.where(
                        crop_mask[:, :, None].astype(bool),
                        masked_crop,
                        np.full_like(masked_crop, 128),
                    )
                else:
                    crop_mask = None

            gpu_crops_reid.append(masked_crop)
            gpu_crops_raw.append((raw_crop, crop_mask))

        total_persons = len(crop_map)
        total_gpu = len(gpu_crops_reid)

        logger.info("early_filter_summary",
            anchor_survivors=len(anchor_survivors),
            capped_survivors=len(capped_survivors),
            after_face_rejection=total_persons,
            faceless_dropped=len(_faceless_dropped),
        )
        if _faceless_dropped:
            logger.info("faceless_rejection_detail",
                dropped=_faceless_dropped[:20],  # cap to avoid log bloat
                total_dropped=len(_faceless_dropped),
            )

        # ── Stage E: faceless rescue for zero-survivor images ─────
        # When faceless rejection kills every person in an image (e.g. a
        # dominant back-facing runner inflated the anchor and the only
        # anchor survivor had no face), re-scan raw_boxes with a relaxed
        # anchor and run InsightFace on candidates.  Any with a face are
        # added as survivors.  Only fires on images that currently have
        # zero persons, so it cannot degrade existing correct results.
        if _has_face_model:
            _rescue_factor = detection_settings.RESCUE_ANCHOR_FACTOR
            _rescue_pct = _anchor_pct * _rescue_factor
            _zero_imgs = {
                img_i for img_i in range(len(images))
                if not per_image_person_boxes[img_i]
            }
            if _zero_imgs:
                # Collect rescue candidates: raw detections that failed
                # the original anchor but pass the relaxed one.
                _rescue_cands = []  # same tuple shape as anchor_survivors
                for img_i, x1, y1, x2, y2, conf, h, w, seg_mask in raw_boxes:
                    if img_i not in _zero_imgs:
                        continue
                    person_area = (x2 - x1) * (y2 - y1)
                    _rescue_min = _per_image_anchor.get(img_i, 1.0) * _rescue_pct
                    if person_area < _rescue_min:
                        continue
                    raw_crop = images[img_i][1][y1:y2, x1:x2].copy()
                    if raw_crop.size > 0:
                        gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
                        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    else:
                        blur_score = 0.0
                    is_blurry = _blur_thresh > 0.0 and blur_score < _blur_thresh
                    _rescue_cands.append(
                        (img_i, x1, y1, x2, y2, conf, raw_crop, seg_mask, blur_score, is_blurry)
                    )

                # Cap rescue candidates per image (same MAX_PERSONS_PER_IMAGE)
                _rescue_buckets: Dict[int, list] = {}
                for rc in _rescue_cands:
                    _rescue_buckets.setdefault(rc[0], []).append(rc)
                _rescue_capped = []
                for img_i_r, entries_r in _rescue_buckets.items():
                    if _max_per_img > 0 and len(entries_r) > _max_per_img:
                        entries_r.sort(
                            key=lambda t: t[5] * float((t[3] - t[1]) * (t[4] - t[2])),
                            reverse=True,
                        )
                        entries_r = entries_r[:_max_per_img]
                    _rescue_capped.extend(entries_r)

                # Run InsightFace on rescue candidates
                if _rescue_capped:
                    _rescue_face = [None] * len(_rescue_capped)
                    try:
                        _rf_crops = [rc[6] for rc in _rescue_capped]
                        _rf_masks: List[Optional[np.ndarray]] = []
                        for rc in _rescue_capped:
                            fm = rc[7]
                            if fm is not None:
                                _rf_masks.append(fm[rc[2]:rc[4], rc[1]:rc[3]])
                            else:
                                _rf_masks.append(None)
                        _rf_hints = []
                        for rc in _rescue_capped:
                            _pp = photo_paths[images[rc[0]][0]] if rc[0] < len(images) else "?"
                            _rf_hints.append(_pp.rsplit("/", 1)[-1] if "/" in str(_pp) else str(_pp))
                        _rf_results = self.face_model.extract(
                            _rf_crops, seg_masks=_rf_masks, photo_hints=_rf_hints
                        )
                        for ri, fr in enumerate(_rf_results):
                            if fr and fr.get('embedding') is not None:
                                _rescue_face[ri] = fr
                    except Exception as e:
                        logger.warning("rescue_face_failed", error=str(e))

                    # Add faced rescue candidates as survivors
                    _rescued_count = 0
                    for ri, rc in enumerate(_rescue_capped):
                        if _rescue_face[ri] is None:
                            continue
                        img_i, x1, y1, x2, y2, conf, raw_crop, seg_mask, blur_score, is_blurry = rc
                        person_idx = len(per_image_person_boxes[img_i])
                        per_image_person_boxes[img_i].append((x1, y1, x2, y2, conf))
                        crop_map.append((img_i, person_idx, blur_score, is_blurry))
                        face_kept.append(_rescue_face[ri])

                        masked_crop = raw_crop.copy()
                        crop_mask = None
                        if seg_mask is not None:
                            crop_mask = seg_mask[y1:y2, x1:x2]
                            if crop_mask.shape == masked_crop.shape[:2]:
                                masked_crop = np.where(
                                    crop_mask[:, :, None].astype(bool),
                                    masked_crop,
                                    np.full_like(masked_crop, 128),
                                )
                            else:
                                crop_mask = None
                        gpu_crops_reid.append(masked_crop)
                        gpu_crops_raw.append((raw_crop, crop_mask))
                        _rescued_count += 1

                    if _rescued_count > 0:
                        total_persons = len(crop_map)
                        total_gpu = len(gpu_crops_reid)
                        logger.info("faceless_rescue",
                            zero_images=len(_zero_imgs),
                            rescue_candidates=len(_rescue_capped),
                            rescued=_rescued_count,
                        )

        # ── 3. Batched ReID extraction — masked crops (survivors only) ──
        all_reid_gpu = [None] * total_gpu
        if self.reid_model is not None and total_gpu > 0:
            t_reid = time.time()
            try:
                extracted = self.reid_model.extract(gpu_crops_reid)
                for i, vec in enumerate(extracted):
                    all_reid_gpu[i] = vec
            except Exception as e:
                logger.warning("batched_reid_failed", error=str(e))
            reid_ms = (time.time() - t_reid) * 1000
        else:
            reid_ms = 0

        # ── 4. Batched bib + text + OCR — raw crops (survivors only) ──
        t_bib = time.time()
        all_bibs_gpu = [[] for _ in range(total_gpu)]
        if total_gpu > 0:
            raw_only = [data[0] for data in gpu_crops_raw]
            masks_only = [data[1] for data in gpu_crops_raw]
            all_bibs_gpu = self._batched_bib_pipeline(raw_only, masks_only)
        bib_ms = (time.time() - t_bib) * 1000

        # ── 5. Assemble PersonDetections per image ────────────────
        person_lists = [[] for _ in range(len(images))]

        for crop_idx, (img_i, person_idx, blur_score, is_blurry) in enumerate(crop_map):
            x1, y1, x2, y2, conf = per_image_person_boxes[img_i][person_idx]

            reid_vec  = all_reid_gpu[crop_idx]
            face_data = face_kept[crop_idx]
            bibs      = all_bibs_gpu[crop_idx]

            face_vec     = face_data['embedding']       if face_data else None
            face_quality = face_data.get('quality', 0.0) if face_data else 0.0
            face_yaw     = face_data.get('face_yaw', 0.0) if face_data else 0.0

            # Debug: log bbox ↔ bib pairing so crossing bugs can be diagnosed.
            if bibs:
                logger.debug("person_bib_pairing",
                    img_i=img_i, person_idx=person_idx,
                    bbox=(x1, y1, x2, y2),
                    bibs=[(b.bib_number, round(b.confidence, 3)) for b in bibs],
                )

            person_lists[img_i].append(PersonDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                reid_vector=reid_vec,
                face_vector=face_vec,
                face_quality=face_quality,
                face_yaw=face_yaw,
                blur_score=blur_score,
                is_blurry=is_blurry,
                bibs=bibs,
            ))

        elapsed_ms = (time.time() - t0) * 1000

        # ── 6. Build InferenceResults ─────────────────────────────
        for img_i, (idx, img_bgr, h, w) in enumerate(images):
            per_image_ms = elapsed_ms / len(images)  # approximate per-image
            results[idx] = InferenceResult(
                photo_path=photo_paths[idx],
                persons=person_lists[img_i],
                inference_time_ms=per_image_ms,
                img_width=w,
                img_height=h,
                success=True,
            )

        # ── 7. Expose timing breakdown to callers ────────────────
        self.last_timing = BatchTimingBreakdown(
            load_ms=load_ms,
            det_ms=det_ms,
            reid_ms=reid_ms,
            face_ms=face_ms,
            bib_ms=bib_ms,
            total_ms=elapsed_ms,
            n_images=n,
            n_persons=total_persons,
            n_failed=sum(1 for r in results if r and not r.success),
        )

        logger.info("batch_inference_complete",
            n_images=n, n_persons=total_persons,
            load_ms=round(load_ms), det_ms=round(det_ms),
            reid_ms=round(reid_ms), face_ms=round(face_ms),
            bib_ms=round(bib_ms), total_ms=round(elapsed_ms),
        )

        # ── Stall detection ───────────────────────────────────────
        for phase_name, phase_ms in [
            ("det", det_ms), ("face", face_ms),
            ("reid", reid_ms), ("bib", bib_ms),
        ]:
            if phase_ms > _STALL_WARN_MS:
                logger.warning("gpu_stall_detected",
                    phase=phase_name, phase_ms=round(phase_ms),
                    n_images=n, n_persons=total_persons,
                    batch_counter=self._batch_counter,
                )

        # ── Periodic CUDA cache flush ─────────────────────────────
        self._batch_counter += 1
        if self._batch_counter % _CUDA_CACHE_FLUSH_INTERVAL == 0:
            t_flush = time.time()
            torch.cuda.empty_cache()
            flush_ms = (time.time() - t_flush) * 1000
            logger.info("cuda_cache_flushed",
                batch_counter=self._batch_counter,
                flush_ms=round(flush_ms, 1),
            )

        return results

    # ──────────────────────────────────────────────────────────────
    # Batched Bib Pipeline (bib detect → text detect → OCR)
    # ──────────────────────────────────────────────────────────────

    def _batched_bib_pipeline(
        self, person_crops: List[np.ndarray], crop_masks: List[Optional[np.ndarray]] = None
    ) -> List[List[BibDetection]]:
        """
        Detect bibs, text regions, and OCR across ALL person crops in
        batched passes instead of one-by-one.

        Returns: list-of-lists, one BibDetection list per person crop.
        """
        n = len(person_crops)
        result_bibs: List[List[BibDetection]] = [[] for _ in range(n)]

        # ── Phase A: resize all crops to 640×640 for bib detection ──
        # YOLO bib was trained on squash-stretched person crops (plain 640×640
        # resize, no letterbox padding), so we must feed it the same.
        resized_crops = [
            cv2.resize(c, (640, 640), interpolation=cv2.INTER_LINEAR)
            for c in person_crops
        ]
        # Scale factors for mapping detected bib coords back to original crop space.
        crop_scales_x = [c.shape[1] / 640.0 for c in person_crops]
        crop_scales_y = [c.shape[0] / 640.0 for c in person_crops]

        # ── Phase B: bib YOLO — native TRT batched detection ─────────
        # _YOLODetector.detect_batch() handles chunking to MAX_BATCH=32
        # internally and returns List[np.ndarray] of (K,6) per image.
        bib_batch_results = self.yolo_bib.detect_batch(resized_crops, conf=0.1)

        # ── Phase C: collect all bib crops and text detect them ─────
        # bib_crops_info: (person_idx, bib_box_in_resized, bib_crop)
        bib_crops = []
        bib_crops_info = []  # (person_idx, bib_box_xyxy)

        # Group raw bib detections per person crop so we can pick the
        # best-overlapping bib when multiple are detected.
        per_crop_bibs: Dict[int, list] = {}   # pi → [(x1_640, y1_640, x2_640, y2_640, orig_xyxy)]
        for pi, dets in enumerate(bib_batch_results):
            if dets is None or len(dets) == 0:
                continue
            for det in dets:
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                orig_x1 = int(x1 * crop_scales_x[pi])
                orig_y1 = int(y1 * crop_scales_y[pi])
                orig_x2 = int(x2 * crop_scales_x[pi])
                orig_y2 = int(y2 * crop_scales_y[pi])
                per_crop_bibs.setdefault(pi, []).append(
                    (x1, y1, x2, y2, (orig_x1, orig_y1, orig_x2, orig_y2))
                )

        _BIB_MIN_MASK_OVERLAP = 0.25   # reject bib if <25% on person mask

        for pi, candidates in per_crop_bibs.items():
            # Score every bib detection against the person's seg mask.
            # - Multi-bib: keep only the best-overlapping bib.
            # - Single-bib: reject it if overlap is below threshold
            #   (the bib belongs to a background runner, not this person).
            if crop_masks and crop_masks[pi] is not None:
                mask = crop_masks[pi]
                scored = []
                for (bx1, by1, bx2, by2, orig_xyxy) in candidates:
                    ox1, oy1, ox2, oy2 = orig_xyxy
                    # Clamp to mask bounds
                    mh, mw = mask.shape[:2]
                    mx1 = max(0, min(ox1, mw - 1))
                    my1 = max(0, min(oy1, mh - 1))
                    mx2 = max(0, min(ox2, mw))
                    my2 = max(0, min(oy2, mh))
                    if mx2 <= mx1 or my2 <= my1:
                        scored.append((0.0, bx1, by1, bx2, by2, orig_xyxy))
                        continue
                    region = mask[my1:my2, mx1:mx2]
                    total_pixels = region.size
                    overlap = float(region.sum()) / total_pixels if total_pixels > 0 else 0.0
                    scored.append((overlap, bx1, by1, bx2, by2, orig_xyxy))
                # Keep only the best-overlapping bib
                scored.sort(key=lambda s: s[0], reverse=True)
                # Reject all bibs below the minimum overlap threshold
                candidates = [
                    (s[1], s[2], s[3], s[4], s[5]) for s in scored
                    if s[0] >= _BIB_MIN_MASK_OVERLAP
                ][:1]
                if not candidates:
                    logger.debug("bib_rejected_off_mask",
                        person_idx=pi,
                        best_overlap=round(scored[0][0], 3) if scored else 0.0,
                    )
                    continue

            for (x1, y1, x2, y2, orig_xyxy) in candidates:
                # Pad bib box to capture clipped leading/trailing digits
                bw = x2 - x1
                pad = int(bw * BIB_CROP_PAD_FRAC)
                x1, y1 = max(0, x1 - pad), max(0, y1)
                x2, y2 = min(640, x2 + pad), min(640, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                bib_crop = resized_crops[pi][y1:y2, x1:x2].copy()
                bib_crops.append(bib_crop)
                bib_crops_info.append((pi, orig_xyxy))

        if not bib_crops:
            return result_bibs

        # ── Phase D: stretch bib crops to 320×320 for text detection TRT ─
        text_inputs = [
            cv2.resize(bc, (320, 320), interpolation=cv2.INTER_LINEAR)
            for bc in bib_crops
        ]

        # _YOLODetector.detect_batch() handles chunking internally.
        text_batch_results = self.yolo_text.detect_batch(text_inputs, conf=0.5)

        # ── Phase E: extract OCR crop (text region or full bib) ─────
        ocr_crops = []
        ocr_map = []  # (index into bib_crops)

        for bi, (bib_crop, text_det) in enumerate(zip(bib_crops, text_batch_results)):
            ocr_crop = bib_crop  # fallback: full bib
            if text_det is not None and len(text_det) > 0:
                tx1, ty1, tx2, ty2 = int(text_det[0][0]), int(text_det[0][1]), int(text_det[0][2]), int(text_det[0][3])
                # Scale from 320 back to bib_crop dimensions
                bh, bw = bib_crop.shape[:2]
                tx1 = int(tx1 * bw / 320)
                ty1 = int(ty1 * bh / 320)
                tx2 = int(tx2 * bw / 320)
                ty2 = int(ty2 * bh / 320)
                # Pad text region on both axes to prevent digit clipping.
                # PARSeq is a Vision Transformer — a tight crop that shears off
                # the top of a '5'/'7' or the bottom curve of an '8'/'0' will
                # cause a full decode failure or hallucinate a different digit.
                tw = tx2 - tx1
                th = ty2 - ty1
                tpad_x = int(tw * BIB_CROP_PAD_FRAC)
                tpad_y = int(th * BIB_CROP_PAD_FRAC)
                tx1 = max(0, tx1 - tpad_x)
                tx2 = min(bw, tx2 + tpad_x)
                ty1 = max(0, ty1 - tpad_y)
                ty2 = min(bh, ty2 + tpad_y)
                if tx2 > tx1 and ty2 > ty1:
                    ocr_crop = bib_crop[ty1:ty2, tx1:tx2].copy()
            ocr_crops.append(ocr_crop)
            ocr_map.append(bi)

        # ── Phase F: batched OCR on all OCR crops ───────────────────
        if ocr_crops:
            ocr_results = self.ocr_model.predict(ocr_crops)

            for oi, ocr_out in enumerate(ocr_results):
                bi = ocr_map[oi]
                pi, bib_box = bib_crops_info[bi]

                text = ocr_out.get('text', '')
                ocr_conf = ocr_out.get('confidence', 0.0)

                logger.debug("raw_ocr_output",
                    person_crop_idx=pi, bbox=list(bib_box),
                    raw_text=text, confidence=round(ocr_conf, 4),
                )

                digits = re.findall(r'\d+', text)
                if digits:
                    bib_number = max(digits, key=len)
                    if (len(bib_number) >= detection_settings.MIN_BIB_DIGITS
                            and ocr_conf >= detection_settings.MIN_OCR_CONF):
                        result_bibs[pi].append(BibDetection(
                            bib_number=bib_number,
                            confidence=ocr_conf,
                            bbox=bib_box,
                        ))

        return result_bibs


# ──────────────────────────────────────────────────────────────────
# Image Loading / Preprocessing Helpers
# ──────────────────────────────────────────────────────────────────


def _load_image(path: str):
    """Load an image. Returns (bgr, h, w) or None."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (img, h, w)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Global Singleton
# ──────────────────────────────────────────────────────────────────

_engine = None


def get_engine(device: str = "cuda:0", profile: str = PROFILE_PROBE) -> InferenceEngine:
    """
    Get or create the global inference engine instance.

    Args:
        device: CUDA device string
        profile: 'probe' for detection+OCR only, 'full' to include ReID+Face
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine(device=device)
    _engine.load_models(profile=profile)
    return _engine
