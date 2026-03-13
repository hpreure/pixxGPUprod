"""
pixxEngine Model Wrappers
==========================
Wrappers for active AI models used by the inference pipeline:
  - InsightFaceWrapper  — face embedding (ONNX-RT + TRT EP, buffalo_l)
  - ParseqWrapper       — OCR (PARSeq NAR, TRT FP16)

Removed: TransReIDWrapper (OSNet/ResNet50 legacy ReID).
ReID is now handled exclusively by src/reid_wrapper.py (DINOv2 TRT FP16).
"""

import torch
import numpy as np
import cv2
import logging
from typing import List, Union, Optional
from pathlib import Path

from src.detection_config import settings

# Configure logging
logger = logging.getLogger(__name__)

# NOTE: TransReIDWrapper (OSNet/ResNet50 legacy ReID) was removed.
# The pipeline exclusively uses DINOv2 TRT FP16 via src/reid_wrapper.py (ReIDWrapper).


class InsightFaceWrapper:
    """
    Wrapper for InsightFace face recognition model.
    Extracts face embeddings for identity matching.
    
    OPTIMIZATION: Uses batched detection and recognition for GPU efficiency.
    The ONNX execution provider processes crops in parallel when given as a batch.
    """
    
    def __init__(self, model_name: str = 'buffalo_l', device: str = None):
        """
        Initialize InsightFace model.
        
        Args:
            model_name: Model pack name (buffalo_l, buffalo_s, etc.)
            device: Compute device
        """
        self.device = device or settings.INFERENCE_DEVICE
        self.model_name = model_name
        self.app = None
        self.model = None  # For compatibility with .half() calls
        self._load_model()
    
    def _load_model(self):
        """Load InsightFace model with TensorRT optimization for Blackwell.
        
        IMPORTANT: This will FAIL if GPU execution providers are not available.
        No silent CPU fallback - GPU is required for acceptable performance.
        
        Blackwell supports: FP32, TF32, FP16, BF16, INT8, FP8
        TensorRT with FP16 provides best speed/accuracy tradeoff for face recognition.
        """
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            import os

            # Pre-load TRT shared libs so ONNX Runtime's TRT EP can dlopen
            # libnvinfer.so.10.  Without this, LD_LIBRARY_PATH must already
            # include the tensorrt_libs directory (it is set in the service
            # unit file, but may not be set in ad-hoc script invocations).
            from src.reid_wrapper import ReIDWrapper as _R
            _R._ensure_trt_libs(_R.__new__(_R))

            # Check available ONNX Runtime providers - FAIL if no GPU support
            available_providers = ort.get_available_providers()
            has_cuda = 'CUDAExecutionProvider' in available_providers
            has_trt = 'TensorrtExecutionProvider' in available_providers
            
            if not has_cuda:
                raise RuntimeError(
                    f"FATAL: CUDAExecutionProvider not available in onnxruntime! "
                    f"Available: {available_providers}. "
                    f"Install onnxruntime-gpu: pip install onnxruntime-gpu"
                )
            
            if not has_trt:
                raise RuntimeError(
                    f"FATAL: TensorrtExecutionProvider not available! "
                    f"Available: {available_providers}. "
                    f"Ensure LD_LIBRARY_PATH includes tensorrt_libs and CUDA libs."
                )
            
            logger.info(f"ONNX Runtime providers: CUDA={has_cuda}, TensorRT={has_trt}")
            
            # Determine GPU ID from device string
            gpu_id = 0
            if 'cuda' in self.device:
                try:
                    gpu_id = int(self.device.split(':')[1])
                except (IndexError, ValueError):
                    gpu_id = 0
            
            # Local model directories (vendored under project weights/)
            # These must be populated by scripts/download_model_weights.py
            _insightface_root = Path(__file__).parent.parent / "weights" / "insightface"
            _pack_dir = _insightface_root / "models" / self.model_name
            if not _pack_dir.exists():
                raise FileNotFoundError(
                    f"InsightFace '{self.model_name}' ONNX pack not found: {_pack_dir}\n"
                    f"Run:  pixxEngine_venv/bin/python3 scripts/download_model_weights.py"
                )

            # TensorRT engine cache: project-local, visible to rebuild script
            trt_cache_dir = str(Path(__file__).parent.parent / "weights" / "insightface_trt")
            os.makedirs(trt_cache_dir, exist_ok=True)
            
            # Build provider list - TensorRT primary, CUDA fallback, NO CPU
            # Blackwell TensorRT with FP16 for optimal performance
            # VRAM OPTIMIZED: Reduced workspace for 2-worker setup
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': gpu_id,
                    'trt_max_workspace_size': 134217728,   # 128MB workspace (inference-only, reduced from 256MB)
                    'trt_fp16_enable': True,              # FP16 precision for Blackwell
                    'trt_int8_enable': False,             # INT8 requires calibration data
                    'trt_engine_cache_enable': True,      # Cache compiled engines
                    'trt_engine_cache_path': trt_cache_dir,
                    'trt_timing_cache_enable': True,      # Cache kernel timing
                    'trt_timing_cache_path': trt_cache_dir,
                    'trt_builder_optimization_level': 5,  # Max optimization (slower build, faster inference)
                    'trt_auxiliary_streams': 4,           # Parallel streams for Blackwell
                    'trt_dla_enable': False,              # No DLA on consumer GPUs
                }),
                ('CUDAExecutionProvider', {
                    'device_id': gpu_id,
                    'arena_extend_strategy': 'kSameAsRequested',  # Prevent memory hoarding (was kNextPowerOfTwo)
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit (safe for 2 workers)
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'cudnn_conv_use_max_workspace': '1',
                    'use_tf32': '1',  # TF32 fallback for ops not supported by TRT
                })
            ]
            
            # DO NOT add CPUExecutionProvider - we want to fail if GPU doesn't work
            
            logger.info(f"Initializing InsightFace with TensorRT FP16 (cache: {trt_cache_dir})")
            logger.info("First run will be slow while TensorRT builds optimized engines...")
            
            # Initialize FaceAnalysis from local vendored ONNX pack
            self.app = FaceAnalysis(
                name=self.model_name,
                root=str(_insightface_root),
                providers=providers
            )
            # Use 640x640 detection size for better accuracy on person crops
            self.app.prepare(ctx_id=gpu_id, det_size=(640, 640))
            
            # Verify models are actually using GPU by checking their providers
            for name, model in self.app.models.items():
                if hasattr(model, 'session'):
                    actual_providers = model.session.get_providers()
                    if actual_providers == ['CPUExecutionProvider']:
                        raise RuntimeError(
                            f"FATAL: InsightFace model '{name}' fell back to CPU! "
                            f"This indicates a GPU/TensorRT initialization failure. "
                            f"Check LD_LIBRARY_PATH includes tensorrt_libs."
                        )
                    # Log which provider is actually being used
                    provider_name = actual_providers[0] if actual_providers else 'Unknown'
                    logger.info(f"  {name}: {provider_name}")
            
            # Store reference for half() compatibility
            self.model = self.app
            self.gpu_id = gpu_id
            
            logger.info(f"InsightFace '{self.model_name}' loaded with TensorRT FP16 on GPU {gpu_id} ✓")
            
        except Exception as e:
            logger.error(f"FATAL: Failed to load InsightFace on GPU: {e}")
            raise
    
    @staticmethod
    def _face_mask_overlap(face, mask: np.ndarray) -> float:
        """Fraction of the face bbox area that overlaps the person seg mask."""
        fx1, fy1, fx2, fy2 = (int(round(v)) for v in face.bbox[:4])
        h, w = mask.shape[:2]
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, fx2), min(h, fy2)
        if fx2 <= fx1 or fy2 <= fy1:
            return 0.0
        region = mask[fy1:fy2, fx1:fx2]
        face_area = (fy2 - fy1) * (fx2 - fx1)
        return float(region.sum()) / face_area if face_area > 0 else 0.0

    def extract_batch(
        self,
        crops: List[np.ndarray],
        seg_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[dict]:
        """
        Batched face extraction - processes all crops efficiently.
        
        InsightFace's detection model (RetinaFace) works best with single images,
        but the recognition model (ArcFace) can process embeddings in parallel.
        
        This method:
        1. Runs face detection on each crop (unavoidable per-image)
        2. Batches all detected face crops for recognition (GPU parallel)
        
        Args:
            crops: List of person crop images (BGR, numpy arrays)
            seg_masks: Optional list of binary masks (same H×W as each crop).
                       When provided for a multi-face crop, the face with the
                       highest mask overlap ratio is selected instead of the
                       largest face.  This eliminates background-person faces.
        
        Returns:
            List of dicts with 'embedding', 'landmarks', 'quality', 'bbox' keys
        """
        if not crops:
            return []
        
        _MIN_OVERLAP = 0.25  # reject face if best overlap is below this

        # Results placeholder (maintains order)
        results = [None] * len(crops)
        
        # Phase 1: Detection pass (collect faces and their indices)
        face_indices = []  # Track which crop index each face belongs to
        detected_faces = []  # The face objects from InsightFace
        
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                results[i] = self._empty_result()
                continue
            
            try:
                # Run detection on single crop
                faces = self.app.get(crop)
                
                if faces:
                    mask_i = seg_masks[i] if seg_masks and i < len(seg_masks) else None

                    if len(faces) == 1:
                        best_face = faces[0]
                    elif mask_i is not None:
                        # ── Seg-mask overlap selection (Option C) ─────────
                        # Score each face by the fraction of its bbox that
                        # falls on mask=1 pixels (this person's instance).
                        scored = [(f, self._face_mask_overlap(f, mask_i)) for f in faces]
                        scored.sort(key=lambda t: t[1], reverse=True)
                        best_face, best_overlap = scored[0]

                        if best_overlap < _MIN_OVERLAP:
                            # No face convincingly belongs to this person —
                            # treat as faceless to avoid contamination.
                            logger.info("multi_face_all_off_mask", extra={
                                "crop_idx": i,
                                "face_count": len(faces),
                                "best_overlap": round(best_overlap, 3),
                            })
                            results[i] = self._empty_result()
                            continue

                        logger.debug("multi_face_mask_select", extra={
                            "crop_idx": i,
                            "face_count": len(faces),
                            "best_overlap": round(best_overlap, 3),
                            "runner_up_overlap": round(scored[1][1], 3) if len(scored) > 1 else 0.0,
                        })
                    else:
                        # Fallback: no mask available — use area (legacy)
                        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        logger.debug("multi_face_in_crop", extra={
                            "crop_idx": i,
                            "face_count": len(faces),
                            "selected_area": round(float((best_face.bbox[2] - best_face.bbox[0]) * (best_face.bbox[3] - best_face.bbox[1])), 1),
                        })

                    face_indices.append(i)
                    detected_faces.append(best_face)
                else:
                    # No face found in this crop
                    results[i] = self._empty_result()
                    
            except Exception as e:
                logger.warning(f"Face detection failed for crop {i}: {e}")
                results[i] = self._empty_result()
        
        # Phase 2: Map results back (embeddings already computed by app.get())
        # InsightFace computes embeddings during get() call, so we just need to collect
        # FIX: L2 normalize embeddings to ensure consistent similarity calculations
        for idx, face in zip(face_indices, detected_faces):
            # L2 normalize embedding (critical for angle-invariant matching)
            embedding = face.embedding
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm  # Unit vector
            
            results[idx] = {
                'embedding': embedding,  # 512-dim L2-normalized vector
                'landmarks': face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None,
                'quality': float(face.det_score),
                'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
            }
        
        return results
    
    def extract(
        self,
        face_crops: List[np.ndarray],
        seg_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[dict]:
        """
        Extract face embeddings and landmarks.
        
        Args:
            face_crops: List of face crop images (BGR, numpy arrays)
            seg_masks: Optional per-crop binary masks for face ownership filtering.
        
        Returns:
            List of dicts with 'embedding', 'landmarks', 'quality' keys
        """
        # Delegate to optimized batch method
        return self.extract_batch(face_crops, seg_masks=seg_masks)
    
    def _empty_result(self) -> dict:
        """Return empty result dict for faces not detected."""
        return {
            'embedding': None,
            'landmarks': None,
            'quality': 0.0,
            'bbox': None,
        }
    
    def half(self):
        """Compatibility method for FP16 (handled internally by InsightFace ONNX)."""
        logger.debug("InsightFace half() called - handled internally by ONNX EP")
        return self


class _PARSeqNARFlat(torch.nn.Module):
    """
    Flat NAR-only PARSeq wrapper for ONNX / TRT compilation.

    Bypasses the Lightning wrapper (``PARSeq.forward``) and the
    tokenizer argument entirely.  Pre-computes the BOS embedding
    and position-query slice as static buffers so the forward pass
    contains:

        encode → repeat pos_queries → repeat bos_emb → decoder → head

    No branching, no empty-tensor ``cat``, no ``expand(-1)`` — all of
    which are incompatible with the TRT converter on Blackwell.
    """

    def __init__(self, parseq_inner, bos_id: int, num_steps: int = 9):
        super().__init__()
        self.encoder = parseq_inner.encoder
        self.decoder = parseq_inner.decoder
        self.head    = parseq_inner.head
        # Fixed position queries for num_steps output slots
        self.register_buffer(
            'pos_queries',
            parseq_inner.pos_queries[:, :num_steps].clone(),  # (1, 9, 384)
        )
        # Pre-computed BOS embedding — avoids text_embed + torch.full in forward
        device = next(parseq_inner.parameters()).device
        bos_tok = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            bos_emb = parseq_inner.text_embed(bos_tok)  # (1, 1, 384)
        self.register_buffer('bos_emb', bos_emb)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bs = images.shape[0]
        memory = self.encoder(images)                  # (bs, 128, 384)
        pos_q  = self.pos_queries.repeat(bs, 1, 1)     # (bs,  9, 384)
        tgt    = self.bos_emb.repeat(bs, 1, 1)          # (bs,  1, 384)
        decoded = self.decoder(pos_q, tgt, memory, None, None, None)
        return self.head(decoded)                       # (bs,  9, 95)


class ParseqWrapper:
    """
    Wrapper for PARSeq OCR model.
    Performs text recognition on bib number crops.

    Blackwell Optimization: Compiled with torch-TensorRT FP16 in NAR
    (parallel) decoding mode.  All 9 output positions are decoded in a
    single fused kernel — no Python AR loop, full FP16 end-to-end.
    """

    # Maximum bib-number length (digits only).  Bibs are 1–6 chars;
    # 8 gives a comfortable margin and keeps the sequence short.
    BIB_MAX_LENGTH = 8

    _TRT_CACHE_DIR    = Path(__file__).parent.parent / "weights"
    _TRT_ENGINE_FILE = str(_TRT_CACHE_DIR / "parseq_nar_trt_fp16.engine")
    _ONNX_FILE       = str(_TRT_CACHE_DIR / "parseq_nar_fp16.onnx")
    _TRT_INPUT_NAME  = 'images'
    _TRT_OUTPUT_NAME = 'logits'
    
    def __init__(self, model_name: str = 'parseq', device: str = None):
        """
        Initialize PARSeq OCR model.

        Args:
            model_name: Model name
            device: Compute device
        """
        self.device = device or settings.INFERENCE_DEVICE
        self.model_name = model_name
        self.model = None          # hub model (kept for tokenizer)
        self._trt_engine = None    # native TRT ICudaEngine
        self._trt_context = None   # native TRT IExecutionContext
        self._compiled = False
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load PARSeq model from torch hub and compile with TensorRT."""
        try:
            from torchvision import transforms as T

            self.model = torch.hub.load(
                'baudm/parseq',
                'parseq',
                pretrained=True,
                trust_repo=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # ── Enable NAR (parallel) decoding ───────────────────────────
            # Disables the character-by-character AR loop. The entire
            # forward becomes: encode → single decode step → head.
            # Required for torch-TRT graph compilation.
            self.model.model.decode_ar = False

            self.transform = None   # preprocessing done in _preprocess()
            logger.info(f"PARSeq OCR model loaded on {self.device} (NAR mode)")

            # ── Attempt TRT FP16 compilation ─────────────────────────────
            if 'cuda' in self.device:
                self._compile_tensorrt()

        except Exception as e:
            logger.warning(f"Failed to load PARSeq from hub: {e}, using fallback")
            self._load_fallback()
    
    def _ensure_trt_libs(self):
        """
        Ensure all shared libraries needed by torch-TensorRT are loaded.
        Mirrors reid_wrapper._ensure_trt_libs() but also includes
        tensorrt_libs (libnvinfer.so.10) which is a venv-installed package.
        """
        import os, site, ctypes
        sp = (site.getsitepackages() or [os.path.join(os.path.dirname(torch.__file__), "..")])[0]
        extra = [
            os.path.join(sp, "torch",         "lib"),
            os.path.join(sp, "nvidia",        "cu13", "lib"),
            os.path.join(sp, "nvidia",        "cuda_runtime", "lib"),
            os.path.join(sp, "tensorrt_libs"),                 # libnvinfer.so.10
        ]
        cur  = os.environ.get("LD_LIBRARY_PATH", "")
        miss = [p for p in extra if os.path.isdir(p) and p not in cur]
        if miss:
            os.environ["LD_LIBRARY_PATH"] = ":".join(miss) + ":" + cur
        # Eagerly dlopen every .so so the linker finds them as RTLD_GLOBAL
        for p in extra:
            if not os.path.isdir(p):
                continue
            for lib in sorted(os.listdir(p)):
                if lib.endswith(".so") or ".so." in lib:
                    try:
                        ctypes.CDLL(os.path.join(p, lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    def _compile_tensorrt(self):
        """
        Build (or reload) a native TRT FP16 engine for PARSeq-NAR.

        Pipeline:
            1.  If a cached ``.engine`` exists → deserialize in <100 ms.
            2.  Otherwise: build a flat NAR ``nn.Module`` → ONNX export
                → TRT builder (FP16, dynamic batch 1-32) → serialize.

        This uses the **native TensorRT** C-API (``tensorrt.Builder``)
        instead of ``torch_tensorrt.compile()`` because the latter's
        dynamo-IR and TorchScript-IR backends both fail on PARSeq's
        multi-head-attention reshapes on Blackwell SM 12.0 (TRT 10.14).

        Fallback: on any failure, FP32 eager-mode NAR is used instead.
        """
        import os, copy
        try:
            self._ensure_trt_libs()
            import tensorrt as trt

            os.makedirs(self._TRT_CACHE_DIR, exist_ok=True)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # ── Try loading cached engine ─────────────────────────────
            if os.path.isfile(self._TRT_ENGINE_FILE):
                try:
                    t0 = __import__('time').time()
                    runtime = trt.Runtime(TRT_LOGGER)
                    with open(self._TRT_ENGINE_FILE, 'rb') as f:
                        self._trt_engine = runtime.deserialize_cuda_engine(f.read())
                    self._trt_context = self._trt_engine.create_execution_context()
                    self._compiled = True
                    logger.info(
                        "PARSeq TRT engine loaded from cache in %.3fs ✓ (%s)",
                        __import__('time').time() - t0, self._TRT_ENGINE_FILE,
                    )
                    return
                except Exception as cache_err:
                    logger.warning("Cached PARSeq TRT engine invalid (%s), rebuilding…", cache_err)
                    self._trt_engine = self._trt_context = None
                    try:
                        os.remove(self._TRT_ENGINE_FILE)
                    except OSError:
                        pass

            # ── Step 1: flat NAR module → ONNX ────────────────────────
            logger.info("Building PARSeq-NAR TRT FP16 engine (~20 s) …")
            inner = self.model.model  # The actual PARSeq nn.Module
            bos_id = self.model.tokenizer.bos_id
            num_steps = self.BIB_MAX_LENGTH + 1  # 9

            flat = _PARSeqNARFlat(
                copy.deepcopy(inner), bos_id, num_steps
            ).eval().to(self.device)          # FP32 — let TRT decide where to use FP16

            onnx_path = self._ONNX_FILE
            dummy = torch.randn(8, 3, 32, 128, device=self.device)  # FP32 dummy
            torch.onnx.export(
                flat, dummy, onnx_path,
                input_names=[self._TRT_INPUT_NAME],
                output_names=[self._TRT_OUTPUT_NAME],
                dynamic_axes={
                    self._TRT_INPUT_NAME:  {0: 'batch'},
                    self._TRT_OUTPUT_NAME: {0: 'batch'},
                },
                opset_version=18,
                do_constant_folding=True,
            )
            del flat  # free copy

            # ── Step 2: ONNX → TRT FP16 engine ───────────────────────
            # Export is FP32 ONNX; TRT's FP16 flag applies selective
            # FP16 only to matmul-class ops while keeping LayerNorm and
            # softmax (attention) in FP32.  This avoids the ~10 logit
            # error seen when exporting from a .half() module.
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors):
                    logger.error("TRT ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("TRT ONNX parse failed")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MB
            config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()
            profile.set_shape(
                self._TRT_INPUT_NAME,
                (1,  3, 32, 128),   # min
                (8,  3, 32, 128),   # opt
                (32, 3, 32, 128),   # max
            )
            config.add_optimization_profile(profile)

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TRT engine build returned None")

            engine_data = bytes(serialized)
            with open(self._TRT_ENGINE_FILE, 'wb') as f:
                f.write(engine_data)
            logger.info(
                "PARSeq TRT FP16 engine built → %s (%.1f MB)",
                self._TRT_ENGINE_FILE, len(engine_data) / 1e6,
            )

            # ── Step 3: load the freshly-built engine ─────────────────
            runtime = trt.Runtime(TRT_LOGGER)
            self._trt_engine  = runtime.deserialize_cuda_engine(engine_data)
            self._trt_context = self._trt_engine.create_execution_context()
            self._compiled = True
            logger.info("PARSeq TRT FP16 engine ready ✓")

            # Clean up intermediate ONNX files
            for p in [onnx_path, onnx_path + '.data']:
                try:
                    os.remove(p)
                except OSError:
                    pass

        except ImportError:
            logger.warning(
                "TensorRT not available — PARSeq running as FP32 NAR eager mode"
            )
        except Exception as e:
            logger.warning(
                "PARSeq TRT build failed (%s) — falling back to FP32 NAR eager mode", e
            )

    def _load_fallback(self):
        """Fallback OCR using EasyOCR or basic approach."""
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu='cuda' in self.device)
            self.model = self.reader  # For half() compatibility
            logger.info("Fallback EasyOCR loaded")
        except ImportError:
            logger.warning("EasyOCR not available, OCR will return empty results")
            self.model = None
    
    def _preprocess(self, crops: List[np.ndarray], fp16: bool = False) -> torch.Tensor:
        """
        Preprocess crops for PARSeq using OpenCV + PyTorch (no PIL).
        PARSeq expects (B, 3, 32, 128), values in [-1, 1].
        """
        processed = []
        for crop in crops:
            resized = cv2.resize(crop, (128, 32), interpolation=cv2.INTER_LINEAR)
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor  = (tensor - 0.5) / 0.5
            processed.append(tensor)

        batch = torch.stack(processed).to(self.device)
        if fp16:
            batch = batch.half()
        return batch
    
    @torch.no_grad()
    def predict(self, crops: List[np.ndarray]) -> List[dict]:
        """
        Perform OCR on bib-number crops.

        Hot path (TRT FP16 compiled):
            batch → FP16 → _trt_model (single fused NAR kernel) → softmax → decode

        Fallback (FP32 NAR eager):
            batch → model(max_length=BIB_MAX_LENGTH) → softmax → decode
        """
        if not crops:
            return []

        if hasattr(self, 'reader'):
            return self._predict_easyocr(crops)

        if self.model is None and self._trt_engine is None:
            return [{'text': '', 'confidence': 0.0} for _ in crops]

        results = []
        try:
            if self._compiled and self._trt_context is not None:
                # ── Native TRT FP16 path ──────────────────────────────
                # Engine profile supports batch 1-32.  Chunk any larger
                # inputs so we never exceed the max profile size.
                TRT_MAX_BATCH = 32
                all_logits = []
                for chunk_start in range(0, len(crops), TRT_MAX_BATCH):
                    chunk = crops[chunk_start:chunk_start + TRT_MAX_BATCH]
                    batch = self._preprocess(chunk, fp16=False)
                    bs    = batch.shape[0]
                    out_buf = torch.empty(bs, self.BIB_MAX_LENGTH + 1, 95,
                                          dtype=torch.float32, device=self.device)
                    self._trt_context.set_input_shape(self._TRT_INPUT_NAME,
                                                      (bs, 3, 32, 128))
                    self._trt_context.set_tensor_address(self._TRT_INPUT_NAME,
                                                         batch.data_ptr())
                    self._trt_context.set_tensor_address(self._TRT_OUTPUT_NAME,
                                                         out_buf.data_ptr())
                    self._trt_context.execute_async_v3(
                        torch.cuda.current_stream().cuda_stream)
                    torch.cuda.synchronize()
                    all_logits.append(out_buf)
                logits = torch.cat(all_logits, dim=0)
            else:
                # ── FP32 NAR eager path ───────────────────────────────
                batch  = self._preprocess(crops, fp16=False)
                logits = self.model(batch, max_length=self.BIB_MAX_LENGTH)

            pred = logits.softmax(-1)
            pred_str, pred_conf = self.model.tokenizer.decode(pred)

            for text, conf in zip(pred_str, pred_conf):
                results.append({
                    'text': text,
                    'confidence': float(conf.mean()) if hasattr(conf, 'mean') else float(conf)
                })

        except Exception as e:
            logger.error(f"PARSeq prediction error: {e}")
            results = [{'text': '', 'confidence': 0.0} for _ in crops]

        return results
    
    def _predict_easyocr(self, crops: List[np.ndarray]) -> List[dict]:
        """Fallback OCR using EasyOCR."""
        results = []
        for crop in crops:
            try:
                ocr_results = self.reader.readtext(crop)
                if ocr_results:
                    # Combine all detected text
                    text = ' '.join([r[1] for r in ocr_results])
                    conf = sum([r[2] for r in ocr_results]) / len(ocr_results)
                    results.append({'text': text, 'confidence': conf})
                else:
                    results.append({'text': '', 'confidence': 0.0})
            except Exception as e:
                logger.warning(f"EasyOCR error: {e}")
                results.append({'text': '', 'confidence': 0.0})
        return results
    
    def half(self):
        """No-op when TRT engine is active; FP16 is handled internally."""
        if self._compiled:
            logger.debug("PARSeq half() called — TRT FP16 already active")
            return self
        if self.model is not None and hasattr(self.model, 'half'):
            self.model = self.model.half()
            logger.debug("PARSeq model converted to FP16")
        return self
