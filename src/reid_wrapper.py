import torch
import torch.nn.functional as F
import logging
import numpy as np
from pathlib import Path
from typing import List, Union
from torchvision import transforms
import cv2
import os
from src.transreid_model import make_transreid_model, load_transreid_weights

logger = logging.getLogger(__name__)


class ReIDWrapper:
    """
    Production-Ready ReID using TransReID (ViT-B/16).
    
    Source: Shuting He et al. (ICCV 2021), MSMT17 checkpoint
    Architecture: ViT-B/16 with overlapping stride-12 patches, 768-dim CLS token
    Input: 256×128 (H×W) — standard person ReID resolution
    
    Blackwell Optimization: Uses torch-TensorRT for FP16 compilation on RTX 5070.
    """
    
    def __init__(self, device: str = 'cuda', compile_tensorrt: bool = True):
        self.device = device
        self._compiled = False
        
        logger.info("Loading ReID Model: TransReID ViT-B/16 (MSMT17)...")
        
        # ── Fail fast if weights are missing ────────────────────────
        _weights_path = Path(self._TRT_CACHE_DIR) / "transreid_vit_b16_msmt17.pth"
        if not _weights_path.exists():
            raise FileNotFoundError(
                f"TransReID weights not found: {_weights_path}\n"
                f"Run:  pixxEngine_venv/bin/python3 scripts/download_model_weights.py"
            )
        
        self.model = make_transreid_model()
        load_transreid_weights(self.model, str(_weights_path))
        self.model.to(device)
        self.model.eval()
        
        # TransReID native resolution: 256×128 (H×W)
        self.input_h = 256
        self.input_w = 128
        
        # Standard ImageNet normalization (TransReID expects this)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # GPU normalization constants for zero-copy tensor pipeline
        self.mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std_gpu = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        # Blackwell TensorRT Optimization
        if compile_tensorrt and 'cuda' in device:
            self._compile_tensorrt()
        
        logger.info("ReID Model loaded successfully ✓ (TransReID 256×128)")

    # Path for the pre-compiled TRT engine (main project weights/ dir — not src/weights/)
    _TRT_CACHE_DIR  = str(Path(__file__).resolve().parent.parent / "weights")
    _TRT_CACHE_FILE = str(Path(__file__).resolve().parent.parent / "weights" / "transreid_vit_b16_trt_fp16.ep")

    def _ensure_trt_libs(self):
        """
        Add the pip-installed tensorrt_libs directory to LD_LIBRARY_PATH and
        load the shared objects into the process via ctypes so the OS linker
        can resolve libnvinfer.so.10 for ONNX Runtime's TRT execution provider.

        TensorRT is distributed as a pip package whose .so files live inside
        the venv at  site-packages/tensorrt_libs/  — a directory the OS linker
        never searches automatically.  Without this call, any library that does
        its own dlopen() for libnvinfer (e.g. onnxruntime TRT EP) will fail with
        "libnvinfer.so.10: cannot open shared object file".
        """
        import site, ctypes
        site_pkgs = site.getsitepackages() or []
        # Primary location: pip-installed tensorrt_libs package
        candidates = [os.path.join(sp, "tensorrt_libs") for sp in site_pkgs]
        # Fallback: torch's own lib dir (covers torch-TRT internal libs)
        candidates += [os.path.join(os.path.dirname(torch.__file__), "lib")]

        cur = os.environ.get("LD_LIBRARY_PATH", "")
        missing = [p for p in candidates if p not in cur and os.path.isdir(p)]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(missing) + (":" + cur if cur else "")

        # Pre-load the .so files so subsequent dlopen() calls resolve correctly
        for p in missing:
            for lib in sorted(os.listdir(p)):
                if lib.endswith(".so") or ".so." in lib:
                    try:
                        ctypes.CDLL(os.path.join(p, lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    def _compile_tensorrt(self):
        """
        Compile TransReID with torch-TensorRT for Blackwell FP16 optimization.

        Caches the compiled engine to disk so subsequent launches load
        in <1 s instead of re-compiling (~6 s).
        """
        try:
            self._ensure_trt_libs()
            import torch_tensorrt

            os.makedirs(self._TRT_CACHE_DIR, exist_ok=True)

            # ── Try loading a pre-compiled engine from disk ───────
            if os.path.isfile(self._TRT_CACHE_FILE):
                try:
                    t0 = __import__("time").time()
                    self.model = torch_tensorrt.load(self._TRT_CACHE_FILE).module()
                    self.model.to(self.device)
                    self._compiled = True
                    elapsed = __import__("time").time() - t0
                    logger.info(
                        "TransReID TRT engine loaded from cache in %.2fs ✓ (%s)",
                        elapsed, self._TRT_CACHE_FILE,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Cached TRT engine invalid (%s), rebuilding...", e
                    )
                    try:
                        os.remove(self._TRT_CACHE_FILE)
                    except OSError:
                        pass

            # ── Build fresh engine ────────────────────────────────
            logger.info("Compiling TransReID with torch-TensorRT (FP16) — first time only...")

            self.model.half()

            self.model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    min_shape=(1, 3, 256, 128),
                    opt_shape=(4, 3, 256, 128),
                    max_shape=(16, 3, 256, 128),
                    dtype=torch.half,
                )],
                enabled_precisions={torch.half},
                truncate_long_and_double=True,
                workspace_size=1 << 27,       # 128 MB
                use_explicit_typing=False,    # 2.10.0 default changed; keep FP16 mixed path
            )
            self._compiled = True
            logger.info("TransReID TRT compilation complete ✓")

            # ── Persist to disk for next launch ───────────────────
            # torch-TRT 2.10.0: retrace=False uses the internal torch-TRT
            # exporter which preserves dynamic optimization profiles.
            # retrace=True (default) re-traces with a static tensor, baking
            # in a shape guard and breaking dynamic batching.
            try:
                torch_tensorrt.save(
                    self.model,
                    self._TRT_CACHE_FILE,
                    retrace=False,
                )
                logger.info(
                    "TransReID TRT engine cached → %s (%.1f MB)",
                    self._TRT_CACHE_FILE,
                    os.path.getsize(self._TRT_CACHE_FILE) / 1e6,
                )
            except Exception as e:
                logger.warning("Failed to cache TRT engine to disk: %s", e)

        except ImportError:
            logger.warning("torch-TensorRT not available, using standard PyTorch eager mode")
        except Exception as e:
            logger.warning(f"TensorRT compilation failed: {e}, using standard PyTorch eager mode")

    def half(self):
        """Enable FP16 for RTX 5070 optimization."""
        if self._compiled:
            logger.info("ReID Model already compiled with TensorRT FP16")
            return
        self.model.half()
        # Update GPU constants to FP16
        self.mean_gpu = self.mean_gpu.half()
        self.std_gpu = self.std_gpu.half()
        logger.info("ReID Model converted to FP16")

    def extract(self, crops: List[Union[np.ndarray, torch.Tensor]]) -> List[np.ndarray]:
        """
        Extract features from a batch of crops.
        
        Blackwell Optimization: Accepts GPU tensors directly to avoid CPU roundtrip.
        
        Args:
            crops: List of crops - either numpy arrays (BGR, HWC) or 
                   GPU tensors (RGB, CHW, values 0-1)
        
        Returns:
            List of feature vectors (numpy arrays)
        """
        if not crops:
            return []
        
        # Detect input type from first valid crop
        is_tensor_input = isinstance(crops[0], torch.Tensor)
        
        if is_tensor_input:
            return self._extract_from_tensors(crops)
        else:
            return self._extract_from_numpy(crops)
    
    def _extract_from_tensors(self, crops: List[torch.Tensor]) -> List[np.ndarray]:
        """
        GPU-optimized extraction - no CPU roundtrip (Blackwell zero-copy).
        Direct resize to 256×128 and normalization entirely on GPU.
        
        CRITICAL: All tensors must be contiguous for CUDA kernels (Blackwell fix).
        UINT8 PIPELINE: Accepts uint8 tensors, converts to float JIT.
        """
        batch_tensors = []
        valid_indices = []
        
        for i, crop in enumerate(crops):
            if crop is None or crop.numel() == 0:
                continue
            try:
                # CRITICAL: Ensure input is contiguous (fixes cudaErrorIllegalAddress)
                if not crop.is_contiguous():
                    crop = crop.contiguous()
                
                # Convert uint8 to float 0-1
                if crop.dtype == torch.uint8:
                    crop = crop.float().div(255.0)
                
                # crop is (C, H, W) on GPU, values 0-1
                c, h, w = crop.shape
                if h < 1 or w < 1 or c != 3:
                    logger.warning(f"Invalid crop shape {crop.shape}, skipping")
                    continue
                
                # Direct resize to TransReID's native 256×128 (H×W)
                resized = F.interpolate(
                    crop.unsqueeze(0),
                    size=(self.input_h, self.input_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).contiguous()
                
                batch_tensors.append(resized)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Tensor crop resize failed: {e}")
        
        if not batch_tensors:
            return [None] * len(crops)
        
        # Stack into batch (all on GPU) - ensure contiguous
        batch = torch.stack(batch_tensors).contiguous()  # (B, C, 256, 128)
        
        # Normalize on GPU using precomputed constants
        batch = (batch - self.mean_gpu) / self.std_gpu
        batch = batch.contiguous()
        
        # Handle precision
        use_half = self._compiled or next(self.model.parameters()).dtype == torch.float16
        if use_half:
            batch = batch.half().contiguous()
        
        # Inference — chunk to stay within TensorRT max batch size (16)
        _TRT_MAX = 16
        with torch.no_grad():
            if batch.shape[0] <= _TRT_MAX:
                features = self.model(batch)
            else:
                chunks = [batch[i:i + _TRT_MAX] for i in range(0, batch.shape[0], _TRT_MAX)]
                features = torch.cat([self.model(c) for c in chunks], dim=0)
        
        # L2 Normalize (Critical for Cosine Similarity)
        features = F.normalize(features, p=2, dim=1)
        features = features.float().cpu().numpy()
        
        # Map back
        results = [None] * len(crops)
        for idx, feat in zip(valid_indices, features):
            results[idx] = feat
        
        return results
    
    def _extract_from_numpy(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        CPU path for numpy array inputs (legacy compatibility).
        """
        batch_tensors = []
        valid_indices = []
        
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            try:
                # Convert BGR (OpenCV) -> RGB
                crop_rgb = crop[:, :, ::-1].copy()
                
                # Direct resize to TransReID's native 256×128 (H×W)
                resized = cv2.resize(crop_rgb, (self.input_w, self.input_h),
                                     interpolation=cv2.INTER_LINEAR)
                
                # Normalize for TransReID
                t = self.normalize(resized)
                batch_tensors.append(t)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Crop preprocess failed: {e}")

        if not batch_tensors:
            return [None] * len(crops)

        # Stack & Move to device
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Handle precision
        if self._compiled or next(self.model.parameters()).dtype == torch.float16:
            batch = batch.half()

        # Inference — chunk to stay within TensorRT max batch size (16)
        _TRT_MAX = 16
        with torch.no_grad():
            if batch.shape[0] <= _TRT_MAX:
                features = self.model(batch)
            else:
                chunks = [batch[i:i + _TRT_MAX] for i in range(0, batch.shape[0], _TRT_MAX)]
                features = torch.cat([self.model(c) for c in chunks], dim=0)

        # L2 Normalize
        features = F.normalize(features, p=2, dim=1)
        features = features.float().cpu().numpy()

        # Map back
        results = [None] * len(crops)
        for idx, feat in zip(valid_indices, features):
            results[idx] = feat
            
        return results
