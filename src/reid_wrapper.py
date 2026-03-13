import torch
import torch.nn.functional as F
import timm
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from torchvision import transforms
import cv2
import os

logger = logging.getLogger(__name__)


class DynamicResizeWithPad:
    """
    Dynamic Resizing for ReID: Preserves aspect ratio and pads small crops.
    
    Strategy:
    - If crop is smaller than target_size, pad with gray (128) to preserve geometry
    - If crop is larger, resize with aspect ratio preservation then pad
    - This prevents upscaling artifacts from distant runners (e.g., 50px -> 224px)
    """
    
    def __init__(self, target_size: int = 518, pad_color: int = 128):
        self.target_size = target_size
        self.pad_color = pad_color  # Gray padding (neutral for ImageNet normalization)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: Input image as numpy array (H, W, C) in RGB format
        Returns:
            Padded/resized image as numpy array (target_size, target_size, C)
        """
        h, w = img.shape[:2]
        target = self.target_size
        
        # Case 1: Crop is smaller than target in both dimensions - pad only
        if h <= target and w <= target:
            # No resizing needed, just center-pad
            result = np.full((target, target, 3), self.pad_color, dtype=np.uint8)
            y_offset = (target - h) // 2
            x_offset = (target - w) // 2
            result[y_offset:y_offset + h, x_offset:x_offset + w] = img
            return result
        
        # Case 2: Crop is larger - resize with aspect ratio preservation, then pad
        scale = min(target / h, target / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use INTER_AREA for downscaling (better quality)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center-pad to target size
        result = np.full((target, target, 3), self.pad_color, dtype=np.uint8)
        y_offset = (target - new_h) // 2
        x_offset = (target - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result


class ReIDWrapper:
    """
    Production-Ready ReID using DINOv2 (ViT-Base).
    
    Source: Meta AI (via timm)
    Architecture: ViT-Base (Self-Supervised)
    Capabilities: Holistic body structure analysis (Cross-Clothes ready)
    
    Blackwell Optimization: Uses torch-TensorRT for FP16 compilation on RTX 5070.
    """
    
    def __init__(self, device: str = 'cuda', compile_tensorrt: bool = True):
        self.device = device
        self._compiled = False
        
        # DINOv2-Base (Standard 14x14 patch size)
        model_name = 'vit_base_patch14_reg4_dinov2.lvd142m'  # 518x518 input, 768-dim features
        
        logger.info(f"Loading ReID Model: {model_name}...")
        
        # Use DINOv2's native 518x518 resolution for maximum discrimination.
        # The model was pretrained with 37×37 = 1369 patch tokens at this size.
        # At 224 we only got 16×16 = 256 tokens — an 81% reduction that destroyed
        # the fine-grained spatial encoding needed for ReID (d' was only 1.56).
        # NOTE: global_pool='' is intentional.
        # Newer timm renames the final layer norm from 'norm' → 'fc_norm' when
        # global_pool='avg', but the pretrained weights on HuggingFace Hub still
        # use the 'norm' key.  Loading with global_pool='avg' raises:
        #   Missing key(s): "fc_norm.weight", "fc_norm.bias"
        #   Unexpected key(s): "norm.weight", "norm.bias"
        # Using global_pool='' loads correctly (uses 'norm' naming), and we
        # replicate the avg-pool-over-patch-tokens behaviour with a thin wrapper.
        # ── Fail fast if local weights are missing ────────────────────────
        # DINOv2 weights must be vendored locally before the pipeline runs.
        # Run:  pixxEngine_venv/bin/python3 scripts/download_model_weights.py
        _dinov2_weights = Path(self._TRT_CACHE_DIR) / "dinov2_vit_b14_reg4.pth"
        if not _dinov2_weights.exists():
            raise FileNotFoundError(
                f"DINOv2 weights not found: {_dinov2_weights}\n"
                f"Run:  pixxEngine_venv/bin/python3 scripts/download_model_weights.py"
            )

        logger.info(f"Loading DINOv2 from vendored weights: {_dinov2_weights.name} …")
        self.model = timm.create_model(
            model_name,
            pretrained=False,   # weights loaded from local vendored file below
            num_classes=0,  # Remove classification head (we want features)
            img_size=518,   # Native DINOv2 resolution — no pos-embed interpolation
            global_pool=''  # '' → returns patch tokens [B, N, C]; we avg-pool below
        )
        self.model.load_state_dict(
            torch.load(str(_dinov2_weights), weights_only=True, map_location="cpu"),
            strict=True,
        )

        # Wrap forward to reproduce global_pool='avg':
        # average over the N patch tokens → [B, 768] holistic descriptor.
        # DINOv2-reg4 prepends 1 CLS + 4 register tokens (num_prefix_tokens=5);
        # global_pool='' already strips those, so all returned tokens are patches.
        _base_forward = self.model.forward

        def _avg_pool_forward(x: torch.Tensor) -> torch.Tensor:
            feats = _base_forward(x)  # [B, N, C]
            return feats.mean(dim=1)  # [B, C]

        self.model.forward = _avg_pool_forward
        
        self.model.to(device)
        self.model.eval()
        
        # Input size matches the native DINOv2 resolution
        self.input_size = 518
        
        # Dynamic resize with padding (preserves aspect ratio for small crops)
        self.dynamic_resize = DynamicResizeWithPad(target_size=self.input_size, pad_color=128)
        
        # Standard ImageNet normalization (DINOv2 expects this)
        # Note: No Resize transform - handled by DynamicResizeWithPad
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
        
        logger.info("ReID Model loaded successfully ✓ (Dynamic Resizing enabled)")

    # Path for the pre-compiled TRT engine (main project weights/ dir — not src/weights/)
    _TRT_CACHE_DIR  = str(Path(__file__).resolve().parent.parent / "weights")
    _TRT_CACHE_FILE = str(Path(__file__).resolve().parent.parent / "weights" / "dinov2_vit_b14_trt_fp16.ep")

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
        Compile DINOv2 with torch-TensorRT for Blackwell FP16 optimization.

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
                        "DINOv2 TRT engine loaded from cache in %.2fs ✓ (%s)",
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
            logger.info("Compiling DINOv2 with torch-TensorRT (FP16) — first time only...")

            self.model.half()

            self.model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    min_shape=(1, 3, 518, 518),
                    opt_shape=(4, 3, 518, 518),
                    max_shape=(16, 3, 518, 518),
                    dtype=torch.half,
                )],
                enabled_precisions={torch.half},
                truncate_long_and_double=True,
                workspace_size=1 << 27,       # 128 MB
                use_explicit_typing=False,    # 2.10.0 default changed; keep FP16 mixed path
            )
            self._compiled = True
            logger.info("DINOv2 TRT compilation complete ✓")

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
                    "DINOv2 TRT engine cached → %s (%.1f MB)",
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
        Performs letterbox resize and normalization entirely on GPU.
        
        Uses aspect-ratio preserving resize with gray padding to match
        the quality of the CPU path (DynamicResizeWithPad).
        
        CRITICAL: All tensors must be contiguous for CUDA kernels (Blackwell fix).
        UINT8 PIPELINE: Accepts uint8 tensors, converts to float JIT.
        """
        batch_tensors = []
        valid_indices = []
        target = self.input_size  # 224
        
        for i, crop in enumerate(crops):
            if crop is None or crop.numel() == 0:
                continue
            try:
                # CRITICAL: Ensure input is contiguous (fixes cudaErrorIllegalAddress)
                if not crop.is_contiguous():
                    crop = crop.contiguous()
                
                # --- JIT NORMALIZATION FOR UINT8 ---
                # Convert uint8 to float 0-1 (only for small crops, minimal cost)
                if crop.dtype == torch.uint8:
                    crop = crop.float().div(255.0)
                # ------------------------------------
                
                # crop is (C, H, W) on GPU, values 0-1
                c, h, w = crop.shape
                
                # Validate dimensions
                if h < 1 or w < 1 or c != 3:
                    logger.warning(f"Invalid crop shape {crop.shape}, skipping")
                    continue
                
                # 1. Calculate scale to preserve aspect ratio (letterbox)
                scale = min(target / h, target / w)
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                
                # 2. Resize with aspect ratio preservation
                resized = F.interpolate(
                    crop.unsqueeze(0),  # (1, C, H, W)
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).contiguous()  # Back to (C, H, W), ensure contiguous
                
                # 3. Create gray canvas (0.5 ≈ 128/255 for 0-1 normalized tensor)
                canvas = torch.full(
                    (c, target, target), 
                    0.5,  # Gray padding
                    device=crop.device, 
                    dtype=crop.dtype
                )
                
                # 4. Paste resized image at center
                y_off = (target - new_h) // 2
                x_off = (target - new_w) // 2
                canvas[:, y_off:y_off + new_h, x_off:x_off + new_w] = resized
                
                # CRITICAL: Ensure canvas is contiguous before batching
                batch_tensors.append(canvas.contiguous())
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Tensor crop letterbox failed: {e}")
        
        if not batch_tensors:
            return [None] * len(crops)
        
        # Stack into batch (all on GPU) - ensure contiguous
        batch = torch.stack(batch_tensors).contiguous()  # (B, C, H, W)
        
        # Normalize on GPU using precomputed constants
        batch = (batch - self.mean_gpu) / self.std_gpu
        batch = batch.contiguous()  # Ensure contiguous after normalization
        
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
                
                # Apply dynamic resize with padding
                resized = self.dynamic_resize(crop_rgb)
                
                # Normalize for DINOv2
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
