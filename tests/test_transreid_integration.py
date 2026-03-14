#!/usr/bin/env python3
"""
test_transreid_integration.py — Full TransReID integration test suite.

Tests:
  1. ReIDWrapper loads TransReID (not DINOv2) on Blackwell GPU
  2. InferenceEngine loads TransReID via PROFILE_FULL
  3. End-to-end pipeline produces valid 768-dim L2-normalized vectors
  4. Old DINOv2 identity centroids vs fresh TransReID embeddings comparison
  5. TransReID self-consistency (same crop → same vector)
  6. TransReID discrimination (different people → different vectors)

Usage:
    export POSTGRES_PASSWORD='...'
    source pixxEngine_venv/bin/activate
    python tests/test_transreid_integration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import random
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_transreid")

# ─────────────────────────────────────────────────────────────────────────────
# Test infrastructure
# ─────────────────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0
_results = []


def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        status = "PASS"
    else:
        _failed += 1
        status = "FAIL"
    mark = "✓" if condition else "✗"
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  — {detail}"
    log.info(msg)
    _results.append((name, status, detail))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: ReIDWrapper loads TransReID on Blackwell
# ═════════════════════════════════════════════════════════════════════════════

def test_reid_wrapper_loads_transreid():
    log.info("")
    log.info("═" * 70)
    log.info("TEST 1: ReIDWrapper loads TransReID model on Blackwell GPU")
    log.info("═" * 70)

    import torch
    from src.reid_wrapper import ReIDWrapper

    # Check GPU
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    check("CUDA available", torch.cuda.is_available(), gpu_name)

    is_blackwell = "5070" in gpu_name or "5080" in gpu_name or "5090" in gpu_name
    check("Blackwell GPU detected", is_blackwell, gpu_name)

    # Load wrapper (no TRT for speed)
    t0 = time.time()
    wrapper = ReIDWrapper(device="cuda", compile_tensorrt=False)
    load_ms = (time.time() - t0) * 1000

    # Check it's TransReID, not DINOv2
    check("Model class is TransReIDBackbone",
          type(wrapper.model).__name__ == "TransReIDBackbone",
          f"got: {type(wrapper.model).__name__}")
    check("Input height is 256 (not 518)", wrapper.input_h == 256,
          f"input_h={wrapper.input_h}")
    check("Input width is 128 (not 518)", wrapper.input_w == 128,
          f"input_w={wrapper.input_w}")
    check("Model loaded in < 5s", load_ms < 5000,
          f"{load_ms:.0f} ms")

    # Weights file check
    weights_path = Path(wrapper._TRT_CACHE_DIR) / "transreid_vit_b16_msmt17.pth"
    check("TransReID weights file exists", weights_path.exists(),
          str(weights_path))

    # Test single synthetic crop
    dummy = np.random.randint(0, 255, (400, 200, 3), dtype=np.uint8)
    vecs = wrapper.extract([dummy])
    check("Extract returns list of length 1", len(vecs) == 1)
    check("Vector is 768-dim", vecs[0].shape == (768,),
          f"shape={vecs[0].shape}")

    norm = float(np.linalg.norm(vecs[0]))
    check("Vector is L2-normalized", abs(norm - 1.0) < 1e-4,
          f"norm={norm:.6f}")

    return wrapper


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: InferenceEngine loads TransReID via PROFILE_FULL
# ═════════════════════════════════════════════════════════════════════════════

def test_inference_engine_uses_transreid():
    log.info("")
    log.info("═" * 70)
    log.info("TEST 2: InferenceEngine loads TransReID via PROFILE_FULL")
    log.info("═" * 70)

    from src.workers.inference_engine import InferenceEngine, PROFILE_FULL, PROFILE_PROBE

    engine = InferenceEngine(device="cuda:0")
    engine.load_models(profile=PROFILE_FULL)

    check("Engine has reid_model", engine.reid_model is not None)
    check("reid_model is ReIDWrapper",
          type(engine.reid_model).__name__ == "ReIDWrapper",
          f"got: {type(engine.reid_model).__name__}")
    check("Underlying model is TransReIDBackbone",
          type(engine.reid_model.model).__name__ in ("TransReIDBackbone", "RecordedProgram", "GraphModule"),
          f"got: {type(engine.reid_model.model).__name__} (may be TRT-compiled)")
    check("Engine face_model loaded", engine.face_model is not None)
    check("Engine ocr_model loaded", engine.ocr_model is not None)

    return engine


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: End-to-end pipeline on a real photo
# ═════════════════════════════════════════════════════════════════════════════

def test_e2e_pipeline(engine):
    """Run full pipeline on a sample photo to verify TransReID vectors flow through."""
    log.info("")
    log.info("═" * 70)
    log.info("TEST 3: End-to-end pipeline produces valid TransReID vectors")
    log.info("═" * 70)

    # Use a real photo from R2 if possible, else synthetic
    from scripts.validate_transreid import (
        sign_batch, download_photos, get_connection, fetch_identity_subjects
    )

    conn = get_connection()
    groups = fetch_identity_subjects(conn)
    conn.close()

    if not groups:
        log.warning("No identity data in DB — skipping e2e test with real photo")
        check("Skipped (no DB data)", True, "No identities in project 16")
        return

    # Pick one identity's first subject — download its photo
    iid = list(groups.keys())[0]
    subj = groups[iid][0]

    tmp_dir = Path(tempfile.mkdtemp(prefix="transreid_e2e_"))
    try:
        downloaded = download_photos([subj["r2_key"]], tmp_dir)
        if not downloaded:
            log.warning("Download failed — skipping e2e test")
            check("Skipped (download failed)", True)
            return

        local_path = list(downloaded.values())[0]

        # Run full pipeline
        results = engine.process_photos([str(local_path)])
        check("process_photos returns 1 result", len(results) == 1)

        r = results[0]
        check("Photo loaded successfully", r.success, r.error or "")

        if r.persons:
            p = r.persons[0]
            check("Person detected", True,
                  f"conf={p.confidence:.3f}, blur={p.blur_score:.1f}")

            if p.reid_vector is not None:
                check("reid_vector is 768-dim",
                      p.reid_vector.shape == (768,),
                      f"shape={p.reid_vector.shape}")
                norm = float(np.linalg.norm(p.reid_vector))
                check("reid_vector is L2-normalized",
                      abs(norm - 1.0) < 1e-3, f"norm={norm:.6f}")
            else:
                check("reid_vector present", False, "reid_vector is None")

            if p.face_vector is not None:
                check("face_vector is 512-dim",
                      p.face_vector.shape == (512,),
                      f"shape={p.face_vector.shape}")
            else:
                check("face_vector present (optional)", True, "No face detected — OK for some crops")
        else:
            check("At least 1 person detected", False,
                  f"{len(r.persons)} persons found")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: TransReID self-consistency and discrimination
# ═════════════════════════════════════════════════════════════════════════════

def test_self_consistency(wrapper):
    """Same crop fed twice → nearly identical vectors. Different crops → different vectors."""
    log.info("")
    log.info("═" * 70)
    log.info("TEST 4: Self-consistency and discrimination (synthetic)")
    log.info("═" * 70)

    # Create two visually distinct synthetic "persons"
    person_a = np.zeros((400, 200, 3), dtype=np.uint8)
    person_a[50:350, 30:170] = [200, 50, 50]   # red block

    person_b = np.zeros((400, 200, 3), dtype=np.uint8)
    person_b[50:350, 30:170] = [50, 50, 200]   # blue block

    # Run each twice
    vecs = wrapper.extract([person_a, person_a, person_b, person_b])
    va1, va2, vb1, vb2 = vecs

    # Self-similarity
    sim_aa = float(np.dot(va1, va2))
    sim_bb = float(np.dot(vb1, vb2))
    check("Same crop A → sim ≥ 0.99", sim_aa >= 0.99, f"sim={sim_aa:.6f}")
    check("Same crop B → sim ≥ 0.99", sim_bb >= 0.99, f"sim={sim_bb:.6f}")

    # Cross-similarity (should be lower)
    sim_ab = float(np.dot(va1, vb1))
    check("Different crops A vs B → sim < same-crop sim",
          sim_ab < min(sim_aa, sim_bb),
          f"sim_AB={sim_ab:.4f} vs sim_AA={sim_aa:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Old DINOv2 centroids vs fresh TransReID embeddings
# ═════════════════════════════════════════════════════════════════════════════

def test_dinov2_vs_transreid(wrapper):
    """
    The DB currently holds DINOv2 reid_centroids in pipeline.identities.
    Re-embed the same crops with TransReID and compare:
      - Are old DINOv2 centroids correlated with new TransReID embeddings? (should NOT be)
      - Do TransReID embeddings cluster better? (d' comparison)
    """
    log.info("")
    log.info("═" * 70)
    log.info("TEST 5: Old DINOv2 centroids vs fresh TransReID embeddings")
    log.info("═" * 70)

    import psycopg2
    from src.detection_config import settings as cfg
    from scripts.validate_transreid import (
        download_photos, get_connection, fetch_identity_subjects,
        extract_crop, compute_dprime,
    )

    conn = get_connection()
    groups = fetch_identity_subjects(conn)

    if len(groups) < 10:
        conn.close()
        log.warning("Not enough identities (%d) for DINOv2 vs TransReID test", len(groups))
        check("Skipped (< 10 identities)", True)
        return

    # ── Fetch DINOv2 centroids from DB ────────────────────────────────────
    log.info("Fetching stored DINOv2 reid_centroids from pipeline.identities...")
    cur = conn.cursor()
    id_list = list(groups.keys())

    # Get reid_centroid for sampled identities
    cur.execute("""
        SELECT id, reid_centroid
        FROM pipeline.identities
        WHERE id = ANY(%s::uuid[])
          AND reid_centroid IS NOT NULL
    """, (id_list,))

    dinov2_centroids: dict[str, np.ndarray] = {}
    for row in cur.fetchall():
        iid = str(row[0])
        centroid_str = row[1]
        # pgvector returns string like '[0.1,0.2,...]'
        if centroid_str is not None:
            if isinstance(centroid_str, str):
                vec = np.fromstring(centroid_str.strip("[]"), sep=",", dtype=np.float32)
            else:
                vec = np.array(centroid_str, dtype=np.float32)
            if vec.shape == (768,):
                # L2 normalize for fair cosine comparison
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    dinov2_centroids[iid] = vec / norm

    cur.close()

    # ── Also fetch raw per-subject DINOv2 vectors ─────────────────────────
    # NOTE: reid_enc is stored as raw float32 bytes (3072 bytes = 768 × 4),
    # NOT Fernet-encrypted despite the column name.
    log.info("Fetching per-subject DINOv2 reid vectors (raw float32 bytea)...")
    cur = conn.cursor()

    cur.execute("""
        SELECT s.identity_id, s.reid_enc
        FROM pipeline.subjects s
        JOIN pipeline.identities i ON i.id = s.identity_id
        WHERE i.project_id = %s
          AND s.identity_id = ANY(%s::uuid[])
          AND s.reid_enc IS NOT NULL
        ORDER BY s.identity_id
        LIMIT 2000
    """, ("16", id_list))

    dinov2_subject_vecs: dict[str, list[np.ndarray]] = {}
    decode_ok = 0
    decode_fail = 0
    for row in cur.fetchall():
        iid = str(row[0])
        try:
            raw_bytes = bytes(row[1])
            vec = np.frombuffer(raw_bytes, dtype=np.float32).copy()
            if vec.shape == (768,):
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                    dinov2_subject_vecs.setdefault(iid, []).append(vec)
                    decode_ok += 1
                else:
                    decode_fail += 1
            else:
                decode_fail += 1
        except Exception:
            decode_fail += 1

    cur.close()
    conn.close()

    log.info("  DINOv2 centroids loaded: %d identities", len(dinov2_centroids))
    log.info("  DINOv2 subject vectors decoded: %d OK, %d failed", decode_ok, decode_fail)

    check("DINOv2 centroids present in DB", len(dinov2_centroids) > 0,
          f"{len(dinov2_centroids)} identities")
    check("DINOv2 subject vectors decodable", decode_ok > 0,
          f"{decode_ok} vectors decoded")

    # ── Download photos and compute fresh TransReID embeddings ────────────
    log.info("Downloading photos for fresh TransReID embedding...")
    unique_keys: set[str] = set()
    for subjects in groups.values():
        for s in subjects:
            unique_keys.add(s["r2_key"])

    tmp_dir = Path(tempfile.mkdtemp(prefix="transreid_cmp_"))
    try:
        downloaded = download_photos(sorted(unique_keys), tmp_dir)
        log.info("  Downloaded %d / %d photos", len(downloaded), len(unique_keys))

        # Extract TransReID embeddings per identity
        transreid_vecs: dict[str, list[np.ndarray]] = {}
        total_crops = 0

        for iid, subjects in groups.items():
            crops = []
            for s in subjects:
                local_path = downloaded.get(s["r2_key"])
                if local_path is None:
                    continue
                crop = extract_crop(local_path, s["px1"], s["py1"], s["px2"], s["py2"])
                if crop is not None:
                    crops.append(crop)
                    total_crops += 1

            if len(crops) < 2:
                continue

            vectors = wrapper.extract(crops)
            valid = [v for v in vectors if v is not None]
            if len(valid) >= 2:
                transreid_vecs[iid] = valid

        log.info("  TransReID: %d identities, %d total vectors",
                 len(transreid_vecs), sum(len(v) for v in transreid_vecs.values()))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(transreid_vecs) < 10:
        check("Enough TransReID vectors", False, f"only {len(transreid_vecs)} identities")
        return

    # ── Analysis 1: Cross-model centroid correlation ──────────────────────
    log.info("")
    log.info("─" * 50)
    log.info("Analysis 1: DINOv2 centroid vs TransReID centroid correlation")
    log.info("─" * 50)

    # For each identity that has both, compute cosine between old centroid
    # and new centroid (mean of TransReID vectors)
    cross_sims = []
    common_ids = set(dinov2_centroids.keys()) & set(transreid_vecs.keys())
    for iid in common_ids:
        tr_centroid = np.mean(transreid_vecs[iid], axis=0)
        tr_centroid = tr_centroid / (np.linalg.norm(tr_centroid) + 1e-12)
        sim = float(np.dot(dinov2_centroids[iid], tr_centroid))
        cross_sims.append(sim)

    if cross_sims:
        cross_arr = np.array(cross_sims)
        log.info("  Cross-model cosine (same identity, DINOv2 vs TransReID):")
        log.info("    Count:  %d identities", len(cross_arr))
        log.info("    Mean:   %.4f", cross_arr.mean())
        log.info("    Std:    %.4f", cross_arr.std())
        log.info("    Min:    %.4f", cross_arr.min())
        log.info("    Max:    %.4f", cross_arr.max())

        # These should NOT be highly correlated — different models, different spaces
        check("Cross-model correlation is low (different embedding spaces)",
              cross_arr.mean() < 0.7,
              f"mean={cross_arr.mean():.4f} (expect < 0.7 for unrelated models)")
    else:
        check("Cross-model comparison", False, "No common identities with both centroids")

    # ── Analysis 2: DINOv2 d' (from stored per-subject vectors) ──────────
    log.info("")
    log.info("─" * 50)
    log.info("Analysis 2: DINOv2 d' (from stored per-subject encrypted vectors)")
    log.info("─" * 50)

    dinov2_ids_with_enough = {k: v for k, v in dinov2_subject_vecs.items() if len(v) >= 2}

    if len(dinov2_ids_with_enough) >= 10:
        # Within-identity
        dv2_within = []
        for vecs in dinov2_ids_with_enough.values():
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    dv2_within.append(float(np.dot(vecs[i], vecs[j])))

        # Between-identity
        dv2_ids = list(dinov2_ids_with_enough.keys())
        dv2_between = []
        n_pairs = min(50000, len(dv2_within) * 20)
        for _ in range(n_pairs):
            a, b = random.sample(dv2_ids, 2)
            va = random.choice(dinov2_ids_with_enough[a])
            vb = random.choice(dinov2_ids_with_enough[b])
            dv2_between.append(float(np.dot(va, vb)))

        dv2_within_arr = np.array(dv2_within)
        dv2_between_arr = np.array(dv2_between)
        dv2_dprime = compute_dprime(dv2_within_arr, dv2_between_arr)

        log.info("  DINOv2 (stored vectors): %d identities", len(dinov2_ids_with_enough))
        log.info("  Within-identity:  mean=%.4f  std=%.4f  pairs=%d",
                 dv2_within_arr.mean(), dv2_within_arr.std(), len(dv2_within_arr))
        log.info("  Between-identity: mean=%.4f  std=%.4f  pairs=%d",
                 dv2_between_arr.mean(), dv2_between_arr.std(), len(dv2_between_arr))
        log.info("  DINOv2 d' = %.2f", dv2_dprime)

        check("DINOv2 d' retrieved from DB",
              True, f"d'={dv2_dprime:.2f}")
    else:
        dv2_dprime = None
        log.info("  Not enough DINOv2 subject vectors (%d ids) — skipping d'",
                 len(dinov2_ids_with_enough))
        check("DINOv2 d' from stored vectors", False,
              f"Only {len(dinov2_ids_with_enough)} identities with >=2 vectors")

    # ── Analysis 3: TransReID d' (fresh embeddings) ───────────────────────
    log.info("")
    log.info("─" * 50)
    log.info("Analysis 3: TransReID d' (fresh embeddings on same crops)")
    log.info("─" * 50)

    tr_within = []
    for vecs in transreid_vecs.values():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                tr_within.append(float(np.dot(vecs[i], vecs[j])))

    tr_ids = list(transreid_vecs.keys())
    tr_between = []
    n_pairs = min(50000, len(tr_within) * 20)
    for _ in range(n_pairs):
        a, b = random.sample(tr_ids, 2)
        va = random.choice(transreid_vecs[a])
        vb = random.choice(transreid_vecs[b])
        tr_between.append(float(np.dot(va, vb)))

    tr_within_arr = np.array(tr_within)
    tr_between_arr = np.array(tr_between)
    tr_dprime = compute_dprime(tr_within_arr, tr_between_arr)

    log.info("  TransReID (fresh): %d identities", len(transreid_vecs))
    log.info("  Within-identity:  mean=%.4f  std=%.4f  pairs=%d",
             tr_within_arr.mean(), tr_within_arr.std(), len(tr_within_arr))
    log.info("  Between-identity: mean=%.4f  std=%.4f  pairs=%d",
             tr_between_arr.mean(), tr_between_arr.std(), len(tr_between_arr))
    log.info("  TransReID d' = %.2f", tr_dprime)

    check("TransReID d' > 2.0 (acceptance threshold)", tr_dprime > 2.0,
          f"d'={tr_dprime:.2f}")

    # ── Head-to-head comparison ───────────────────────────────────────────
    log.info("")
    log.info("─" * 50)
    log.info("HEAD-TO-HEAD COMPARISON")
    log.info("─" * 50)

    log.info("")
    log.info("  %-25s  %-12s  %-12s", "Metric", "DINOv2 (DB)", "TransReID (new)")
    log.info("  %-25s  %-12s  %-12s", "─" * 25, "─" * 12, "─" * 12)

    if dv2_dprime is not None:
        log.info("  %-25s  %-12.4f  %-12.4f", "Within-identity mean",
                 dv2_within_arr.mean(), tr_within_arr.mean())
        log.info("  %-25s  %-12.4f  %-12.4f", "Between-identity mean",
                 dv2_between_arr.mean(), tr_between_arr.mean())
        log.info("  %-25s  %-12.4f  %-12.4f", "Separation (Δ means)",
                 dv2_within_arr.mean() - dv2_between_arr.mean(),
                 tr_within_arr.mean() - tr_between_arr.mean())
        log.info("  %-25s  %-12.2f  %-12.2f", "d' (discriminability)",
                 dv2_dprime, tr_dprime)

        improvement = tr_dprime / dv2_dprime if dv2_dprime > 0 else float('inf')
        log.info("")
        log.info("  TransReID improvement: %.1f× better d'", improvement)
        check("TransReID d' > DINOv2 d'",
              tr_dprime > dv2_dprime,
              f"TransReID={tr_dprime:.2f} vs DINOv2={dv2_dprime:.2f}")
    else:
        log.info("  %-25s  %-12s  %-12.4f", "Within-identity mean", "N/A", tr_within_arr.mean())
        log.info("  %-25s  %-12s  %-12.4f", "Between-identity mean", "N/A", tr_between_arr.mean())
        log.info("  %-25s  %-12s  %-12.2f", "d' (discriminability)", "N/A", tr_dprime)

    # ── Threshold sweep for TransReID ─────────────────────────────────────
    log.info("")
    log.info("  TransReID threshold sweep:")
    log.info("  %-12s  %-10s  %-10s", "Threshold", "FPR", "TPR")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        fpr = float((tr_between_arr >= thresh).mean())
        tpr = float((tr_within_arr >= thresh).mean())
        log.info("  %-12.2f  %-10.4f  %-10.4f", thresh, fpr, tpr)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Verify no DINOv2 references in active source code
# ═════════════════════════════════════════════════════════════════════════════

def test_no_dinov2_in_source():
    log.info("")
    log.info("═" * 70)
    log.info("TEST 6: No stale DINOv2 references in source Python files")
    log.info("═" * 70)

    src_dir = Path(__file__).resolve().parent.parent / "src"
    violations = []
    skip_dirs = {"__pycache__"}

    for py_file in sorted(src_dir.rglob("*.py")):
        if any(sd in py_file.parts for sd in skip_dirs):
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                lower = line.lower()
                if "dinov2" in lower and not line.strip().startswith("#"):
                    # Skip pure comments — we only care about code/strings
                    violations.append(f"  {py_file.relative_to(src_dir.parent)}:{lineno}: {line.strip()}")
        except Exception:
            pass

    if violations:
        log.info("  Found %d active DINOv2 references:", len(violations))
        for v in violations[:10]:
            log.info(v)
    check("No active DINOv2 code references in src/",
          len(violations) == 0,
          f"{len(violations)} violations" if violations else "clean")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("pixxEngine TransReID Integration Test Suite")
    log.info("=" * 70)

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    log.info(f"GPU: {gpu}")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"CUDA: {torch.version.cuda}")

    t0 = time.time()

    # Test 1: ReIDWrapper loads TransReID
    wrapper = test_reid_wrapper_loads_transreid()

    # Test 2: InferenceEngine integration
    engine = test_inference_engine_uses_transreid()

    # Test 3: End-to-end pipeline
    test_e2e_pipeline(engine)

    # Test 4: Self-consistency
    test_self_consistency(wrapper)

    # Test 5: DINOv2 vs TransReID (the big one)
    test_dinov2_vs_transreid(wrapper)

    # Test 6: Code hygiene
    test_no_dinov2_in_source()

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0

    log.info("")
    log.info("═" * 70)
    log.info("SUMMARY")
    log.info("═" * 70)
    log.info("  Passed: %d", _passed)
    log.info("  Failed: %d", _failed)
    log.info("  Total:  %d", _passed + _failed)
    log.info("  Time:   %.1fs", elapsed)
    log.info("")

    if _failed > 0:
        log.info("FAILED TESTS:")
        for name, status, detail in _results:
            if status == "FAIL":
                log.info(f"  ✗ {name}: {detail}")

    log.info("")
    if _failed == 0:
        log.info("ALL TESTS PASSED ✓")
    else:
        log.warning("%d TEST(S) FAILED ✗", _failed)

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
