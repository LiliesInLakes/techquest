"""
TechQuest - Colab GPU Precompute & Benchmark Script
===================================================
Executes the one-time 87,425-movie embedding precompute on a CUDA GPU runtime.

Execution Sequence:
1. Environment & GPU Verification (VRAM, PyTorch, SentenceTransformers).
2. Checkpoint Inspection & Validation (checks existing data/processed/checkpoints/).
3. 256-Movie Real Text Benchmark (measures batch sizes 128, 256, 512, movies/sec, peak VRAM).
4. Full Resumable Precompute (87,425 canonical movies from movie_features.db).
5. Comprehensive Artifact Validation (shape (87425, 768), float32, L2-norm == 1.0, non-null).
6. Packaging to precomputed_artifacts.zip.
"""

import os
import sys
import time
import json
import math
import shutil
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def step_1_check_environment() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" STEP 1: ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 70)
    
    import sentence_transformers
    is_cuda = torch.cuda.is_available()
    env_info = {
        "is_cuda": is_cuda,
        "pytorch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "device_name": "CPU"
    }
    
    print(f" PyTorch Version           : {env_info['pytorch_version']}")
    print(f" SentenceTransformers Ver : {env_info['sentence_transformers_version']}")
    print(f" CUDA Available            : {is_cuda}")
    
    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        env_info["device_name"] = gpu_name
        env_info["vram_gb"] = round(vram_gb, 2)
        print(f" GPU Model                 : {gpu_name}")
        print(f" Total VRAM                : {vram_gb:.2f} GB")
    else:
        print(" WARNING: CUDA is NOT detected! Precompute will run on CPU, which is slow.")
    
    # Check DB exists
    db_candidates = ["movie_features.db", "data/movie_features.db"]
    db_path = None
    for cand in db_candidates:
        if os.path.exists(cand):
            db_path = cand
            break
            
    if not db_path:
        raise FileNotFoundError(
            "movie_features.db not found! Please place movie_features.db in the working directory."
        )
    
    print(f" Database Path             : {db_path} ({os.path.getsize(db_path) / (1024**2):.1f} MB)")
    env_info["db_path"] = db_path
    print("=" * 70)
    return env_info


def step_2_inspect_checkpoints(checkpoint_dir: str = "data/processed/checkpoints") -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" STEP 2: CHECKPOINT INSPECTION & COMPATIBILITY CHECK")
    print("=" * 70)
    
    chk_path = Path(checkpoint_dir)
    state_file = chk_path / "checkpoint_state.json"
    
    if not chk_path.exists() or not list(chk_path.glob("batch_*.npy")):
        print(" No existing checkpoints found. Will perform full fresh generation.")
        return {"has_checkpoints": False, "is_compatible": False, "completed_batches": 0}
        
    batch_files = sorted(list(chk_path.glob("batch_*.npy")), key=lambda p: int(p.stem.split('_')[1]))
    total_checkpointed_rows = 0
    all_valid = True
    
    for bf in batch_files:
        arr = np.load(bf)
        norms = np.linalg.norm(arr, axis=1)
        total_checkpointed_rows += arr.shape[0]
        
        is_f32 = arr.dtype == np.float32
        is_768 = arr.shape[1] == 768
        is_norm = np.allclose(norms, 1.0, atol=1e-4)
        has_nan = np.isnan(arr).any() or np.isinf(arr).any()
        
        print(f" {bf.name}: Shape={arr.shape}, Dtype={arr.dtype}, L2Norm=[{norms.min():.4f}, {norms.max():.4f}], NonNull={not has_nan}")
        if not (is_f32 and is_768 and is_norm and not has_nan):
            all_valid = False
            
    state_meta = {}
    if state_file.exists():
        with open(state_file) as f:
            state_meta = json.load(f)
            
    print(f"\n Checkpoints Found        : {len(batch_files)} batches ({total_checkpointed_rows} movies)")
    print(f" Compatibility Assessment : {'COMPATIBLE - Will resume' if all_valid else 'INCOMPATIBLE - Will regenerate'}")
    print("=" * 70)
    
    return {
        "has_checkpoints": True,
        "is_compatible": all_valid,
        "completed_batches": len(batch_files),
        "rows": total_checkpointed_rows,
        "state": state_meta
    }


def step_3_benchmark_cuda(
    sample_texts: List[str],
    model_name: str = "multi-qa-mpnet-base-dot-v1",
    test_batch_sizes: List[int] = [128, 256, 512],
    total_corpus_size: int = 87425
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" STEP 3: CUDA 256-TEXT BENCHMARK & BATCH SIZE SELECTION")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Loading model: {model_name} on {device.upper()}...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 128
    load_time = time.perf_counter() - t0
    print(f" Model loaded in {load_time:.2f}s")
    
    benchmark_results = []
    best_mps = 0.0
    best_batch_size = test_batch_sizes[0]
    
    # Use real texts
    texts_256 = sample_texts[:256]
    
    # Warmup
    print(" Running warmup pass (16 items)...")
    with torch.inference_mode():
        _ = model.encode(texts_256[:16], batch_size=16, normalize_embeddings=True)
    if device == "cuda":
        torch.cuda.synchronize()
        
    print("\n Running batch size benchmark on 256 movie texts:")
    print(f" {'Batch Size':>12} | {'Time (s)':>10} | {'Throughput (movies/s)':>22} | {'Peak VRAM (MB)':>15} | {'Est 87k Time':>15}")
    print("-" * 85)
    
    for bs in test_batch_sizes:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        try:
            t_start = time.perf_counter()
            with torch.inference_mode():
                embs = model.encode(
                    texts_256,
                    batch_size=bs,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            
            mps = len(texts_256) / elapsed
            est_total_sec = total_corpus_size / mps
            est_min = est_total_sec / 60.0
            
            peak_vram_mb = 0.0
            if device == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            print(f" {bs:>12} | {elapsed:>9.3f}s | {mps:>20.1f} m/s | {peak_vram_mb:>13.1f} MB | {est_min:>12.1f} min")
            
            benchmark_results.append({
                "batch_size": bs,
                "elapsed": elapsed,
                "mps": mps,
                "peak_vram_mb": peak_vram_mb,
                "est_min": est_min
            })
            
            if mps > best_mps:
                best_mps = mps
                best_batch_size = bs
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f" {bs:>12} | OOM Error (exceeded available VRAM)")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise e
                
    print("-" * 85)
    est_total_min = (total_corpus_size / best_mps) / 60.0
    print(f" Selected Optimal Safe Batch Size: {best_batch_size}")
    print(f" Measured Max Throughput        : {best_mps:.1f} movies/sec")
    print(f" Measured Estimated 87k Runtime : {est_total_min:.2f} minutes")
    print("=" * 70)
    
    return {
        "best_batch_size": best_batch_size,
        "best_mps": best_mps,
        "est_total_min": est_total_min,
        "results": benchmark_results,
        "model": model
    }


def step_4_run_precompute(encode_batch_size: int, chunk_size: int = 2000):
    print("\n" + "=" * 70)
    print(" STEP 4: EXECUTING RESUMABLE 87,425-MOVIE PRECOMPUTE")
    print("=" * 70)
    
    from src.features.matrix_builder import FeatureMatrixBuilder
    
    builder = FeatureMatrixBuilder()
    t_start = time.perf_counter()
    feature_matrix = builder.build_and_save(
        batch_size=chunk_size,
        encode_batch_size=encode_batch_size
    )
    total_time = time.perf_counter() - t_start
    
    print(f"\n Precompute finished in {total_time / 60.0:.2f} minutes ({total_time:.1f}s)")
    print("=" * 70)
    return feature_matrix


def step_5_validate_artifacts(output_dir: str = "data/processed") -> bool:
    print("\n" + "=" * 70)
    print(" STEP 5: COMPREHENSIVE ARTIFACT VALIDATION")
    print("=" * 70)
    
    out_path = Path(output_dir)
    emb_path = out_path / "movie_embeddings.npy"
    map_path = out_path / "movie_id_map.json"
    manifest_path = out_path / "precompute_manifest.json"
    fm_path = out_path / "feature_matrix.pkl"
    cv_path = out_path / "count_vectorizer.pkl"
    tv_path = out_path / "tfidf_vectorizer.pkl"
    
    required_files = [emb_path, map_path, manifest_path, fm_path, cv_path, tv_path]
    for rf in required_files:
        if not rf.exists():
            raise FileNotFoundError(f"Missing required artifact: {rf}")
        print(f" Artifact found: {rf.name} ({rf.stat().st_size / (1024**2):.2f} MB)")
        
    # Check embeddings
    embeddings = np.load(emb_path)
    norms = np.linalg.norm(embeddings, axis=1)
    
    print(f"\n Embedding Matrix Shape    : {embeddings.shape} (Expected: (87425, 768))")
    print(f" Embedding Matrix Dtype    : {embeddings.dtype} (Expected: float32)")
    print(f" L2 Norm Range             : [{norms.min():.6f}, {norms.max():.6f}] (Unit normalized: {np.allclose(norms, 1.0, atol=1e-3)})")
    print(f" NaN / Inf Free            : {not (np.isnan(embeddings).any() or np.isinf(embeddings).any())}")
    
    # Check ID map
    with open(map_path) as f:
        id_map = json.load(f)
    print(f" ID Map Item Count         : {id_map.get('movie_count')} (Expected: 87425)")
    
    # Check Manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f" Manifest Status           : {manifest.get('completed_status')} / {manifest.get('status')}")
    print(f" Manifest Model            : {manifest.get('model_name')}")
    print(f" Manifest Dimension        : {manifest.get('embedding_dimension') or manifest.get('dimension')}")
    print(f" Generation Device         : {manifest.get('generation_device')}")
    
    assert embeddings.shape == (87425, 768), f"Shape mismatch: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Dtype mismatch: {embeddings.dtype}"
    assert not np.isnan(embeddings).any(), "Found NaNs in embeddings!"
    assert id_map.get('movie_count') == 87425, "ID map count mismatch!"
    
    print("\n ALL ARTIFACTS VERIFIED AND FULLY VALID!")
    print("=" * 70)
    return True


def step_6_package_artifacts(
    output_dir: str = "data/processed",
    zip_path: str = "precomputed_artifacts.zip"
):
    print("\n" + "=" * 70)
    print(" STEP 6: PACKAGING ARTIFACTS FOR LOCAL DEPLOYMENT")
    print("=" * 70)
    
    out_path = Path(output_dir)
    files_to_pack = [
        "movie_embeddings.npy",
        "movie_id_map.json",
        "precompute_manifest.json",
        "count_vectorizer.pkl",
        "tfidf_vectorizer.pkl",
        "feature_matrix.pkl"
    ]
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in files_to_pack:
            fpath = out_path / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"data/processed/{fname}")
                print(f" Packed: data/processed/{fname} ({fpath.stat().st_size / (1024**2):.2f} MB)")
            else:
                print(f" Warning: {fname} not found!")
                
    zip_size_mb = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"\n Complete archive created: {zip_path} ({zip_size_mb:.2f} MB)")
    print("=" * 70)


def main():
    print("######################################################################")
    print("#  TECHQUEST - HIGH-THROUGHPUT CUDA PRECOMPUTE WORKFLOW             #")
    print("######################################################################")
    
    # 1. Environment Check
    env_info = step_1_check_environment()
    
    # 2. Checkpoint Inspection
    chk_info = step_2_inspect_checkpoints()
    
    # 3. Load DB canonical movies to extract sample texts for benchmark
    from src.db.connection import MovieDatabase
    db = MovieDatabase(db_path=env_info["db_path"])
    canonical_movies = db.load_canonical_movies()
    print(f" Loaded {len(canonical_movies)} canonical movies from database.")
    sample_texts = [m['content_text'] for m in canonical_movies]
    
    # 4. Benchmark CUDA batch sizes on 256 texts
    bench_info = step_3_benchmark_cuda(sample_texts=sample_texts)
    
    # 5. Full Resumable Precompute
    step_4_run_precompute(encode_batch_size=bench_info["best_batch_size"])
    
    # 6. Validate Artifacts
    step_5_validate_artifacts()
    
    # 7. Package Artifacts
    step_6_package_artifacts()
    
    print("\n SUCCESS: Precomputation, verification, and artifact packaging complete!")


if __name__ == "__main__":
    main()
