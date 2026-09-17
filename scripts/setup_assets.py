#!/usr/bin/env python3
"""
TechQuest Asset Setup & Verification Tool

Downloads prebuilt runtime artifacts (SQLite movie database, 768-d MPNet embeddings,
and sparse TF-IDF matrices) from a GitHub Release asset or extracts from a local archive.
Verifies SHA256 checksums to ensure file integrity.
"""

import os
import sys
import json
import hashlib
import zipfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_REPO = "LiliesInLakes/techquest"
DEFAULT_TAG = "v1.0.0"
BUNDLE_FILENAME_TEMPLATE = "techquest-artifacts-{tag}.zip"

# Authoritative Manifest for Runtime Artifacts
ARTIFACT_MANIFEST = {
    "movie_features.db": {
        "dest": "movie_features.db",
        "description": "Authoritative SQLite database (87,425 canonical movies)",
        "size_bytes": 256065536,
        "sha256": "73155f5b45758bb6325fb89f4889e4f67bb29ae969dda42ca9a69618b36ef4c2"
    },
    "data/processed/movie_embeddings.npy": {
        "dest": "data/processed/movie_embeddings.npy",
        "description": "87,425 x 768 float32 L2-normalized sentence embeddings",
        "size_bytes": 268569728,
        "sha256": "db4bb154923a4e20a671080e7771d3b0fe08f1b98d2151ba96d1d1faf5400e83"
    },
    "data/processed/movie_id_map.json": {
        "dest": "data/processed/movie_id_map.json",
        "description": "TMDB ID to matrix index mapping for 87,425 movies",
        "size_bytes": 1433734,
        "sha256": "4c0cdfd23f9b16a418dd7186308a516d5f9c94690c7098bf82f4eb24cd4f7ad6"
    },
    "data/processed/count_vectorizer.pkl": {
        "dest": "data/processed/count_vectorizer.pkl",
        "description": "Fitted 5,000-feature CountVectorizer",
        "size_bytes": 149409,
        "sha256": "4b0ae7725b34f8c09156c226bce6fc05333399b6a242a8b91d5eb04ac645ffc6"
    },
    "data/processed/tfidf_vectorizer.pkl": {
        "dest": "data/processed/tfidf_vectorizer.pkl",
        "description": "Fitted 5,000-feature TfidfVectorizer",
        "size_bytes": 189647,
        "sha256": "58f58e185895afdda64573e7c5b1ce2f066edbd7a5bb975dac4fa17b923eef5f"
    },
    "data/processed/count_matrix.npz": {
        "dest": "data/processed/count_matrix.npz",
        "description": "Sparse Count feature matrix (87,425 x 5,000)",
        "size_bytes": 5821303,
        "sha256": "b6029a777010c1dc59b49509d8b5871a809cac22fcffa5b6411b9f07f67a6672"
    },
    "data/processed/tfidf_matrix.npz": {
        "dest": "data/processed/tfidf_matrix.npz",
        "description": "Sparse TF-IDF feature matrix (87,425 x 5,000)",
        "size_bytes": 29071877,
        "sha256": "43b4d701fcc35758cf1110064c0882bec4f5e15f5c6beb798943810025d71aa2"
    },
    "data/processed/keyword_feature_embeddings.npy": {
        "dest": "data/processed/keyword_feature_embeddings.npy",
        "description": "Vocabulary keyword embeddings matrix",
        "size_bytes": 15360128,
        "sha256": "a344235f9ad76d96a4f146dd83190697d5ebf8c763350c2652bd27da03fa87cb"
    },
    "data/processed/keyword_embeddings.pkl": {
        "dest": "data/processed/keyword_embeddings.pkl",
        "description": "Keyword to embedding dictionary",
        "size_bytes": 15570922,
        "sha256": "560070c3df01a54206264830ecbd11594092e68bcacaf4785819efbaa7b5dc80"
    },
    "data/processed/precompute_manifest.json": {
        "dest": "data/processed/precompute_manifest.json",
        "description": "Precomputation build metadata and settings",
        "size_bytes": 734,
        "sha256": "4f60250c1ef6561e7aaaf82fac156592157006a3b6c924dd1144121329c338a4"
    },
    "data/movie_db_manifest.json": {
        "dest": "data/movie_db_manifest.json",
        "description": "Database verification manifest",
        "size_bytes": 242,
        "sha256": "5a3837a70ae75226a9e4429ac7665270e3d0239852c56d019f5b98c038e2357c"
    }
}


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA256 checksum of a file in 8MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_installed_artifacts(root_dir: Path, verbose: bool = True) -> bool:
    """Checks if all required runtime artifacts are present and have valid checksums."""
    all_valid = True
    if verbose:
        print("=== Checking Runtime Artifact Status ===")

    for rel_path, meta in ARTIFACT_MANIFEST.items():
        p = root_dir / rel_path
        if not p.exists():
            if verbose:
                print(f"  [MISSING] {rel_path} ({meta['description']})")
            all_valid = False
            continue

        actual_size = p.stat().st_size
        expected_size = meta["size_bytes"]
        if actual_size != expected_size:
            if verbose:
                print(f"  [SIZE MISMATCH] {rel_path}: expected {expected_size} bytes, got {actual_size}")
            all_valid = False
            continue

        actual_sha = compute_sha256(p)
        expected_sha = meta["sha256"]
        if actual_sha != expected_sha:
            if verbose:
                print(f"  [CORRUPT / HASH MISMATCH] {rel_path}")
            all_valid = False
            continue

        if verbose:
            print(f"  [OK] {rel_path} ({actual_size / (1024 * 1024):.2f} MB)")

    return all_valid


def download_file_with_progress(url: str, dest_path: Path):
    """Downloads a file via HTTP with a progress counter."""
    print(f"Connecting to: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TechQuest-Asset-Downloader/1.0"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            total_length = response.headers.get("content-length")
            total_bytes = int(total_length) if total_length else None
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            print(f"Downloading {dest_path.name} (Total: {total_bytes / (1024*1024):.1f} MB if known)...")
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        sys.stdout.write(f"\r  Progress: {mb_done:.1f} / {mb_total:.1f} MB ({pct:.1f}%)")
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
                    sys.stdout.flush()
            print("\nDownload complete.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Release asset not found (HTTP 404) at {url}.\n"
                f"Please verify that the release tag exists on GitHub or supply --local-zip <path>."
            ) from e
        raise RuntimeError(f"HTTP download error: {e.code} - {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network connection failed: {e.reason}") from e


def extract_archive(zip_path: Path, root_dir: Path):
    """Extracts the archive and verifies each file."""
    print(f"\nExtracting archive {zip_path.name} to {root_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()
        for member in file_list:
            # Skip directories or OS junk
            if member.endswith("/") or member.startswith("__MACOSX"):
                continue
            dest_target = root_dir / member
            dest_target.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting: {member}")
            zf.extract(member, path=root_dir)

    print("Extraction completed. Verifying installed files...")
    if verify_installed_artifacts(root_dir, verbose=True):
        print("\nAll runtime artifacts installed and verified successfully!")
    else:
        raise RuntimeError("One or more extracted artifacts failed integrity check.")


def main():
    parser = argparse.ArgumentParser(description="Download and verify TechQuest precomputed runtime assets.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo owner/name (default: {DEFAULT_REPO})")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG})")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files, do not download")
    parser.add_argument("--force", action="store_true", help="Force download and overwrite existing files")
    parser.add_argument("--local-zip", type=str, default=None, help="Path to a local zip archive to extract instead of downloading")

    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent.parent

    # 1. Verification-only mode
    if args.verify_only:
        valid = verify_installed_artifacts(root_dir, verbose=True)
        if valid:
            print("\n[SUCCESS] All runtime artifacts are present and valid.")
            sys.exit(0)
        else:
            print("\n[INCOMPLETE] Missing or corrupt artifacts detected. Run without --verify-only to download.")
            sys.exit(1)

    # 2. Check if already complete and valid
    if not args.force and verify_installed_artifacts(root_dir, verbose=False):
        print("All runtime artifacts are already installed and verified! No download needed.")
        print("Use --force if you wish to re-download and overwrite.")
        sys.exit(0)

    # 3. Handle local zip
    if args.local_zip:
        local_zip_path = Path(args.local_zip).resolve()
        if not local_zip_path.exists():
            print(f"Error: Local zip archive not found at: {local_zip_path}", file=sys.stderr)
            sys.exit(1)
        extract_archive(local_zip_path, root_dir)
        sys.exit(0)

    # 4. Download from GitHub Release
    bundle_name = BUNDLE_FILENAME_TEMPLATE.format(tag=args.tag)
    release_url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{bundle_name}"
    
    tmp_download_path = root_dir / "data" / f"temp_{bundle_name}"
    tmp_download_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file_with_progress(release_url, tmp_download_path)
        extract_archive(tmp_download_path, root_dir)
    finally:
        if tmp_download_path.exists():
            try:
                tmp_download_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
