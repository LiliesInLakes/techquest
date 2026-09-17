import sys
import os
import math
import json
import hashlib
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
import yaml

from src.db.connection import MovieDatabase


@dataclass
class FeatureMatrix:
    movie_ids: List[int]
    movie_titles: List[str]
    count_matrix: Any
    tfidf_matrix: Any
    semantic_embeddings: np.ndarray
    feature_names: List[str]
    tfidf_feature_names: List[str]
    movie_metadata: Dict[int, Dict]
    tmdb_id_to_idx: Dict[int, int] = field(default_factory=dict)
    idx_to_tmdb_id: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tmdb_id_to_idx and self.movie_ids:
            self.tmdb_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
            self.idx_to_tmdb_id = {i: mid for i, mid in enumerate(self.movie_ids)}


class FeatureMatrixBuilder:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.data_config = self.config.get("data", {})
        self.db_path = self.data_config.get("db_path", "movie_features.db")
        
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.semantic_model_name = self.config.get("recommendation", {}).get(
            "semantic_model_name", "multi-qa-mpnet-base-dot-v1"
        )
        self._semantic_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def print_device_banner(self):
        import sentence_transformers
        print("=" * 65, file=sys.stderr)
        print(f" PRECOMPUTE ENVIRONMENT & DEVICE DIAGNOSTICS", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        print(f" PyTorch Version           : {torch.__version__}", file=sys.stderr)
        print(f" SentenceTransformers Ver : {sentence_transformers.__version__}", file=sys.stderr)
        print(f" Target Model              : {self.semantic_model_name} (768-d)", file=sys.stderr)
        print(f" Execution Device          : {self.device.upper()}", file=sys.stderr)
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            print(f" GPU Model                 : {gpu_name}", file=sys.stderr)
            print(f" GPU VRAM                  : {vram_gb:.2f} GB", file=sys.stderr)
        else:
            import multiprocessing
            print(f" CPU Cores / Threads       : {multiprocessing.cpu_count()} cores, {torch.get_num_threads()} threads", file=sys.stderr)
        print("=" * 65, file=sys.stderr)

    @property
    def semantic_model(self) -> SentenceTransformer:
        if self._semantic_model is None:
            if self.device == "cpu":
                import multiprocessing
                try:
                    torch.set_num_threads(multiprocessing.cpu_count())
                except Exception:
                    pass
            self.print_device_banner()
            self._semantic_model = SentenceTransformer(self.semantic_model_name, device=self.device)
            self._semantic_model.max_seq_length = 128
        return self._semantic_model

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def load_canonical_dataset(self) -> List[Dict[str, Any]]:
        """
        Loads the authoritative canonical movie corpus from movie_features.db.
        """
        print(f"Loading canonical movie corpus from {self.db_path}...", file=sys.stderr)
        db = MovieDatabase(db_path=self.db_path)
        valid, msg = db.verify_integrity()
        if not valid:
            raise RuntimeError(f"Database integrity verification failed: {msg}")
        return db.load_canonical_movies()

    def _compute_dataset_hash(self, movie_ids: List[int]) -> str:
        h = hashlib.sha256()
        for mid in movie_ids:
            h.update(str(mid).encode('utf-8'))
        return h.hexdigest()

    def build_semantic_embeddings_resumable(
        self,
        texts: List[str],
        movie_ids: List[int],
        batch_size: int = 2000,
        encode_batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Builds semantic embeddings with batch-level checkpointing for fault-tolerance.
        batch_size: Number of items per checkpoint chunk file (e.g. 2000).
        encode_batch_size: Model inference mini-batch size (e.g. 128/256/512).
        """
        if encode_batch_size is None:
            encode_batch_size = 256 if self.device == "cuda" else 128

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        total_items = len(texts)
        num_batches = math.ceil(total_items / batch_size)
        
        dataset_hash = hashlib.sha256("".join(str(i) for i in movie_ids).encode()).hexdigest()
        state_file = self.checkpoint_dir / "checkpoint_state.json"
        
        completed_batches = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("dataset_hash") == dataset_hash and state.get("model_name") == self.semantic_model_name:
                    completed_batches = state.get("completed_batches", {})
                    print(f"Resuming precompute from checkpoint: {len(completed_batches)}/{num_batches} batches already complete.", file=sys.stderr)
                else:
                    print("Previous checkpoint metadata mismatch. Starting fresh precompute...", file=sys.stderr)
            except Exception as e:
                print(f"Could not read checkpoint state ({e}), starting fresh...", file=sys.stderr)

        for b in range(num_batches):
            b_str = str(b)
            chk_path = self.checkpoint_dir / f"batch_{b}.npy"
            
            if b_str in completed_batches and chk_path.exists():
                continue
            
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, total_items)
            batch_texts = texts[start_idx:end_idx]
            
            print(f"Encoding batch {b+1}/{num_batches} (items {start_idx} to {end_idx}, encode_batch_size={encode_batch_size})...", file=sys.stderr)
            with torch.inference_mode():
                batch_embeddings = self.semantic_model.encode(
                    batch_texts,
                    batch_size=encode_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True
                ).astype(np.float32)
            
            np.save(chk_path, batch_embeddings)
            completed_batches[b_str] = {
                "batch_index": b,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "count": len(batch_texts)
            }
            
            # Save updated checkpoint state atomically
            temp_state = state_file.with_suffix(".json.tmp")
            with open(temp_state, "w") as f:
                json.dump({
                    "dataset_hash": dataset_hash,
                    "model_name": self.semantic_model_name,
                    "total_items": total_items,
                    "num_batches": num_batches,
                    "completed_batches": completed_batches
                }, f, indent=2)
            temp_state.replace(state_file)

        # Concatenate all batches in order
        print("Assembling complete embedding matrix from batch checkpoints...", file=sys.stderr)
        all_embeddings = []
        for b in range(num_batches):
            chk_path = self.checkpoint_dir / f"batch_{b}.npy"
            if not chk_path.exists():
                raise RuntimeError(f"Missing checkpoint file: {chk_path}")
            all_embeddings.append(np.load(chk_path))
            
        full_matrix = np.vstack(all_embeddings).astype(np.float32)
        print(f"Full embedding matrix assembled with shape {full_matrix.shape}.", file=sys.stderr)
        return full_matrix

    def build_matrices(
        self,
        movies: List[Dict[str, Any]],
        batch_size: int = 2000,
        encode_batch_size: Optional[int] = None
    ) -> FeatureMatrix:
        texts = [m['content_text'] for m in movies]
        movie_ids = [m['tmdb_id'] for m in movies]
        movie_titles = [m['title'] for m in movies]
        
        print("Building CountVectorizer matrix (5000 features)...", file=sys.stderr)
        count_matrix = self.count_vectorizer.fit_transform(texts)
        feature_names = self.count_vectorizer.get_feature_names_out().tolist()
        
        print("Building TF-IDF matrix (5000 features)...", file=sys.stderr)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        print(f"Building semantic embeddings using {self.semantic_model_name} (768-d)...", file=sys.stderr)
        semantic_embeddings = self.build_semantic_embeddings_resumable(
            texts=texts,
            movie_ids=movie_ids,
            batch_size=batch_size,
            encode_batch_size=encode_batch_size
        )
        
        movie_metadata = {}
        tmdb_id_to_idx = {}
        idx_to_tmdb_id = {}
        
        for idx, m in enumerate(movies):
            mid = m['tmdb_id']
            tmdb_id_to_idx[mid] = idx
            idx_to_tmdb_id[idx] = mid
            movie_metadata[mid] = {
                'id': mid,
                'tmdb_id': mid,
                'movielens_id': m.get('movielens_id'),
                'title': m['title'],
                'genres': m.get('genres', []),
                'keywords': m.get('keywords', []),
                'tags': m.get('tags', []),
                'overview': m.get('overview', ''),
                'vote_average': m.get('vote_average', 0.0),
                'vote_count': m.get('vote_count', 0),
                'avg_rating': m.get('avg_rating', 0.0),
                'rating_count': m.get('rating_count', 0),
                'director': m.get('director', []),
                'cast': m.get('cast', []),
                'release_date': m.get('release_date', ''),
                'original_language': m.get('original_language', 'en'),
                'content_text': m['content_text']
            }
        
        return FeatureMatrix(
            movie_ids=movie_ids,
            movie_titles=movie_titles,
            count_matrix=count_matrix,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=semantic_embeddings,
            feature_names=feature_names,
            tfidf_feature_names=tfidf_feature_names,
            movie_metadata=movie_metadata,
            tmdb_id_to_idx=tmdb_id_to_idx,
            idx_to_tmdb_id=idx_to_tmdb_id
        )

    def build_and_save(
        self,
        batch_size: int = 2000,
        encode_batch_size: Optional[int] = None
    ) -> FeatureMatrix:
        movies = self.load_canonical_dataset()
        feature_matrix = self.build_matrices(
            movies=movies,
            batch_size=batch_size,
            encode_batch_size=encode_batch_size
        )
        self._save_feature_matrix(feature_matrix)
        return feature_matrix

    def _save_feature_matrix(self, feature_matrix: FeatureMatrix):
        from datetime import datetime, timezone
        print(f"Saving precomputed artifacts to {self.output_dir}...", file=sys.stderr)
        
        # 1. Save embeddings matrix as numpy file
        emb_path = self.output_dir / "movie_embeddings.npy"
        np.save(emb_path, feature_matrix.semantic_embeddings)
        
        # 2. Save vectorizers
        with open(self.output_dir / "count_vectorizer.pkl", "wb") as f:
            pickle.dump(self.count_vectorizer, f)
        
        with open(self.output_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
            
        # 3. Save persistent sparse matrices
        sp.save_npz(self.output_dir / "count_matrix.npz", feature_matrix.count_matrix)
        sp.save_npz(self.output_dir / "tfidf_matrix.npz", feature_matrix.tfidf_matrix)
        
        # 4. Save ID mapping
        with open(self.output_dir / "movie_id_map.json", "w") as f:
            json.dump({
                "tmdb_id_to_idx": {str(k): v for k, v in feature_matrix.tmdb_id_to_idx.items()},
                "movie_count": len(feature_matrix.movie_ids)
            }, f)
        
        # 5. Save FeatureMatrix instance for backward compatibility
        with open(self.output_dir / "feature_matrix.pkl", "wb") as f:
            pickle.dump(feature_matrix, f)
        
        # 6. Write precompute manifest
        manifest_path = self.output_dir / "precompute_manifest.json"
        dataset_hash = self._compute_dataset_hash(feature_matrix.movie_ids)
        
        device_str = "cpu"
        if torch.cuda.is_available():
            device_str = f"cuda ({torch.cuda.get_device_name(0)})"

        manifest_data = {
            "model_name": self.semantic_model_name,
            "embedding_dimension": int(feature_matrix.semantic_embeddings.shape[1]),
            "canonical_movie_count": len(feature_matrix.movie_ids),
            "movie_id_hash": dataset_hash,
            "dataset_hash": dataset_hash,
            "embedding_shape": list(feature_matrix.semantic_embeddings.shape),
            "count_matrix_shape": list(feature_matrix.count_matrix.shape),
            "tfidf_matrix_shape": list(feature_matrix.tfidf_matrix.shape),
            "count_features": len(feature_matrix.feature_names),
            "tfidf_features": len(feature_matrix.tfidf_feature_names),
            "dtype": str(feature_matrix.semantic_embeddings.dtype),
            "normalization_status": "L2-normalized",
            "generation_device": device_str,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_status": "complete",
            "preprocessing_settings": {
                "max_seq_length": 128,
                "count_max_features": 5000,
                "tfidf_max_features": 5000,
                "ngram_range": [1, 2],
                "stop_words": "english"
            }
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        
        print(f"Precomputed feature matrix and artifacts saved successfully ({len(feature_matrix.movie_ids)} movies).", file=sys.stderr)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> Tuple['FeatureMatrixBuilder', FeatureMatrix, Optional[Dict]]:
        builder = cls(config_path)
        output_dir = Path("data/processed")
        
        emb_path = output_dir / "movie_embeddings.npy"
        map_path = output_dir / "movie_id_map.json"
        cv_path = output_dir / "count_vectorizer.pkl"
        tv_path = output_dir / "tfidf_vectorizer.pkl"
        cnt_npz = output_dir / "count_matrix.npz"
        tfd_npz = output_dir / "tfidf_matrix.npz"
        
        # Verify required authoritative artifacts exist
        missing = [str(p) for p in [emb_path, map_path, cv_path, tv_path, cnt_npz, tfd_npz] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing precomputed authoritative artifacts: {missing}. "
                f"Please run `python main.py precompute`."
            )
            
        # 1. Load vectorizers
        with open(cv_path, "rb") as f:
            builder.count_vectorizer = pickle.load(f)
            
        with open(tv_path, "rb") as f:
            builder.tfidf_vectorizer = pickle.load(f)
            
        # 2. Load ID mapping
        with open(map_path, "r") as f:
            id_map_data = json.load(f)
        tmdb_id_to_idx = {int(k): v for k, v in id_map_data["tmdb_id_to_idx"].items()}
        idx_to_tmdb_id = {v: int(k) for k, v in tmdb_id_to_idx.items()}
        
        # 3. Load embeddings
        semantic_embeddings = np.load(emb_path)
        expected_dim = 768
        actual_dim = semantic_embeddings.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: found {actual_dim}d, expected {expected_dim}d ({builder.semantic_model_name}). "
                f"Please run `python main.py precompute` to generate valid 768-d embeddings."
            )
            
        # 4. Load canonical dataset and metadata from SQLite
        canonical_movies = builder.load_canonical_dataset()
        movie_ids = [m['tmdb_id'] for m in canonical_movies]
        movie_titles = [m['title'] for m in canonical_movies]
        movie_metadata = {m['tmdb_id']: m for m in canonical_movies}
        
        # 5. Load persistent sparse count and TF-IDF matrices
        count_matrix = sp.load_npz(cnt_npz)
        tfidf_matrix = sp.load_npz(tfd_npz)
        
        if count_matrix.shape != (len(movie_ids), 5000):
            raise ValueError(f"Count matrix shape mismatch: {count_matrix.shape}, expected ({len(movie_ids)}, 5000)")
        if tfidf_matrix.shape != (len(movie_ids), 5000):
            raise ValueError(f"TF-IDF matrix shape mismatch: {tfidf_matrix.shape}, expected ({len(movie_ids)}, 5000)")
            
        feature_names = builder.count_vectorizer.get_feature_names_out().tolist()
        tfidf_feature_names = builder.tfidf_vectorizer.get_feature_names_out().tolist()
        
        # 6. Construct lightweight compatibility container
        feature_matrix = FeatureMatrix(
            movie_ids=movie_ids,
            movie_titles=movie_titles,
            count_matrix=count_matrix,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=semantic_embeddings,
            feature_names=feature_names,
            tfidf_feature_names=tfidf_feature_names,
            movie_metadata=movie_metadata,
            tmdb_id_to_idx=tmdb_id_to_idx,
            idx_to_tmdb_id=idx_to_tmdb_id
        )
            
        return builder, feature_matrix, None

    def transform_query(self, query: str) -> Tuple[Any, Any, np.ndarray]:
        count_vec = self.count_vectorizer.transform([query])
        tfidf_vec = self.tfidf_vectorizer.transform([query])
        semantic_vec = self.semantic_model.encode([query], normalize_embeddings=True)
        return count_vec, tfidf_vec, semantic_vec


def main():
    builder = FeatureMatrixBuilder()
    builder.build_and_save()


if __name__ == "__main__":
    main()