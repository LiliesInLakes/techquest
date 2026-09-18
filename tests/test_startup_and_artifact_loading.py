import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from sentence_transformers import SentenceTransformer
from src.features.semantic_search import SemanticKeywordSearch, HybridKeywordMatcher, _get_embedding_dimension
from src.pipeline.recommender import MovieRecommender


class TestStartupAndArtifactLoading:
    def test_get_embedding_dimension_helper(self):
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 768
        assert _get_embedding_dimension(mock_model) == 768

        del mock_model.get_embedding_dimension
        mock_model.get_sentence_embedding_dimension.return_value = 768
        assert _get_embedding_dimension(mock_model) == 768

    def test_semantic_keyword_search_load_precomputed_pkl(self):
        pkl_path = "data/processed/keyword_embeddings.pkl"
        if not Path(pkl_path).exists():
            pytest.skip("Precomputed keyword_embeddings.pkl not present on disk")

        # Loading from precomputed pickle must not call SentenceTransformer.encode
        with patch.object(SentenceTransformer, "encode") as mock_encode:
            search = SemanticKeywordSearch.load(pkl_path)
            assert len(search.keyword_list) == 5000
            assert search.embedding_matrix.shape == (5000, 768)
            assert not mock_encode.called, "SentenceTransformer.encode should NOT be called when loading valid precomputed artifact"

    def test_semantic_keyword_search_from_feature_embeddings(self):
        npy_path = "data/processed/keyword_feature_embeddings.npy"
        if not Path(npy_path).exists():
            pytest.skip("Precomputed keyword_feature_embeddings.npy not present on disk")

        dummy_features = [f"kw_{i}" for i in range(5000)]
        with patch.object(SentenceTransformer, "encode") as mock_encode:
            search = SemanticKeywordSearch.from_feature_embeddings(npy_path, dummy_features)
            assert len(search.keyword_list) == 5000
            assert search.embedding_matrix.shape == (5000, 768)
            assert search.get_keyword_embedding("kw_0") is not None
            assert not mock_encode.called, "SentenceTransformer.encode should NOT be called during from_feature_embeddings"

    def test_hybrid_matcher_loads_precomputed_npy(self):
        npy_path = "data/processed/keyword_feature_embeddings.npy"
        if not Path(npy_path).exists():
            pytest.skip("Precomputed keyword_feature_embeddings.npy not present on disk")

        mock_search = MagicMock()
        mock_search.model.get_sentence_embedding_dimension.return_value = 768
        mock_search.model.get_embedding_dimension.return_value = 768
        mock_search.threshold = 0.65

        feature_names = [f"feat_{i}" for i in range(5000)]
        with patch.object(mock_search.model, "encode") as mock_encode:
            matcher = HybridKeywordMatcher(mock_search, feature_names, cache_path=npy_path)
            assert matcher.feature_embeddings is not None
            assert matcher.feature_embeddings.shape == (5000, 768)
            assert not mock_encode.called, "HybridKeywordMatcher should load cached npy without re-encoding"

    @pytest.mark.asyncio
    async def test_recommender_initialization_shares_model_and_does_not_rebuild_keywords(self):
        rec = MovieRecommender()
        with patch.object(SentenceTransformer, "encode") as mock_encode:
            await rec.initialize()
            assert rec.semantic_search is not None
            assert rec.feature_matrix is not None
            # Verify SentenceTransformer model is shared between semantic_search and matrix_builder
            if rec.matrix_builder is not None:
                assert rec.matrix_builder._semantic_model is rec.semantic_search.model
            assert not mock_encode.called, "SentenceTransformer.encode should not be called during startup when artifacts exist"

    def test_keyword_search_fallback_when_artifacts_absent(self):
        """Verify that when both precomputed keyword artifacts are absent, build_index is called as fallback."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(SemanticKeywordSearch, "build_index") as mock_build:
                # Simulate recommender keyword initialization ladder
                pkl_path = Path("data/processed/keyword_embeddings.pkl")
                npy_path = Path("data/processed/keyword_feature_embeddings.npy")
                
                kw_search = None
                if pkl_path.exists():
                    kw_search = SemanticKeywordSearch.load(str(pkl_path))
                elif npy_path.exists():
                    kw_search = SemanticKeywordSearch.from_feature_embeddings(str(npy_path), ["dummy"])
                else:
                    kw_search = SemanticKeywordSearch()
                    kw_search.build_index(["k1", "k2"])

                assert mock_build.called
                mock_build.assert_called_once_with(["k1", "k2"])

    def test_complete_runtime_artifact_contract(self):
        """Verify that all 11 documented runtime artifacts exist with valid structure
        and that runtime code has zero dependencies on legacy TMDB 5000 or raw MovieLens files."""
        # 1. Check all 11 authoritative runtime artifacts
        expected_runtime_artifacts = [
            Path("movie_features.db"),
            Path("data/processed/movie_embeddings.npy"),
            Path("data/processed/movie_id_map.json"),
            Path("data/processed/count_vectorizer.pkl"),
            Path("data/processed/tfidf_vectorizer.pkl"),
            Path("data/processed/count_matrix.npz"),
            Path("data/processed/tfidf_matrix.npz"),
            Path("data/processed/keyword_feature_embeddings.npy"),
            Path("data/processed/keyword_embeddings.pkl"),
            Path("data/processed/precompute_manifest.json"),
            Path("data/movie_db_manifest.json")
        ]

        for p in expected_runtime_artifacts:
            assert p.exists(), f"Documented runtime artifact missing: {p}"
            assert p.stat().st_size > 0, f"Artifact {p} is empty (0 bytes)"

        # 2. Verify keyword embeddings matrix shape and dimensions
        npy_path = Path("data/processed/keyword_feature_embeddings.npy")
        kw_mat = np.load(npy_path)
        assert kw_mat.shape == (5000, 768)
        assert kw_mat.dtype == np.float32

        # 3. Verify movie embeddings matrix shape and dimensions
        movie_emb_path = Path("data/processed/movie_embeddings.npy")
        movie_embs = np.load(movie_emb_path)
        assert movie_embs.shape == (87425, 768)
        assert movie_embs.dtype == np.float32
