import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from src.pipeline.recommender import MovieRecommender, MAX_DYNAMIC_LIVE_CANDIDATES
from src.llm.nl_feature_extractor import ExtractedFeatures


def _create_mock_candidate(movie_id: int, is_local: bool = False):
    return {
        'id': movie_id,
        'title': f'Movie {movie_id}',
        'overview': f'Overview for movie {movie_id}',
        'genres': ['Horror', 'Thriller'],
        'release_date': '2025-01-01',
        'vote_average': 7.5,
        'vote_count': 500,
        'popularity': 50.0,
        'is_local': is_local
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("live_count,expected_encoded", [
    (0, 0),
    (50, 50),
    (200, 200),
    (500, 200),
    (5000, 200),
])
async def test_dynamic_embedding_candidate_cap(live_count, expected_encoded):
    """
    Test that dynamic semantic embedding is strictly bounded by MAX_DYNAMIC_LIVE_CANDIDATES (200),
    and that SentenceTransformer.encode() NEVER receives more than 200 candidates even if 5,000 are supplied.
    """
    rec = MovieRecommender()

    # Mock matrix builder and semantic model
    mock_matrix_builder = MagicMock()
    mock_semantic_model = MagicMock()

    def fake_encode(texts, **kwargs):
        # Verify that the input passed to encode() NEVER exceeds the 200 hard cap
        assert len(texts) <= MAX_DYNAMIC_LIVE_CANDIDATES, f"SentenceTransformer.encode() received {len(texts)} > {MAX_DYNAMIC_LIVE_CANDIDATES}"
        assert len(texts) == expected_encoded
        # Return exact batch size embeddings array
        return np.zeros((len(texts), 768), dtype=np.float32)

    mock_semantic_model.encode.side_effect = fake_encode
    mock_matrix_builder.semantic_model = mock_semantic_model

    mock_tfidf = MagicMock()
    def fake_transform(texts):
        assert len(texts) <= MAX_DYNAMIC_LIVE_CANDIDATES
        assert len(texts) == expected_encoded
        return [MagicMock() for _ in range(len(texts))]
    mock_tfidf.transform.side_effect = fake_transform
    mock_matrix_builder.tfidf_vectorizer = mock_tfidf
    mock_matrix_builder.transform_query.return_value = (None, None, np.zeros(768, dtype=np.float32))

    rec.matrix_builder = mock_matrix_builder
    rec.feature_matrix = MagicMock()
    rec.feature_matrix.movie_metadata = {}
    rec.feature_matrix.feature_names = []

    # Mock ranker and diversity filter
    rec.ranker = MagicMock()
    rec.ranker.rank_movies_stage1.return_value = []
    rec.ranker.get_top_k.return_value = []
    rec.ranker.rank_movies_stage2.return_value = []
    rec.diversity_filter = MagicMock()
    rec.diversity_filter.filter.return_value = []

    # Create candidate pool
    simulated_candidates = [_create_mock_candidate(i, is_local=False) for i in range(1, live_count + 1)]

    # Mock candidate discovery to return our simulated candidates
    rec._get_candidate_movies_stage1 = AsyncMock(return_value=simulated_candidates)
    rec.nl_extractor = MagicMock()
    rec.nl_extractor.extract = AsyncMock(return_value=ExtractedFeatures())
    rec.nl_extractor.map_to_vocabulary.return_value = ExtractedFeatures()
    rec.preference_extractor = MagicMock()
    rec.preference_extractor.extract_preferences.return_value = MagicMock()

    with patch.object(rec, 'initialize', new=AsyncMock()):
        await rec.get_recommendations(
            user_id="test_cap",
            natural_language="horror movie after 2024",
            top_k=5
        )

    if expected_encoded > 0:
        assert mock_semantic_model.encode.called
        call_args = mock_semantic_model.encode.call_args[0][0]
        assert len(call_args) == expected_encoded
        assert len(call_args) <= MAX_DYNAMIC_LIVE_CANDIDATES
    else:
        assert not mock_semantic_model.encode.called


@pytest.mark.asyncio
async def test_memory_safety_no_large_array_allocation():
    """
    Verify memory safety: when 5,000 live candidates exist, the system
    slices the candidate list BEFORE creating embedding arrays or text payloads.
    """
    rec = MovieRecommender()

    encoded_shapes = []
    mock_semantic = MagicMock()
    def record_encode(texts, **kwargs):
        encoded_shapes.append(len(texts))
        return np.zeros((len(texts), 768), dtype=np.float32)
    mock_semantic.encode.side_effect = record_encode

    mock_mb = MagicMock()
    mock_mb.semantic_model = mock_semantic
    mock_mb.tfidf_vectorizer = MagicMock()
    mock_mb.tfidf_vectorizer.transform.return_value = [MagicMock() for _ in range(200)]
    mock_mb.transform_query.return_value = (None, None, np.zeros(768, dtype=np.float32))

    rec.matrix_builder = mock_mb
    rec.feature_matrix = MagicMock()
    rec.feature_matrix.movie_metadata = {}
    rec.feature_matrix.feature_names = []
    rec.ranker = MagicMock()
    rec.ranker.rank_movies_stage1.return_value = []
    rec.ranker.get_top_k.return_value = []
    rec.ranker.rank_movies_stage2.return_value = []
    rec.diversity_filter = MagicMock()
    rec.diversity_filter.filter.return_value = []

    candidates_5k = [_create_mock_candidate(i, is_local=False) for i in range(1, 5001)]
    rec._get_candidate_movies_stage1 = AsyncMock(return_value=candidates_5k)
    rec.nl_extractor = MagicMock()
    rec.nl_extractor.extract = AsyncMock(return_value=ExtractedFeatures())
    rec.nl_extractor.map_to_vocabulary.return_value = ExtractedFeatures()
    rec.preference_extractor = MagicMock()
    rec.preference_extractor.extract_preferences.return_value = MagicMock()

    with patch.object(rec, 'initialize', new=AsyncMock()):
        await rec.get_recommendations(user_id="test_mem", natural_language="huge test", top_k=5)

    assert len(encoded_shapes) == 1
    assert encoded_shapes[0] == 200, f"Expected exactly 200 candidates encoded, got {encoded_shapes[0]}"
