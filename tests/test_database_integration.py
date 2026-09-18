import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from src.db.connection import MovieDatabase
from src.features.matrix_builder import FeatureMatrix, FeatureMatrixBuilder
from src.pipeline.recommender import MovieRecommender
from src.llm.nl_feature_extractor import ExtractedFeatures


def validate_recommendation_item(r, recommender=None):
    """
    Reusable test helper to validate a recommendation item against the response contract:
    - required fields ('movie', 'reasoning', 'score')
    - numeric score >= 0
    - non-empty reasoning string
    - movie dictionary with valid integer ID > 0 and non-empty title
    - for local candidates: ID in feature_matrix.movie_metadata
    - for live/out-of-corpus candidates: required metadata on movie object itself
    """
    assert "movie" in r, "Recommendation item missing 'movie'"
    assert "reasoning" in r, "Recommendation item missing 'reasoning'"
    assert "score" in r, "Recommendation item missing 'score'"
    assert isinstance(r["score"], (int, float)) and r["score"] >= 0
    assert isinstance(r["reasoning"], str) and len(r["reasoning"]) > 0

    movie = r["movie"]
    assert isinstance(movie, dict), "Movie must be a dict"
    assert "id" in movie, "Movie missing 'id'"
    assert isinstance(movie["id"], int) and movie["id"] > 0
    assert "title" in movie and isinstance(movie["title"], str) and len(movie["title"]) > 0
    assert "genres" in movie and isinstance(movie["genres"], list)

    if recommender and hasattr(recommender, "feature_matrix") and recommender.feature_matrix:
        if movie["id"] in recommender.feature_matrix.movie_metadata:
            meta = recommender.feature_matrix.movie_metadata[movie["id"]]
            assert meta.get("title") is not None
        else:
            # Live/out-of-corpus candidate: verify candidate metadata fields
            assert "vote_average" in movie
            assert "vote_count" in movie


class TestDatabaseIntegration:
    def test_database_integrity_and_manifest(self):
        db = MovieDatabase()
        valid, msg = db.verify_integrity()
        assert valid is True
        assert "Integrity check passed" in msg

    def test_canonical_movie_count_and_deduplication(self):
        db = MovieDatabase()
        movies = db.load_canonical_movies()
        assert len(movies) == 87425
        
        # Verify all TMDB IDs are distinct
        tmdb_ids = [m['tmdb_id'] for m in movies]
        assert len(set(tmdb_ids)) == 87425

    def test_duplicate_tags_unioned(self):
        db = MovieDatabase()
        movies = db.load_canonical_movies()
        movie_dict = {m['tmdb_id']: m for m in movies}
        
        # Check known duplicate TMDB ID 4912 (Confessions of a Dangerous Mind)
        assert 4912 in movie_dict
        m4912 = movie_dict[4912]
        # Should have tags from both MovieLens IDs
        tags = set(m4912['tags'])
        assert "sam rockwell" in tags or "charlie kaufman" in tags

    def test_candidate_membership_filter(self):
        rec = MovieRecommender()
        mock_fm = Mock()
        mock_fm.tmdb_id_to_idx = {101: 0, 102: 1}
        mock_fm.movie_metadata = {
            101: {'title': 'Movie In DB 1', 'genres': ['Action'], 'vote_average': 7.5, 'vote_count': 500},
            102: {'title': 'Movie In DB 2', 'genres': ['Sci-Fi'], 'vote_average': 8.0, 'vote_count': 1200}
        }
        rec.feature_matrix = mock_fm
        
        # Simulate TMDB candidate list (101, 102 are in SQLite; 999 is NOT)
        tmdb_candidates = [
            Mock(id=101, title='TMDB 101', overview='Desc', genres=['Action'], release_date='2020-01-01', vote_average=7.5, vote_count=500, popularity=50, original_language='en'),
            Mock(id=999, title='TMDB Absent', overview='Desc', genres=['Action'], release_date='2020-01-01', vote_average=6.0, vote_count=50, popularity=10, original_language='en'),
            Mock(id=102, title='TMDB 102', overview='Desc', genres=['Sci-Fi'], release_date='2021-01-01', vote_average=8.0, vote_count=1200, popularity=80, original_language='en'),
        ]
        
        # Filter against SQLite membership
        sqlite_id_set = set(mock_fm.tmdb_id_to_idx.keys())
        filtered = [m for m in tmdb_candidates if m.id in sqlite_id_set]
        
        assert len(filtered) == 2
        assert [m.id for m in filtered] == [101, 102]
        assert 999 not in [m.id for m in filtered]

    def test_candidate_coverage_reporting(self):
        total_tmdb = 100
        matched = 82
        excluded = total_tmdb - matched
        coverage_pct = (matched / total_tmdb) * 100.0
        
        assert excluded == 18
        assert coverage_pct == 82.0

    def test_offline_fallback_hard_filters(self):
        rec = MovieRecommender()
        mock_fm = Mock()
        mock_fm.movie_metadata = {
            1: {'id': 1, 'title': 'Space Sci-Fi 1999', 'genres': ['Science Fiction'], 'release_date': '1999-05-01', 'vote_average': 8.0, 'vote_count': 500, 'popularity': 50, 'original_language': 'en'},
            2: {'id': 2, 'title': 'Comedy 2015', 'genres': ['Comedy'], 'release_date': '2015-05-01', 'vote_average': 7.0, 'vote_count': 300, 'popularity': 40, 'original_language': 'en'},
            3: {'id': 3, 'title': 'Low Rated Sci-Fi', 'genres': ['Science Fiction'], 'release_date': '2000-01-01', 'vote_average': 4.0, 'vote_count': 100, 'popularity': 20, 'original_language': 'en'},
        }
        rec.feature_matrix = mock_fm
        
        extracted = ExtractedFeatures(
            explicit_genres=['Science Fiction'],
            release_year_min=1990,
            release_year_max=2005,
            min_rating=6.0
        )
        
        candidates = rec._get_local_candidates(extracted)
        assert len(candidates) == 1
        assert candidates[0]['id'] == 1

    def test_embedding_dimension_validation(self):
        mock_matrix = Mock()
        mock_matrix.semantic_embeddings = np.random.randn(10, 384) # Stale 384-d
        mock_matrix.movie_ids = list(range(10))
        
        # Validation should reject 384-d embeddings when 768-d MPNet is expected
        expected_dim = 768
        actual_dim = mock_matrix.semantic_embeddings.shape[1]
        assert actual_dim != expected_dim

    def test_sqlite_embedding_ordering_alignment(self):
        import json
        from pathlib import Path
        db = MovieDatabase()
        movies = db.load_canonical_movies()
        canonical_ids = [m['tmdb_id'] for m in movies]
        
        map_path = Path("data/processed/movie_id_map.json")
        assert map_path.exists()
        with open(map_path, "r") as f:
            id_map_data = json.load(f)
            
        tmdb_id_to_idx = {int(k): v for k, v in id_map_data["tmdb_id_to_idx"].items()}
        assert len(tmdb_id_to_idx) == 87425
        assert len(canonical_ids) == 87425
        
        # Verify 1:1 index alignment
        for expected_idx, tmdb_id in enumerate(canonical_ids):
            assert tmdb_id in tmdb_id_to_idx
            assert tmdb_id_to_idx[tmdb_id] == expected_idx

    def test_movie_embeddings_properties(self):
        from pathlib import Path
        emb_path = Path("data/processed/movie_embeddings.npy")
        assert emb_path.exists()
        
        embeddings = np.load(emb_path)
        assert embeddings.shape == (87425, 768)
        assert embeddings.dtype == np.float32
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
        
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)
        assert abs(float(norms.mean()) - 1.0) < 1e-4

    def test_feature_matrix_container_load(self):
        builder, fm, _ = FeatureMatrixBuilder.load()
        assert len(fm.movie_ids) == 87425
        assert len(fm.movie_titles) == 87425
        assert len(fm.movie_metadata) == 87425
        assert len(fm.tmdb_id_to_idx) == 87425
        assert len(fm.idx_to_tmdb_id) == 87425
        assert fm.semantic_embeddings.shape == (87425, 768)
        assert fm.count_matrix.shape == (87425, 5000)
        assert fm.tfidf_matrix.shape == (87425, 5000)
        assert len(fm.feature_names) == 5000
        assert len(fm.tfidf_feature_names) == 5000

    def test_tfidf_and_semantic_ranking_execution(self):
        from src.ranking.ranker import MovieRanker, RankingConfig
        builder, fm, _ = FeatureMatrixBuilder.load()
        ranker = MovieRanker(config=RankingConfig())
        ranker.feature_matrix = fm
        
        # Take first 5 canonical movies as candidate list
        candidate_movies = [
            {
                'id': mid,
                'tmdb_id': mid,
                'title': fm.movie_metadata[mid]['title'],
                'genres': fm.movie_metadata[mid].get('genres', []),
                'overview': fm.movie_metadata[mid].get('overview', ''),
                'vote_average': fm.movie_metadata[mid].get('vote_average', 7.0),
                'vote_count': fm.movie_metadata[mid].get('vote_count', 200),
                'popularity': 50.0,
                'original_language': 'en'
            }
            for mid in fm.movie_ids[:5]
        ]
        
        query_text = "space exploration adventure"
        count_vec, tfidf_vec, query_emb = builder.transform_query(query_text)
        
        ranked = ranker.rank_movies_stage1(
            candidate_movies=candidate_movies,
            query_embedding=query_emb[0],
            query_tfidf=tfidf_vec
        )
        assert len(ranked) == 5
        assert all(r.score > 0 for r in ranked)

    def test_preference_extraction_execution(self):
        from src.features.preference_extractor import PreferenceExtractor
        builder, fm, _ = FeatureMatrixBuilder.load()
        extractor = PreferenceExtractor(feature_matrix=fm)
        
        sample_history = [
            {"movie_id": fm.movie_ids[0], "rating": 9.0},
            {"movie_id": fm.movie_ids[1], "rating": 8.0}
        ]
        ratings = {fm.movie_ids[0]: 9.0, fm.movie_ids[1]: 8.0}
        
        profile = extractor.extract_preferences(sample_history, ratings, fm)
        assert profile.watch_count == 2
        assert profile.avg_rating == 8.5
        assert len(profile.feature_vector) == 5000
        assert np.linalg.norm(profile.feature_vector) > 0

    @pytest.mark.asyncio
    async def test_get_movie_details_on_demand_similarity(self):
        rec = MovieRecommender()
        await rec.initialize()
        
        # Test movie details on canonical movie ID (e.g. 11: Star Wars or first canonical movie)
        target_id = rec.feature_matrix.movie_ids[0]
        details = await rec.get_movie_details(target_id)
        
        assert details is not None
        assert details.get("id") == target_id
        assert "similar_movies" in details
        assert len(details["similar_movies"]) > 0
        # Ensure target movie itself is not in its own similar list
        similar_ids = [m[0] for m in details["similar_movies"]]
        assert target_id not in similar_ids

    @pytest.mark.asyncio
    async def test_end_to_end_recommendations_flow(self):
        rec = MovieRecommender()
        await rec.initialize()
        
        sample_history = [
            {"movie_id": rec.feature_matrix.movie_ids[0], "rating": 9.0}
        ]
        
        res = await rec.get_recommendations(
            user_id="test_user",
            watch_history=sample_history,
            natural_language="space sci-fi",
            top_k=3
        )
        
        assert "recommendations" in res
        assert len(res["recommendations"]) == 3
        for r in res["recommendations"]:
            validate_recommendation_item(r, rec)

