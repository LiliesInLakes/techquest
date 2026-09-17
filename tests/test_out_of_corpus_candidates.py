import pytest
import asyncio
import numpy as np
from scipy.sparse import csr_matrix
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.recommender import MovieRecommender
from src.ranking.ranker import MovieRanker, RankingConfig, RankedMovie
from src.llm.nl_feature_extractor import ExtractedFeatures
from src.tmdb.client import Movie as TMDBMovie


@pytest.fixture
def recommender():
    rec = MovieRecommender()
    return rec


@pytest.fixture
def ranker():
    config = RankingConfig()
    return MovieRanker(config)


@pytest.mark.asyncio
async def test_out_of_corpus_horror_after_2024(recommender):
    """
    Test query: 'horror movie of 2hrs and released after 2024'
    Must return recommendations (out-of-corpus TMDB candidates) rather than 0 results.
    """
    await recommender.initialize()
    res = await recommender.get_recommendations(
        user_id="test_user",
        natural_language="horror movie of 2hrs and released after 2024",
        top_k=5
    )
    recs = res.get("recommendations", [])
    assert len(recs) > 0, "Expected non-zero recommendations for horror movie after 2024"
    for r in recs:
        movie = r["movie"]
        # Hard filters preserved: horror genre
        genres = [g.lower() for g in movie.get("genres", [])]
        assert "horror" in genres, f"Movie {movie.get('title')} does not have horror genre: {genres}"
        # Hard filters preserved: release year >= 2024
        rd = movie.get("release_date", "")
        if rd and len(rd) >= 4:
            year = int(rd[:4])
            assert year >= 2024, f"Movie {movie.get('title')} has year {year} < 2024"


@pytest.mark.asyncio
async def test_latest_horror_movie_temporal_window(recommender):
    """
    Test query: 'latest horror movie'
    Must return horror movies within the configured temporal window (2023+).
    """
    await recommender.initialize()
    res = await recommender.get_recommendations(
        user_id="test_user",
        natural_language="latest horror movie",
        top_k=5
    )
    recs = res.get("recommendations", [])
    assert len(recs) > 0, "Expected non-zero recommendations for latest horror movie"
    for r in recs:
        movie = r["movie"]
        rd = movie.get("release_date", "")
        if rd and len(rd) >= 4:
            year = int(rd[:4])
            assert year >= 2023, f"Movie {movie.get('title')} has year {year} < 2023 for 'latest'"


@pytest.mark.asyncio
async def test_recent_scifi_outside_local_corpus(recommender):
    """
    Test query: recent sci-fi movie outside local corpus (e.g. released in 2025).
    """
    await recommender.initialize()
    res = await recommender.get_recommendations(
        user_id="test_user",
        natural_language="sci-fi movie released in 2025",
        top_k=5
    )
    recs = res.get("recommendations", [])
    assert len(recs) > 0, "Expected non-zero recommendations for 2025 sci-fi movie"
    for r in recs:
        movie = r["movie"]
        genres_norm = [recommender._normalize_genre_name(g) for g in movie.get("genres", [])]
        assert "science fiction" in genres_norm, f"Expected science fiction in {genres_norm}"
        rd = movie.get("release_date", "")
        if rd and len(rd) >= 4:
            assert int(rd[:4]) == 2025


@pytest.mark.asyncio
async def test_older_query_continues_using_sqlite(recommender):
    """
    Test query: 'classic 90s sci-fi movie'
    Should continue returning canonical SQLite movies with full features.
    """
    await recommender.initialize()
    res = await recommender.get_recommendations(
        user_id="test_user",
        natural_language="sci-fi movie released between 1990 and 1999",
        top_k=5
    )
    recs = res.get("recommendations", [])
    assert len(recs) > 0
    for r in recs:
        movie = r["movie"]
        rd = movie.get("release_date", "")
        if rd and len(rd) >= 4:
            year = int(rd[:4])
            assert 1990 <= year <= 1999


def test_mixed_local_and_tmdb_candidates_no_duplicates(ranker):
    """
    Test ranking with a mix of local corpus candidates and live TMDB candidates.
    Ensures no duplicate IDs and both paths compute valid scores.
    """
    local_cand = {
        'id': 1001,
        'title': 'Local Movie',
        'overview': 'A great space adventure.',
        'genres': ['Science Fiction', 'Adventure'],
        'release_date': '2020-01-01',
        'vote_average': 8.0,
        'vote_count': 1500,
        'is_local': True
    }
    live_cand = {
        'id': 999999,
        'title': 'Live Future Movie',
        'overview': 'A terrifying futuristic space horror in deep space.',
        'genres': ['Horror', 'Science Fiction'],
        'release_date': '2025-06-01',
        'vote_average': 7.5,
        'vote_count': 300,
        'is_local': False,
        'tfidf_vector': csr_matrix(np.zeros((1, 10))),
        'semantic_embedding': np.random.randn(768).astype(np.float32)
    }

    query_emb = np.random.randn(768).astype(np.float32)
    query_tfidf = csr_matrix(np.zeros((1, 10)))

    ranked = ranker.rank_movies_stage1([local_cand, live_cand], query_emb, query_tfidf)
    assert len(ranked) == 2
    ids = [r.movie_id for r in ranked]
    assert len(ids) == len(set(ids)), "Duplicate candidate IDs found"
    assert set(ids) == {1001, 999999}
    for r in ranked:
        assert 0.0 <= r.score <= 1.0
        assert r.imdb_score >= 0.0
        assert r.popularity_score >= 0.0


def test_hard_filters_enforced_on_tmdb_candidates(recommender):
    """
    Ensures _passes_all_hard_filters strictly filters out non-matching TMDB candidates.
    """
    extracted = ExtractedFeatures(
        explicit_genres=["Horror"],
        release_year_min=2024,
        release_year_max=2026,
        exclude_genres=["Comedy"]
    )

    # 1. Matching candidate
    cand_ok = {
        'id': 101,
        'title': 'Good Horror',
        'genres': ['Horror', 'Thriller'],
        'release_date': '2025-01-10',
        'is_local': False
    }
    assert recommender._passes_all_hard_filters(cand_ok, extracted) is True

    # 2. Fails genre
    cand_wrong_genre = {
        'id': 102,
        'title': 'Action Movie',
        'genres': ['Action', 'Thriller'],
        'release_date': '2025-01-10',
        'is_local': False
    }
    assert recommender._passes_all_hard_filters(cand_wrong_genre, extracted) is False

    # 3. Fails year (too old)
    cand_old = {
        'id': 103,
        'title': 'Old Horror',
        'genres': ['Horror'],
        'release_date': '2022-01-10',
        'is_local': False
    }
    assert recommender._passes_all_hard_filters(cand_old, extracted) is False

    # 4. Fails excluded genre
    cand_excl = {
        'id': 104,
        'title': 'Horror Comedy',
        'genres': ['Horror', 'Comedy'],
        'release_date': '2025-01-10',
        'is_local': False
    }
    assert recommender._passes_all_hard_filters(cand_excl, extracted) is False


def test_no_fabricated_data_for_tmdb_candidates():
    """
    Verify that TMDB candidates with missing fields use safe neutral defaults,
    and do not invent rating counts or streaming providers.
    """
    ranker = MovieRanker(RankingConfig())
    cand_sparse = {
        'id': 888888,
        'title': 'Indie Release',
        'release_date': '2025-01-01',
        'vote_average': 0.0,
        'vote_count': 0,
        'is_local': False
    }
    pop_score = ranker._compute_popularity_score(cand_sparse)
    assert pop_score == 0.0, "Expected 0.0 popularity score for 0 vote count without fabrication"

    imdb_score = ranker._compute_imdb_score(cand_sparse)
    assert imdb_score == 0.0


def test_missing_tmdb_overview_and_keywords_handled_safely():
    """
    Verify that out-of-corpus candidates with missing overview/keywords do not raise exceptions.
    """
    ranker = MovieRanker(RankingConfig())
    cand_sparse = {
        'id': 777777,
        'title': 'No Metadata Movie',
        'overview': None,
        'genres': None,
        'release_date': None,
        'is_local': False,
        'tfidf_vector': None,
        'semantic_embedding': None
    }
    query_emb = np.zeros(768, dtype=np.float32)
    query_tfidf = csr_matrix(np.zeros((1, 10)))

    # Ranking should not crash
    ranked = ranker.rank_movies_stage1([cand_sparse], query_emb, query_tfidf)
    assert len(ranked) == 1
    assert ranked[0].movie_id == 777777
    assert ranked[0].score >= 0.0
