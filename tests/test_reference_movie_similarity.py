import pytest
import numpy as np
from src.llm.nl_feature_extractor import NLFeatureExtractor
from src.pipeline.recommender import MovieRecommender


class TestReferenceMovieSimilarity:
    def test_reference_movie_extraction_patterns(self):
        """Test extraction of reference movie titles across various phrasing patterns."""
        extractor = NLFeatureExtractor()
        extractor._llm_available = False

        f1 = extractor._fallback_extraction("movies similar to Harry Potter", {})
        assert "Harry Potter" in f1.reference_movies
        assert "similar to" not in f1.semantic_search_query.lower()
        assert "harry potter" not in f1.semantic_search_query.lower()

        f2 = extractor._fallback_extraction("movies like The Dark Knight", {})
        assert "The Dark Knight" in f2.reference_movies
        # Actor regex must not be polluted by words from reference movie title
        assert "The Dark" not in f2.actors

        f3 = extractor._fallback_extraction("something similar to Interstellar", {})
        assert "Interstellar" in f3.reference_movies

    @pytest.mark.asyncio
    async def test_reference_movie_title_resolution(self):
        """Test exact, normalized, and franchise reference title resolution."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        # 1. Exact match
        res_exact = await recommender._resolve_reference_movie("Interstellar")
        assert res_exact is not None
        assert res_exact['id'] == 157336
        assert res_exact['idx'] is not None

        # 2. Normalized title match (handles articles and punctuation)
        res_norm = await recommender._resolve_reference_movie("The Dark Knight")
        assert res_norm is not None
        assert "Dark Knight" in res_norm['title']

        # 3. Franchise / Canonical prefix match (e.g. Harry Potter -> foundational/highest popularity entry)
        res_hp = await recommender._resolve_reference_movie("Harry Potter")
        assert res_hp is not None
        assert "Harry Potter" in res_hp['title']
        assert res_hp['id'] in [671, 672, 673, 674, 675, 12444, 12445]

    @pytest.mark.asyncio
    async def test_reference_movie_excluded_from_recommendations(self):
        """Verify that the reference movie itself is excluded from final recommendations."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        res_hp = await recommender._resolve_reference_movie("Harry Potter")
        anchor_id = res_hp['id']

        recs = await recommender.get_recommendations(
            user_id="test",
            natural_language="movies similar to Harry Potter",
            top_k=5
        )
        rec_ids = [r['movie']['id'] for r in recs['recommendations']]
        assert anchor_id not in rec_ids

    @pytest.mark.asyncio
    async def test_similarity_derived_from_reference_embedding(self):
        """Verify candidates and ranking are computed from reference movie semantic embedding."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        res = await recommender._resolve_reference_movie("Interstellar")
        assert res is not None
        ref_idx = res['idx']
        ref_emb = recommender.feature_matrix.semantic_embeddings[ref_idx]

        # Candidates retrieved by stage 1
        features = recommender.nl_extractor._fallback_extraction("something similar to Interstellar", {})
        candidates = await recommender._get_candidate_movies_stage1(
            features, [], "US", "en", None, resolved_ref=res
        )
        assert len(candidates) > 0

        # Candidate IDs must have high cosine similarity with reference embedding
        cand_id = candidates[0]['id']
        cand_idx = recommender.feature_matrix.tmdb_id_to_idx[cand_id]
        cand_emb = recommender.feature_matrix.semantic_embeddings[cand_idx]
        sim = float(np.dot(ref_emb, cand_emb))
        assert sim > 0.5

    @pytest.mark.asyncio
    async def test_reference_plus_temporal_constraint(self):
        """Test combined reference movie + temporal constraint (e.g. 'movies similar to Harry Potter after 2010')."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        recs = await recommender.get_recommendations(
            user_id="test",
            natural_language="movies similar to Harry Potter after 2010",
            top_k=5
        )
        for r in recs['recommendations']:
            rd = r['movie'].get('release_date', '')
            if rd and len(rd) >= 4:
                year = int(rd[:4])
                assert year >= 2011  # strictly after 2010

    @pytest.mark.asyncio
    async def test_reference_plus_genre_constraint(self):
        """Test combined reference movie + genre constraint."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        recs = await recommender.get_recommendations(
            user_id="test",
            natural_language="comedy movie like The Dark Knight",
            top_k=5
        )
        assert len(recs['recommendations']) > 0
        for r in recs['recommendations']:
            genres_lower = [g.lower() for g in r['movie'].get('genres', [])]
            assert "comedy" in genres_lower

    @pytest.mark.asyncio
    async def test_unresolved_title_safe_fallback(self):
        """Test that an unknown/unresolved title safely falls back without crashing."""
        recommender = MovieRecommender()
        await recommender.initialize()
        recommender.reasoner = None
        recommender.nl_extractor._llm_available = False

        recs = await recommender.get_recommendations(
            user_id="test",
            natural_language="movies similar to NonExistentFictionalMovieTitleXYZ999",
            top_k=5
        )
        assert isinstance(recs, dict)
        assert "recommendations" in recs
