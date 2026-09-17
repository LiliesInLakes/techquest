import unittest
import asyncio
from src.llm.nl_feature_extractor import NLFeatureExtractor, ExtractedFeatures


class TestExtractedFeatures(unittest.TestCase):
    def test_schema_backward_compatibility(self):
        """Verify that legacy fields and new fields are automatically synchronized."""
        # Test legacy instantiation
        features = ExtractedFeatures(
            genres=["Action", "Science Fiction"],
            exclude_genres=["Horror"],
            keywords=["space", "heist"],
            query_text="space heist action"
        )
        self.assertEqual(features.primary_genres, ["Action", "Science Fiction"])
        self.assertEqual(features.excluded_genres, ["Horror"])
        self.assertEqual(features.tmdb_keywords, ["space", "heist"])
        self.assertEqual(features.semantic_search_query, "space heist action")

        # Test new schema instantiation
        new_features = ExtractedFeatures(
            primary_genres=["Comedy", "Romance"],
            loose_genres=["Drama"],
            excluded_genres=["Thriller"],
            release_year_min=1990,
            release_year_max=1999,
            max_runtime_minutes=120,
            original_language="en",
            tmdb_keywords=["dating", "wedding"],
            mood="feel-good",
            theme_or_plot="childhood friends reconnecting",
            reference_movies=["When Harry Met Sally"],
            semantic_search_query="90s feel-good romantic comedy about childhood friends"
        )
        self.assertEqual(new_features.genres, ["Comedy", "Romance"])
        self.assertEqual(new_features.exclude_genres, ["Thriller"])
        self.assertEqual(new_features.keywords, ["dating", "wedding"])
        self.assertEqual(new_features.query_text, "90s feel-good romantic comedy about childhood friends")
        self.assertEqual(new_features.release_year_min, 1990)
        self.assertEqual(new_features.release_year_max, 1999)
        self.assertEqual(new_features.max_runtime_minutes, 120)

    def test_to_dict_keys(self):
        """Ensure to_dict returns all expected hard and soft feature keys."""
        features = ExtractedFeatures(
            primary_genres=["Drama"],
            loose_genres=["Romance"],
            excluded_genres=["Horror"],
            mood="emotional",
            release_year_min=2000,
            release_year_max=2010
        )
        d = features.to_dict()
        required_keys = [
            "genres", "primary_genres", "loose_genres",
            "exclude_genres", "excluded_genres",
            "actors", "directors", "decades",
            "release_year_min", "release_year_max",
            "min_runtime_minutes", "max_runtime_minutes",
            "original_language", "keywords", "tmdb_keywords",
            "exclude_keywords", "min_rating", "mood",
            "theme_or_plot", "reference_movies",
            "semantic_search_query", "query_text", "confidence", "original_text"
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key: {key}")


class TestNLFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = NLFeatureExtractor()

    def test_tmdb_genre_constants(self):
        """Verify all 19 official TMDB genres are registered with correct IDs."""
        genres = self.extractor.get_tmdb_genre_ids()
        self.assertEqual(len(genres), 19)
        self.assertEqual(genres["Action"], 28)
        self.assertEqual(genres["Adventure"], 12)
        self.assertEqual(genres["Animation"], 16)
        self.assertEqual(genres["Comedy"], 35)
        self.assertEqual(genres["Crime"], 80)
        self.assertEqual(genres["Documentary"], 99)
        self.assertEqual(genres["Drama"], 18)
        self.assertEqual(genres["Family"], 10751)
        self.assertEqual(genres["Fantasy"], 14)
        self.assertEqual(genres["History"], 36)
        self.assertEqual(genres["Horror"], 27)
        self.assertEqual(genres["Music"], 10402)
        self.assertEqual(genres["Mystery"], 9648)
        self.assertEqual(genres["Romance"], 10749)
        self.assertEqual(genres["Science Fiction"], 878)
        self.assertEqual(genres["TV Movie"], 10770)
        self.assertEqual(genres["Thriller"], 53)
        self.assertEqual(genres["War"], 10752)
        self.assertEqual(genres["Western"], 37)

    def test_genre_normalization(self):
        """Test colloquial genre mapping to official TMDB genres."""
        valid = self.extractor._validate_genres(["sci-fi", "romcom", "spooky"])
        self.assertIn("Science Fiction", valid)
        self.assertIn("Comedy", valid)
        self.assertIn("Horror", valid)

    def test_loose_genres_expansion(self):
        """Test that query expansion derives related loose genres."""
        loose = self.extractor._derive_loose_genres(["Science Fiction"])
        self.assertTrue(len(loose) > 0)
        self.assertTrue(any(g in ["Action", "Mystery", "Adventure"] for g in loose))

        loose_thriller = self.extractor._derive_loose_genres(["Thriller"])
        self.assertTrue(any(g in ["Mystery", "Crime", "Action"] for g in loose_thriller))

    def test_offline_fallback_extraction_complex_prompt(self):
        """Test extraction of a complex query with hard and soft features in offline mode."""
        prompt = "I want a fast-paced 90s sci-fi heist movie like Inception, but no horror and under 2 hours in French"
        features = self.extractor._fallback_extraction(prompt, {})

        # Hard filters
        self.assertIn("Science Fiction", features.primary_genres)
        self.assertIn("Horror", features.excluded_genres)
        self.assertEqual(features.release_year_min, 1990)
        self.assertEqual(features.release_year_max, 1999)
        self.assertEqual(features.max_runtime_minutes, 120)
        self.assertEqual(features.original_language, "fr")

        # Soft features
        self.assertIn("Inception", features.reference_movies)
        self.assertTrue(len(features.semantic_search_query) > 0)
        self.assertTrue(len(features.loose_genres) > 0)

    def test_offline_fallback_runtime_variations(self):
        """Test various ways users express runtime limits."""
        f1 = self.extractor._fallback_extraction("funny movie under 90 mins", {})
        self.assertEqual(f1.max_runtime_minutes, 90)

        f2 = self.extractor._fallback_extraction("quick watch drama", {})
        self.assertEqual(f2.max_runtime_minutes, 95)

        f3 = self.extractor._fallback_extraction("action movie within 2.5 hours", {})
        self.assertEqual(f3.max_runtime_minutes, 150)

    def test_offline_fallback_mood_and_ratings(self):
        """Test mood extraction and rating thresholds."""
        f1 = self.extractor._fallback_extraction("dark gritty crime thriller, highly rated", {})
        self.assertIn("dark", [f1.mood, "dark"])
        self.assertGreaterEqual(f1.min_rating, 6.0)

        f2 = self.extractor._fallback_extraction("feel-good animated family movie", {})
        self.assertEqual(f2.mood, "feel-good")

    def test_build_tmdb_discover_params(self):
        """Test that ExtractedFeatures are translated to correct TMDB /discover/movie query params."""
        features = ExtractedFeatures(
            primary_genres=["Action", "Science Fiction"],
            explicit_genres=["Action", "Science Fiction"],
            excluded_genres=["Horror"],
            release_year_min=1990,
            release_year_max=1999,
            max_runtime_minutes=120,
            original_language="en",
            min_rating=7.0
        )
        params = self.extractor.build_tmdb_discover_params(
            features, subscriptions=["8", "337"], region="US"
        )

        self.assertEqual(params["with_genres"], "28,878")  # Action=28, Sci-Fi=878
        self.assertEqual(params["without_genres"], "27")     # Horror=27
        self.assertEqual(params["primary_release_date_gte"], "1990-01-01")
        self.assertEqual(params["primary_release_date_lte"], "1999-12-31")
        self.assertEqual(params["with_runtime_lte"], 120)
        self.assertEqual(params["min_rating"], 7.0)
        self.assertEqual(params["sort_by"], "popularity.desc")
        self.assertEqual(params["with_watch_providers"], ["8", "337"])
        self.assertEqual(params["watch_region"], "US")
        self.assertEqual(params["language"], "en")

    def test_build_fallback_tmdb_params(self):
        """Test query loosening: switches from AND (comma) to OR (pipe) and includes loose genres."""
        features = ExtractedFeatures(
            primary_genres=["Science Fiction"],
            explicit_genres=["Science Fiction"],
            loose_genres=["Action", "Mystery"],
            excluded_genres=["Horror"]
        )
        params = self.extractor.build_fallback_tmdb_params(features, region="US")

        self.assertIn("|", params["with_genres"])
        # Should include Science Fiction (878), Action (28), and Mystery (9648)
        self.assertIn("878", params["with_genres"])
        self.assertIn("28", params["with_genres"])
        self.assertIn("9648", params["with_genres"])
        self.assertEqual(params["without_genres"], "27")
        self.assertNotIn("min_rating", params)

    def test_soft_vs_hard_constraint_distinction(self):
        """Verify soft queries omit hard filters and explicit queries include them."""
        # Soft query
        soft_f = self.extractor._fallback_extraction("something dark and psychological", {})
        soft_params = self.extractor.build_tmdb_discover_params(soft_f)
        self.assertNotIn("primary_release_date_gte", soft_params)
        self.assertNotIn("with_runtime_lte", soft_params)
        self.assertNotIn("with_cast", soft_params)
        self.assertNotIn("with_crew", soft_params)
        self.assertNotIn("with_keywords", soft_params)
        self.assertNotIn("language", soft_params)
        self.assertNotEqual(soft_f.include_adult, True)

        # Explicit query
        hard_f = self.extractor._fallback_extraction("1990s movie with Tom Hanks", {})
        self.assertEqual(hard_f.release_year_min, 1990)
        self.assertEqual(hard_f.release_year_max, 1999)
        self.assertIn("Tom Hanks", hard_f.actors)
        hard_params = self.extractor.build_tmdb_discover_params(hard_f)
        self.assertEqual(hard_params["primary_release_date_gte"], "1990-01-01")
        self.assertEqual(hard_params["primary_release_date_lte"], "1999-12-31")
        self.assertEqual(hard_params["with_cast"], ["Tom Hanks"])

    def test_adult_content_parsing(self):
        """Test that include_adult is True only on explicit adult requests."""
        f1 = self.extractor._fallback_extraction("dark violent mature thriller", {})
        self.assertNotEqual(f1.include_adult, True)

        f2 = self.extractor._fallback_extraction("nsfw adult movie", {})
        self.assertEqual(f2.include_adult, True)


class TestTMDBClientHelpers(unittest.IsolatedAsyncioTestCase):
    async def test_search_person_and_keywords(self):
        """Verify search_person and search_keyword resolvers."""
        from unittest.mock import AsyncMock, patch
        from src.tmdb.client import TMDBClient

        client = TMDBClient(api_key="fake_key_for_testing")
        client._request = AsyncMock()

        # Mock search_person
        client._request.return_value = {
            "results": [{"id": 500, "name": "Tom Cruise"}]
        }
        person_id = await client.search_person("Tom Cruise")
        self.assertEqual(person_id, 500)
        client._request.assert_awaited_with("/search/person", {"query": "Tom Cruise"})

        # Mock search_keyword
        client._request.return_value = {
            "results": [{"id": 9663, "name": "heist"}]
        }
        keyword_id = await client.search_keyword("heist")
        self.assertEqual(keyword_id, 9663)
        client._request.assert_awaited_with("/search/keyword", {"query": "heist"})

    async def test_discover_movies_with_advanced_filters(self):
        """Verify that discover_movies passes new parameters to the TMDB API request."""
        from unittest.mock import AsyncMock
        from src.tmdb.client import TMDBClient

        client = TMDBClient(api_key="fake_key_for_testing")
        client._request = AsyncMock(return_value={"results": []})
        client.search_person = AsyncMock(side_effect=lambda name: 500 if name == "Tom Cruise" else None)

        await client.discover_movies(
            with_genres="28,878",
            without_genres="27",
            with_cast=["Tom Cruise"],
            primary_release_date_gte="1990-01-01",
            primary_release_date_lte="1999-12-31",
            with_runtime_lte=120,
            language="en"
        )

        call_args = client._request.call_args
        endpoint = call_args[0][0]
        params = call_args[0][1]

        self.assertEqual(endpoint, "/discover/movie")
        self.assertEqual(params["with_genres"], "28,878")
        self.assertEqual(params["without_genres"], "27")
        self.assertEqual(params["with_cast"], "500")
        self.assertEqual(params["primary_release_date.gte"], "1990-01-01")
        self.assertEqual(params["primary_release_date.lte"], "1999-12-31")
        self.assertEqual(params["with_runtime.lte"], 120)
        self.assertEqual(params["with_original_language"], "en")


if __name__ == "__main__":
    unittest.main()

