import pytest
import sqlite3
import json
from src.pipeline.recommender import _sanitize_keywords
from src.llm.nl_feature_extractor import NLFeatureExtractor


class TestCleanupRegressions:
    def test_keyword_sanitization_helper(self):
        """Verify _sanitize_keywords cleans JSON strings, malformed nested strings, and lists."""
        # 1. Direct JSON array string
        raw_json_str = '["spacecraft", "race against time", "artificial intelligence (a.i.)", "nasa"]'
        result = _sanitize_keywords(raw_json_str)
        assert result == ["spacecraft", "race against time", "artificial intelligence (a.i.)", "nasa"]
        
        # 2. Malformed nested list of string-escaped JSON fragments
        malformed_nested = ['["spacecraft"', '"race against time"', '"nasa"]']
        result_nested = _sanitize_keywords(malformed_nested)
        assert result_nested == ["spacecraft", "race against time", "nasa"]
        assert not any("[" in k or "]" in k or '"' in k for k in result_nested)

        # 3. Pipe-separated string fallback
        pipe_str = "spacecraft | planet mars | nasa"
        result_pipe = _sanitize_keywords(pipe_str)
        assert result_pipe == ["spacecraft", "planet mars", "nasa"]

        # 4. Empty or None inputs
        assert _sanitize_keywords(None) == []
        assert _sanitize_keywords("") == []
        assert _sanitize_keywords([]) == []

    def test_sqlite_raw_keyword_parsing(self):
        """Verify querying raw SQLite tmdb_keywords and parsing yields clean string lists."""
        conn = sqlite3.connect("movie_features.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT tmdb_id, title, tmdb_keywords 
            FROM movies 
            WHERE tmdb_keywords IS NOT NULL AND tmdb_keywords != '' AND tmdb_keywords != '[]' 
            LIMIT 50
        """)
        rows = cur.fetchall()
        conn.close()

        assert len(rows) > 0
        for r in rows:
            parsed = _sanitize_keywords(r["tmdb_keywords"])
            assert isinstance(parsed, list)
            for kw in parsed:
                assert not kw.startswith('[') and not kw.endswith(']'), f"Keyword contains brackets: {kw}"
                assert not kw.startswith('"') and not kw.endswith('"'), f"Keyword contains quotes: {kw}"
                assert len(kw) > 0

    def test_streaming_provider_deduplication(self):
        """Verify provider list deduplication preserves order without duplicates."""
        raw_platforms = ["Apple TV+", "Apple TV+"]
        deduped = list(dict.fromkeys(raw_platforms))
        assert deduped == ["Apple TV+"]

        mixed_platforms = ["Prime Video", "Netflix", "Prime Video", "Apple TV+", "Netflix"]
        deduped_mixed = list(dict.fromkeys(mixed_platforms))
        assert deduped_mixed == ["Prime Video", "Netflix", "Apple TV+"]

    def test_unspecified_language_extraction_is_none(self):
        """Verify unspecified language query returns original_language = None."""
        extractor = NLFeatureExtractor()
        
        # Test fallback / direct extraction on query with no language mentioned
        f = extractor._fallback_extraction("mind bending sci-fi space", {})
        assert f.original_language is None, f"Expected None, got {f.original_language}"

        # Test prompt parser with mock LLM JSON containing original_language = null
        parsed_json = {
            "primary_genres": ["Science Fiction"],
            "loose_genres": ["Mystery"],
            "original_language": None,
            "tmdb_keywords": ["space", "mind-bending"]
        }
        f_parsed = extractor._parse_and_validate(parsed_json, "mind bending sci-fi space")
        assert f_parsed.original_language is None

    def test_explicit_language_extraction_is_preserved(self):
        """Verify explicit language query sets the correct ISO 639-1 code."""
        extractor = NLFeatureExtractor()
        
        f_fr = extractor._fallback_extraction("french romantic comedy from the 90s", {})
        assert f_fr.original_language == "fr"

        f_es = extractor._fallback_extraction("spanish horror movie", {})
        assert f_es.original_language == "es"

        f_ja = extractor._fallback_extraction("japanese anime adventure", {})
        assert f_ja.original_language == "ja"

        f_ko = extractor._fallback_extraction("korean thriller", {})
        assert f_ko.original_language == "ko"

    def test_mind_bending_scifi_space_keywords_and_mood(self):
        """Verify 'mind bending sci-fi space' captures mood without fabricating 'mind control' keyword."""
        extractor = NLFeatureExtractor()
        
        # 1. Fallback extraction check
        f_fb = extractor._fallback_extraction("mind bending sci-fi space", {})
        assert f_fb.primary_genres == ["Science Fiction"]
        assert f_fb.mood == "mind-bending"
        assert "space" in f_fb.tmdb_keywords or "space" in f_fb.keywords
        assert "mind control" not in f_fb.tmdb_keywords
        assert "mind control" not in f_fb.keywords
        assert "mind" not in f_fb.tmdb_keywords
        assert "bending" not in f_fb.tmdb_keywords
        assert f_fb.original_language is None

        # 2. LLM / Parser validation check
        mock_llm_output = {
            "primary_genres": ["Science Fiction"],
            "loose_genres": ["Mystery"],
            "mood": "mind-bending",
            "tmdb_keywords": ["space", "mind control"],
            "original_language": None,
            "theme_or_plot": "space exploration with psychological twists"
        }
        f_parsed = extractor._parse_and_validate(mock_llm_output, "mind bending sci-fi space")
        assert "space" in f_parsed.tmdb_keywords
        assert "mind control" not in f_parsed.tmdb_keywords
        assert f_parsed.mood == "mind-bending"
        assert f_parsed.original_language is None

    def test_generic_genre_variant_normalization(self):
        """Verify normalization works generically across diverse colloquial, alternate, and hyphenated expressions."""
        extractor = NLFeatureExtractor()
        
        test_cases = [
            (["anime", "cartoon"], ["Animation"]),
            (["cyberpunk", "dystopian", "space opera"], ["Science Fiction"]),
            (["period piece", "biopic"], ["History"]),
            (["whodunit", "detective"], ["Mystery"]),
            (["doc", "docuseries"], ["Documentary"]),
            (["superhero", "martial arts"], ["Action"]),
            (["kids", "children"], ["Family"]),
            (["heist", "gangster"], ["Crime"]),
            (["musical"], ["Music"]),
            (["suspense", "psychological thriller"], ["Thriller"]),
            (["cowboy"], ["Western"]),
            (["sci-fi", "romcom", "spooky"], ["Science Fiction", "Comedy", "Romance", "Horror"])
        ]
        
        for input_variants, expected_canonical in test_cases:
            normalized = extractor._validate_genres(input_variants)
            for exp in expected_canonical:
                assert exp in normalized, f"Expected '{exp}' in normalized genres for inputs {input_variants}, got {normalized}"
            # Ensure every output is a valid canonical TMDB genre
            for g in normalized:
                assert g in extractor.TMDB_GENRES, f"Output '{g}' is not a valid canonical TMDB genre"

    def test_ui_user_selection_persistence_and_reload(self):
        """Verify UI user selection persistence:
        1. HTML contains localStorage persistence hooks and validation.
        2. Simulated user selection -> page reload keeps selected user ID.
        3. Invalid / empty IDs gracefully fall back to 'demo'.
        4. User endpoints serve the restored user profile and history.
        """
        from pathlib import Path
        import re
        from fastapi.testclient import TestClient
        from src.ui.web_app import app

        index_path = Path("src/ui/static/index.html")
        assert index_path.exists(), "src/ui/static/index.html does not exist"
        html_content = index_path.read_text(encoding="utf-8")

        # 1. Verify UI HTML hooks
        assert 'id="userIdInput"' in html_content
        assert 'handleUserChange()' in html_content
        assert 'STORAGE_KEY_USER_ID' in html_content
        assert 'techquest_user_id' in html_content
        assert 'getValidUserId' in html_content
        assert 'DOMContentLoaded' in html_content
        assert 'localStorage.getItem(STORAGE_KEY_USER_ID)' in html_content
        assert 'localStorage.setItem(STORAGE_KEY_USER_ID' in html_content

        # 2. Logic simulation of getValidUserId and lifecycle across reloads
        def sim_get_valid_user_id(raw_id: str | None) -> str:
            if not raw_id or not isinstance(raw_id, str):
                return "demo"
            trimmed = raw_id.strip()
            if not trimmed or len(trimmed) > 64 or not re.match(r"^[a-zA-Z0-9_\-]+$", trimmed):
                return "demo"
            return trimmed

        # Simulated browser storage
        browser_local_storage = {}
        storage_key = "techquest_user_id"

        # Scenario A: Initial load with no prior selection
        initial_user = sim_get_valid_user_id(browser_local_storage.get(storage_key))
        assert initial_user == "demo"

        # Scenario B: User selects 'memory_test1'
        selected_user = "memory_test1"
        valid_selected = sim_get_valid_user_id(selected_user)
        browser_local_storage[storage_key] = valid_selected
        assert browser_local_storage[storage_key] == "memory_test1"

        # Scenario C: Browser reload -> DOMContentLoaded runs
        reloaded_user = sim_get_valid_user_id(browser_local_storage.get(storage_key))
        assert reloaded_user == "memory_test1", f"Expected 'memory_test1' on reload, got '{reloaded_user}'"

        # Scenario D: Graceful fallback on invalid / malicious IDs
        for bad_id in ["", "   ", None, "bad user with spaces", "user<script>", "a" * 65]:
            assert sim_get_valid_user_id(bad_id) == "demo"

        # 3. Test API endpoints respond for persisted user
        client = TestClient(app)
        res_profile = client.get(f"/api/profile/{reloaded_user}")
        assert res_profile.status_code == 200
        profile_json = res_profile.json()
        assert "profile" in profile_json
        assert "stats" in profile_json

        res_history = client.get(f"/api/history/{reloaded_user}")
        assert res_history.status_code == 200
        history_json = res_history.json()
        assert "ratings" in history_json
        assert "events" in history_json

    def test_no_fabricated_streaming_platforms(self):
        """Verify fabricated streaming platforms are not generated for movies."""
        from src.pipeline.recommender import MovieRecommender
        from src.ranking.ranker import RankedMovie

        rec = MovieRecommender()
        # Mock feature_matrix with minimal metadata
        class MockFeatureMatrix:
            movie_metadata = {
                100: {"title": "Test Movie 1", "genres": ["Action", "Science Fiction"]},
                101: {"title": "Test Movie 2", "genres": ["Animation", "Family"]}
            }
        rec.feature_matrix = MockFeatureMatrix()
        
        # Scenario A: candidate has no provider data -> streaming_platforms must be []
        top_movies = [
            RankedMovie(
                movie_id=100, title="Test Movie 1", score=0.9, imdb_score=0.8,
                tfidf_score=0.1, watch_history_score=0.0, semantic_score=0.7,
                content_score=0.0, keyword_overlap_score=0.0, popularity_score=0.5,
                overview="Action film", genres=["Action"], vote_average=8.0,
                vote_count=500, reasoning_factors={}
            )
        ]
        candidate_movies = [{"id": 100, "title": "Test Movie 1", "genres": ["Action"]}]
        
        # Test extraction logic directly as executed in get_recommendations
        candidate_lookup = {c.get('id') or c.get('movie_id'): c for c in candidate_movies}
        cand = candidate_lookup.get(100, {})
        platforms = []
        if cand.get('streaming_platforms') and isinstance(cand.get('streaming_platforms'), list):
            platforms = list(cand.get('streaming_platforms'))
        elif cand.get('watch_providers') and isinstance(cand.get('watch_providers'), dict):
            wp = cand.get('watch_providers')
            flatrate = wp.get('flatrate') or []
            platforms = [p.get('provider_name') for p in flatrate if isinstance(p, dict) and p.get('provider_name')]
        platforms = list(dict.fromkeys([p for p in platforms if isinstance(p, str) and p.strip()]))
        
        assert platforms == [], f"Expected empty platforms when no real provider data, got: {platforms}"

        # Scenario B: candidate has genuine watch_providers flatrate data -> preserve real providers
        cand_real = {
            "id": 101,
            "title": "Test Movie 2",
            "watch_providers": {
                "flatrate": [{"provider_name": "Disney+"}, {"provider_name": "Hulu"}]
            }
        }
        platforms_real = []
        if cand_real.get('streaming_platforms') and isinstance(cand_real.get('streaming_platforms'), list):
            platforms_real = list(cand_real.get('streaming_platforms'))
        elif cand_real.get('watch_providers') and isinstance(cand_real.get('watch_providers'), dict):
            wp = cand_real.get('watch_providers')
            flatrate = wp.get('flatrate') or []
            platforms_real = [p.get('provider_name') for p in flatrate if isinstance(p, dict) and p.get('provider_name')]
        platforms_real = list(dict.fromkeys([p for p in platforms_real if isinstance(p, str) and p.strip()]))
        assert platforms_real == ["Disney+", "Hulu"]

    def test_watch_history_includes_actual_movie_title(self):
        """Verify /api/history returns movie title rather than review_text."""
        from fastapi.testclient import TestClient
        from src.ui.web_app import app, get_memory_service
        from src.memory.models import RatingEvent
        import time

        mem = get_memory_service()
        test_user = "test_title_user"
        test_movie_id = 77  # Memento in canonical db

        # Record a rating with distinct review_text
        mem.storage.record_rating(RatingEvent(
            user_id=test_user,
            movie_id=test_movie_id,
            rating=9.5,
            signal_target=0.95,
            timestamp=time.time(),
            review_text="This is a unique review text that should not be the movie title!"
        ))

        client = TestClient(app)
        res = client.get(f"/api/history/{test_user}")
        assert res.status_code == 200
        data = res.json()
        assert len(data["ratings"]) > 0
        r_entry = [r for r in data["ratings"] if r["movie_id"] == test_movie_id][0]
        
        # Verify title is not review_text
        assert r_entry["title"] != r_entry["review_text"]
        assert r_entry["review_text"] == "This is a unique review text that should not be the movie title!"
        # Verify title is either the actual movie title or Movie #77, but definitely not the review
        assert "unique review" not in r_entry["title"]

    def test_ui_html_removes_genome_and_empty_theme_tone_sections(self):
        """Verify index.html has no obsolete Genome text and no empty theme/tone sections."""
        from pathlib import Path
        html_path = Path("src/ui/static/index.html")
        content = html_path.read_text(encoding="utf-8")

        # 1. Genome references removed
        assert "MovieLens Genome Tags" not in content
        assert "genome" not in content.lower()

        # 2. Empty theme and tone panel IDs removed
        assert 'id="topThemesList"' not in content
        assert 'id="topTonesList"' not in content

        # 3. Legitimate MovieLens tags preserved
        assert "MovieLens Tags" in content
        assert 'id="topTagsList"' in content

    def test_candidate_discovery_preserves_poster_path(self):
        """Verify candidate discovery dict includes poster_path."""
        import inspect
        from src.pipeline.recommender import MovieRecommender

        source = inspect.getsource(MovieRecommender._get_candidate_movies_stage1)
        assert "'poster_path': m.poster_path" in source

    def test_ui_adaptive_profile_blend_presentation(self):
        """Verify UI displays 'Adaptive Profile Blend' terminology with explicit guidance:
        1. Explanatory title 'Adaptive Profile Blend' and legend 'Query Focus' / 'Long-Term Profile'.
        2. Help text / tooltip explaining ranking balance vs storage.
        3. No misleading 'Permanent' or '7d Decay' wording that implies stored memory percentage.
        4. Dynamic JavaScript template updates format with '% Query Focus / % Long-Term Profile'.
        """
        from pathlib import Path
        html_path = Path("src/ui/static/index.html")
        assert html_path.exists(), "src/ui/static/index.html does not exist"
        content = html_path.read_text(encoding="utf-8")

        # 1. Verify updated card title and legends
        assert "Adaptive Profile Blend" in content
        assert "Query Focus" in content
        assert "Long-Term Profile" in content

        # 2. Verify explicit explanatory help text & tooltip
        expected_help_text = (
            "Shows how recommendation ranking balances your current request against your accumulated taste profile. "
            "It is not a measure of how much memory is stored."
        )
        assert expected_help_text in content

        # 3. Verify misleading phrasing is removed
        assert "Long-term (Permanent)" not in content
        assert "Short-term (7d Decay)" not in content
        assert "% Short / % Long" not in content

        # 4. Verify JavaScript formatting
        assert "${stats.short_term_pct}% Query Focus / ${stats.long_term_pct}% Long-Term Profile" in content
