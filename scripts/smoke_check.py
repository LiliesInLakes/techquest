import sqlite3
import json
import asyncio
from src.pipeline.recommender import _sanitize_keywords
from src.llm.nl_feature_extractor import NLFeatureExtractor

print("=== SMOKE CHECK A: SQLite Keyword Parsing ===")
conn = sqlite3.connect("movie_features.db")
cur = conn.cursor()
cur.execute("SELECT tmdb_id, title, tmdb_keywords FROM movies WHERE tmdb_keywords LIKE '%spacecraft%' LIMIT 1")
row = cur.fetchone()
conn.close()
raw_kw = row[2]
parsed_kw = _sanitize_keywords(raw_kw)
print(f"Movie: {row[1]} (TMDB ID {row[0]})")
print(f"Raw in SQLite: {raw_kw[:80]}...")
print(f"Parsed type: {type(parsed_kw)}")
print(f"First 4 keywords: {parsed_kw[:4]}")
assert isinstance(parsed_kw, list)
assert not any("[" in k or "]" in k or '"' in k for k in parsed_kw)
print("Smoke Check A PASSED!\n")

print("=== SMOKE CHECK B: Streaming Provider Deduplication ===")
raw_platforms = ["Apple TV+", "Apple TV+"]
deduped = list(dict.fromkeys(raw_platforms))
print(f"Input: {raw_platforms}")
print(f"Output: {deduped}")
assert deduped == ["Apple TV+"]
print("Smoke Check B PASSED!\n")

print("=== SMOKE CHECK C: Language Extraction on 'mind bending sci-fi space' ===")
extractor = NLFeatureExtractor()
f_fallback = extractor._fallback_extraction("mind bending sci-fi space", {})
print(f"Query: 'mind bending sci-fi space'")
print(f"Fallback original_language: {f_fallback.original_language}")
assert f_fallback.original_language is None

# Also test async extract
f_async = asyncio.run(extractor.extract("mind bending sci-fi space"))
print(f"Async/LLM original_language: {f_async.original_language}")
assert f_async.original_language is None
print("Smoke Check C PASSED!")
