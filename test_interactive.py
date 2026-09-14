import asyncio
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.llm.nl_feature_extractor import NLFeatureExtractor


async def main():
    extractor = NLFeatureExtractor()
    print("\n" + "=" * 65)
    print("🎬 PRITY'S FEATURE EXTRACTION & TMDB QUERY TESTER 🎬")
    print("Type any movie request below, or type 'exit' to quit.")
    print("=" * 65 + "\n")

    while True:
        try:
            prompt = input("👉 Enter movie prompt: ").strip()
            if not prompt or prompt.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            features = await extractor.extract(prompt)
            tmdb_params = extractor.build_tmdb_discover_params(features)

            print("\n" + "-" * 55)
            print("📦 1. HARD FILTERS (Mapped to TMDB API):")
            print(f"   • Primary Genres:     {features.primary_genres}")
            print(f"   • Loose Genres (Exp): {features.loose_genres}")
            print(f"   • Excluded Genres:    {features.excluded_genres}")
            print(f"   • Release Year Min:   {features.release_year_min}")
            print(f"   • Release Year Max:   {features.release_year_max}")
            print(f"   • Max Runtime:        {features.max_runtime_minutes} mins")
            print(f"   • Language:           {features.original_language}")
            print(f"   • Keywords:           {features.tmdb_keywords[:5]}")

            print("\n✨ 2. SOFT FEATURES (For Vector DB & Final LLM):")
            print(f"   • Mood:               {features.mood}")
            print(f"   • Theme / Plot:       {features.theme_or_plot}")
            print(f"   • Reference Movies:   {features.reference_movies}")
            print(f"   • Semantic Query:     {features.semantic_search_query}")

            print("\n🌐 3. TMDB /discover/movie QUERY PARAMETERS:")
            print(json.dumps(tmdb_params, indent=4))
            print("-" * 55 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
