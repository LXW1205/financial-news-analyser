"""
Financial News Sentiment Analyser
==================================
Fetches live financial headlines from NewsAPI and analyses their sentiment
using Google's Gemini AI. Returns structured JSON with sentiment, confidence,
and reasoning for each headline.

Usage:
    python main.py
"""

import json
import os
import sys
import time

import google.genai as genai
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

NEWS_API_URL = "https://newsapi.org/v2/everything"
HEADLINE_COUNT = 10
SEARCH_QUERY = "gold OR forex OR trading OR EUR/USD OR stock market OR crypto"

GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Step 1 — Fetch live financial headlines from NewsAPI
# ---------------------------------------------------------------------------

def fetch_headlines(count: int = HEADLINE_COUNT) -> list[dict]:
    """Fetch the latest financial news headlines from NewsAPI."""
    if not NEWS_API_KEY or NEWS_API_KEY == "your_key_here":
        print("ERROR: NEWS_API_KEY not set in .env file.", file=sys.stderr)
        sys.exit(1)

    params = {
        "q": SEARCH_QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": count,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to NewsAPI. Check your internet.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("ERROR: NewsAPI request timed out.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            print("ERROR: Invalid NewsAPI key.", file=sys.stderr)
        elif response.status_code == 429:
            print("ERROR: NewsAPI rate limit exceeded. Try again later.", file=sys.stderr)
        else:
            print(f"ERROR: NewsAPI returned HTTP {response.status_code}: {e}", file=sys.stderr)
        sys.exit(1)

    data = response.json()

    if data.get("status") != "ok":
        print(f"ERROR: NewsAPI error — {data.get('message', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    articles = data.get("articles", [])
    if not articles:
        print("ERROR: No articles returned from NewsAPI.", file=sys.stderr)
        sys.exit(1)

    # Return structured headline data
    return [
        {
            "title": article["title"],
            "source": article.get("source", {}).get("name", "Unknown"),
            "published_at": article.get("publishedAt", ""),
            "url": article.get("url", ""),
        }
        for article in articles[:count]
    ]


# ---------------------------------------------------------------------------
# Step 2 — Analyse sentiment with Gemini AI (single batch call)
# ---------------------------------------------------------------------------

def analyse_sentiment(headlines: list[dict]) -> list[dict]:
    """Send all headlines to Gemini in one call and return sentiment analysis."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_key_here":
        print("ERROR: GOOGLE_API_KEY not set in .env file.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Build the prompt with all headlines
    headlines_text = "\n".join(
        f"{i+1}. {h['title']}" for i, h in enumerate(headlines)
    )

    prompt = f"""You are a financial news sentiment analyst. Analyse the following {len(headlines)} financial news headlines.

For each headline, return a JSON object with:
- "headline": the exact headline text
- "sentiment": one of "positive", "negative", or "neutral"
- "confidence": a number between 0.0 and 1.0
- "reason": a brief one-sentence explanation

Return ONLY a valid JSON array. No markdown, no code fences, no extra text.

Headlines:
{headlines_text}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
    except Exception as e:
        error_str = str(e)
        # Handle 429 rate limit with retry
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            # Extract retry delay from error message if available
            retry_delay = 60  # default 60 seconds
            for part in error_str.split():
                if part.replace(".", "").isdigit() and float(part) > 10:
                    retry_delay = min(float(part), 120)
                    break
            print(f"  -> Rate limited. Retrying in {retry_delay:.0f}s...", file=sys.stderr)
            time.sleep(retry_delay)
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    },
                )
            except Exception as e2:
                print(f"ERROR: Gemini API call failed after retry — {e2}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"ERROR: Gemini API call failed — {e}", file=sys.stderr)
            sys.exit(1)

    # Parse the JSON response
    try:
        results = json.loads(response.text)
    except json.JSONDecodeError:
        print("ERROR: Gemini returned invalid JSON.", file=sys.stderr)
        print(f"Raw response: {response.text[:500]}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(results, list):
        print("ERROR: Expected a JSON array from Gemini.", file=sys.stderr)
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Step 3 — Parse and display results
# ---------------------------------------------------------------------------

def display_results(results: list[dict], headlines: list[dict]) -> None:
    """Print the sentiment analysis results in a clean, readable format."""
    print("\n" + "=" * 70)
    print("  FINANCIAL NEWS SENTIMENT ANALYSIS")
    print("=" * 70)

    # Summary counters
    positive = negative = neutral = 0

    for i, result in enumerate(results):
        sentiment = result.get("sentiment", "unknown").lower()
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "No reason provided")
        headline = result.get("headline", headlines[i]["title"] if i < len(headlines) else "N/A")

        # Update counters
        if sentiment == "positive":
            positive += 1
            emoji = "[POSITIVE]"
        elif sentiment == "negative":
            negative += 1
            emoji = "[NEGATIVE]"
        else:
            neutral += 1
            emoji = "[NEUTRAL]"

        print(f"\n{i+1}. {headline}")
        print(f"   Sentiment : {emoji} ({confidence:.0%} confidence)")
        print(f"   Reason    : {reason}")

    # Summary
    total = len(results)
    print("\n" + "-" * 70)
    print(f"  SUMMARY: {positive} positive, {negative} negative, {neutral} neutral out of {total} headlines")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Step 4 — Main pipeline with error handling
# ---------------------------------------------------------------------------

def main():
    """Run the full sentiment analysis pipeline."""
    print("Fetching financial headlines from NewsAPI...")
    headlines = fetch_headlines()
    print(f"  -> Fetched {len(headlines)} headlines.\n")

    print("Analysing sentiment with Gemini AI...")
    results = analyse_sentiment(headlines)
    print(f"  -> Analysed {len(results)} headlines.\n")

    display_results(results, headlines)

    # Also output raw JSON for programmatic use
    print("\n[JSON Output]")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
