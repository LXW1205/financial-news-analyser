# Financial News Sentiment Analyser

> **Note:** This project was built as a preparation exercise for a coding assessment at [Deriv](https://deriv.com). The entire codebase was developed by AI using [opencode](https://opencode.ai) with self-implemented subagents for faster and more accurate development — all completed within one hour to reflect the actual assessment time constraints.

Fetches live financial news headlines and analyses their sentiment using Google's Gemini AI — returning structured JSON with sentiment classification, confidence scores, and reasoning for each headline.

## What It Does

This tool automates a three-step pipeline:

1. **Fetches** the latest financial headlines from [NewsAPI](https://newsapi.org/) using configurable search queries (gold, forex, trading, crypto, etc.)
2. **Analyses** all headlines in a single batch call to [Google Gemini AI](https://ai.google.dev/), classifying each as positive, negative, or neutral with a confidence score and one-sentence explanation
3. **Outputs** results in both a human-readable CLI table and machine-readable JSON

### Example Output

```
======================================================================
  FINANCIAL NEWS SENTIMENT ANALYSIS
======================================================================

1. IBM and US Commerce Department unveil first purpose-built quantum foundry with $1B support
   Sentiment : [POSITIVE] (93% confidence)
   Reason    : Significant investment and government support for a new quantum foundry indicate strong potential for technological advancement and economic growth.

2. Why Indian tourists are rushing to Dubai's gold markets after 15% import duty hike
   Sentiment : [NEGATIVE] (80% confidence)
   Reason    : The headline implies a negative consequence of an import duty hike, driving consumers to seek alternatives, which could impact domestic markets negatively.

----------------------------------------------------------------------
  SUMMARY: 4 positive, 1 negative, 5 neutral out of 10 headlines
======================================================================
```

## Use Cases

- **Traders & analysts** — Quick sentiment overview of today's financial news to gauge market mood
- **Developers** — Reference project demonstrating API integration, AI prompting, and error handling
- **Portfolio piece** — Clean, well-tested project suitable for technical interviews (especially AI/ML roles)
- **Automation pipelines** — JSON output can be piped into dashboards, alerts, or trading bots
- **Learning** — Covers HTTP requests, JSON parsing, environment variables, mocking, and unit/integration testing

## Project Structure

```
financial-news-analyser/
├── .env                    # Your API keys (gitignored — never committed)
├── .env.example            # Template with placeholder values (safe to commit)
├── .gitignore              # Excludes secrets, caches, and build artifacts
├── main.py                 # Application source (~240 lines)
│   ├── fetch_headlines()   # Step 1: NewsAPI integration
│   ├── analyse_sentiment() # Step 2: Gemini AI batch analysis
│   ├── display_results()   # Step 3: Formatted CLI output
│   └── main()              # Step 4: Pipeline orchestration
├── requirements.txt        # Python dependencies (3 packages)
└── tests/
    ├── __init__.py
    ├── test_unit.py        # 31 unit tests (mocked APIs)
    └── test_integration.py # 9 integration tests (end-to-end)
```

## Requirements

### System

- **Python 3.10+** (tested on 3.12)
- Internet connection (for API calls)

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Google Gemini AI SDK |
| `python-dotenv` | Load API keys from `.env` file |
| `requests` | HTTP client for NewsAPI |

Install with:

```bash
pip install -r requirements.txt
```

### API Keys (Required)

You need **two free API keys** to run this project:

#### 1. NewsAPI — `NEWS_API_KEY`

- **Get it:** [newsapi.org/register](https://newsapi.org/register)
- **Cost:** Free tier (100 requests/day, no credit card required)
- **Instant activation:** API key available immediately after signup
- **Limitations:** Free tier works on `localhost` / development only

#### 2. Google Gemini — `GOOGLE_API_KEY`

- **Get it:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Cost:** Free tier (generous daily limits, no credit card required)
- **Model used:** `gemini-2.5-flash` (fast, cost-effective)

### Setting Up Your Keys

1. Copy the template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your real keys:
   ```
   GOOGLE_API_KEY=your_actual_gemini_key_here
   NEWS_API_KEY=your_actual_newsapi_key_here
   ```

3. **Never commit `.env`** — it's already in `.gitignore`, but double-check before pushing.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/LXW1205/financial-news-analyser.git
cd financial-news-analyser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API keys
cp .env.example .env
# Edit .env with your actual keys

# 4. Run it
python main.py
```

## Configuration

All settings are at the top of `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `HEADLINE_COUNT` | `10` | Number of headlines to fetch |
| `SEARCH_QUERY` | `"gold OR forex OR trading OR EUR/USD OR stock market OR crypto"` | NewsAPI search terms |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model to use |
| `NEWS_API_URL` | `"https://newsapi.org/v2/everything"` | NewsAPI endpoint |

Customise the search query to focus on specific markets, assets, or companies:

```python
SEARCH_QUERY = "AAPL OR Tesla OR Bitcoin OR Federal Reserve"
```

## Error Handling

The application handles these failure modes gracefully:

| Error | Behaviour |
|-------|-----------|
| Missing API key | Exits with clear error message |
| Invalid API key | Exits with specific error (401 vs generic) |
| Network timeout | Exits after 15-second timeout |
| Connection failure | Exits with network error message |
| Rate limit (429) | Exits with rate limit warning |
| Empty results | Exits with "no articles" message |
| Gemini API failure | Exits with error details |
| Gemini rate limit | Automatically retries after delay |
| Invalid JSON response | Exits with raw response snippet |

## Testing

### Run All Tests

```bash
python3 -m pytest tests/ -v
```

### Run Tests with Coverage

```bash
pip install pytest-cov
python3 -m pytest tests/ --cov=main --cov-report=term-missing
```

### Test Suite Breakdown

| File | Tests | What It Covers |
|------|-------|----------------|
| `tests/test_unit.py` | 31 | Individual functions with mocked dependencies |
| `tests/test_integration.py` | 9 | End-to-end pipeline with mocked APIs |
| **Total** | **40** | **99% code coverage** |

### Unit Test Categories

- **`fetch_headlines`** (12 tests) — Success, missing key, connection error, timeout, 401, 429, empty articles, bad status, count parameter, missing source
- **`analyse_sentiment`** (10 tests) — Success, missing key, invalid JSON, non-list response, general exception, rate-limit retry, sentiment validation
- **`display_results`** (9 tests) — Positive/negative/neutral counts, mixed sentiments, output format, empty results, fallback behaviour

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   NewsAPI   │────>│  fetch_      │────>│  headlines  │
│   (HTTP)    │     │  headlines() │     │  (list)     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Gemini AI │<────│  analyse_    │<────│  prompt     │
│   (SDK)     │     │  sentiment() │     │  (string)   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │  results    │
                                          │  (JSON)     │
                                          └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │  display_   │
                                          │  results()  │
                                          └─────────────┘
```

Key design decisions:

- **Single batch call** to Gemini — all headlines analysed in one API request (efficient, low latency)
- **JSON response mode** enforced via `response_mime_type: "application/json"` for reliable parsing
- **Low temperature** (0.1) for consistent, deterministic sentiment classification
- **Retry with backoff** for Gemini rate limits
- **Defensive parsing** — validates response structure before use
