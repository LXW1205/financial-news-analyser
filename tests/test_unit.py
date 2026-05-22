"""
Unit tests for Financial News Sentiment Analyser.
Tests individual functions in isolation with mocked external dependencies.
"""

import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch, call

import requests

# Import the module under test. Module-level load_dotenv() and env reads
# have already executed at import time, so we patch module-level constants
# directly in each test.
import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_newsapi_response(articles: list[dict], status: str = "ok", message: str = "") -> MagicMock:
    """Build a mock requests.Response-like object for NewsAPI."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "status": status,
        "message": message,
        "articles": articles,
    }
    return mock_response


def _make_gemini_response(text: str) -> MagicMock:
    """Build a mock Gemini response object with a .text attribute."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


SAMPLE_HEADLINES = [
    {"title": "Gold prices surge amid market uncertainty", "source": "Reuters", "published_at": "2025-01-01T00:00:00Z", "url": "https://example.com/1"},
    {"title": "Forex markets see heavy volatility", "source": "Bloomberg", "published_at": "2025-01-01T01:00:00Z", "url": "https://example.com/2"},
    {"title": "Crypto regulation talks intensify", "source": "CNBC", "published_at": "2025-01-01T02:00:00Z", "url": "https://example.com/3"},
]

SAMPLE_GEMINI_RESULTS = [
    {"headline": "Gold prices surge amid market uncertainty", "sentiment": "positive", "confidence": 0.85, "reason": "Rising prices indicate bullish sentiment"},
    {"headline": "Forex markets see heavy volatility", "sentiment": "neutral", "confidence": 0.60, "reason": "Volatility alone is directionless"},
    {"headline": "Crypto regulation talks intensify", "sentiment": "negative", "confidence": 0.75, "reason": "Regulation typically creates uncertainty"},
]


# ===================================================================
# Test: fetch_headlines
# ===================================================================

class TestFetchHeadlines(unittest.TestCase):
    """Tests for the fetch_headlines() function."""

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_success(self, mock_get):
        """Mock requests.get returning valid NewsAPI response with 3 articles; verify returned list has 3 dicts with correct keys."""
        articles = [
            {"title": "Headline 1", "source": {"name": "Source A"}, "publishedAt": "2025-01-01", "url": "https://a.com"},
            {"title": "Headline 2", "source": {"name": "Source B"}, "publishedAt": "2025-01-02", "url": "https://b.com"},
            {"title": "Headline 3", "source": {"name": "Source C"}, "publishedAt": "2025-01-03", "url": "https://c.com"},
        ]
        mock_get.return_value = _make_newsapi_response(articles)

        result = main.fetch_headlines(count=3)

        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn("title", item)
            self.assertIn("source", item)
            self.assertIn("published_at", item)
            self.assertIn("url", item)

        self.assertEqual(result[0]["title"], "Headline 1")
        self.assertEqual(result[0]["source"], "Source A")
        self.assertEqual(result[0]["published_at"], "2025-01-01")
        self.assertEqual(result[0]["url"], "https://a.com")

    @patch.object(main, "NEWS_API_KEY", None)
    def test_fetch_headlines_missing_api_key_none(self):
        """Set NEWS_API_KEY to None; verify sys.exit(1) is called."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("NEWS_API_KEY not set", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "your_key_here")
    def test_fetch_headlines_missing_api_key_placeholder(self):
        """Set NEWS_API_KEY to placeholder value; verify sys.exit(1) is called."""
        with self.assertRaises(SystemExit) as cm:
            main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_connection_error(self, mock_get):
        """Mock requests.get raising ConnectionError; verify sys.exit(1)."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Could not connect", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_timeout(self, mock_get):
        """Mock requests.get raising Timeout; verify sys.exit(1)."""
        mock_get.side_effect = requests.exceptions.Timeout()

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("timed out", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_401_unauthorized(self, mock_get):
        """Mock response with status_code 401; verify sys.exit(1)."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error")
        mock_get.return_value = mock_response

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Invalid NewsAPI key", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_429_rate_limit(self, mock_get):
        """Mock response with status_code 429; verify sys.exit(1)."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")
        mock_get.return_value = mock_response

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("rate limit", mock_stderr.getvalue().lower())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_empty_articles(self, mock_get):
        """Mock response with empty articles list; verify sys.exit(1)."""
        mock_response = _make_newsapi_response([])
        mock_get.return_value = mock_response

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("No articles returned", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_status_not_ok(self, mock_get):
        """Mock response with status != "ok"; verify sys.exit(1)."""
        mock_response = _make_newsapi_response(
            [{"title": "Test", "source": {"name": "X"}, "publishedAt": "", "url": ""}],
            status="error",
            message="Something went wrong",
        )
        mock_get.return_value = mock_response

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("NewsAPI error", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_respects_count_param(self, mock_get):
        """Pass count=5, return 10 articles; verify only 5 returned."""
        articles = [
            {"title": f"Headline {i}", "source": {"name": "Source"}, "publishedAt": "", "url": ""}
            for i in range(10)
        ]
        mock_get.return_value = _make_newsapi_response(articles)

        result = main.fetch_headlines(count=5)

        self.assertEqual(len(result), 5)

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_handles_missing_source(self, mock_get):
        """Article with no source field; verify defaults to 'Unknown'."""
        articles = [
            {"title": "No source article", "publishedAt": "2025-01-01", "url": "https://x.com"},
        ]
        mock_get.return_value = _make_newsapi_response(articles)

        result = main.fetch_headlines(count=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "Unknown")
        self.assertEqual(result[0]["title"], "No source article")

    @patch.object(main, "NEWS_API_KEY", "test-api-key-123")
    @patch("main.requests.get")
    def test_fetch_headlines_http_error_other_status(self, mock_get):
        """Mock response with non-401/429 HTTP error; verify sys.exit(1) with generic message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.fetch_headlines()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("HTTP 500", mock_stderr.getvalue())


# ===================================================================
# Test: analyse_sentiment
# ===================================================================

class TestAnalyseSentiment(unittest.TestCase):
    """Tests for the analyse_sentiment() function."""

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_success(self, mock_client_cls):
        """Mock Gemini client returning valid JSON array; verify results parsed correctly."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps(SAMPLE_GEMINI_RESULTS)
        )

        result = main.analyse_sentiment(SAMPLE_HEADLINES)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["sentiment"], "positive")
        self.assertEqual(result[0]["confidence"], 0.85)
        self.assertEqual(result[0]["reason"], "Rising prices indicate bullish sentiment")
        mock_client.models.generate_content.assert_called_once()

    @patch.object(main, "GOOGLE_API_KEY", None)
    def test_analyse_sentiment_missing_api_key_none(self):
        """Set GOOGLE_API_KEY to None; verify sys.exit(1)."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("GOOGLE_API_KEY not set", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "your_key_here")
    def test_analyse_sentiment_missing_api_key_placeholder(self):
        """Set GOOGLE_API_KEY to placeholder value; verify sys.exit(1)."""
        with self.assertRaises(SystemExit) as cm:
            main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_invalid_json(self, mock_client_cls):
        """Mock Gemini returning non-JSON text; verify sys.exit(1)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            "This is not valid JSON {{{"
        )

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("invalid JSON", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_non_list_response(self, mock_client_cls):
        """Mock Gemini returning JSON object (not array); verify sys.exit(1)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps({"sentiment": "positive", "confidence": 0.9})
        )

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Expected a JSON array", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_general_exception(self, mock_client_cls):
        """Mock Gemini raising generic Exception; verify sys.exit(1)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Network unreachable")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Gemini API call failed", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    @patch("main.time.sleep")
    def test_analyse_sentiment_rate_limit_retry(self, mock_sleep, mock_client_cls):
        """Mock first call raising 429, second call succeeding; verify retry logic works."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # First call raises a 429-like error, second call succeeds
        rate_limit_error = Exception("429 RESOURCE_EXHAUSTED. Retry in 65.3 seconds.")
        mock_client.models.generate_content.side_effect = [
            rate_limit_error,
            _make_gemini_response(json.dumps(SAMPLE_GEMINI_RESULTS)),
        ]

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            result = main.analyse_sentiment(SAMPLE_HEADLINES)

        # Verify retry happened
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once()
        # Verify result is correct
        self.assertEqual(len(result), 3)
        self.assertIn("Rate limited", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    @patch("main.time.sleep")
    def test_analyse_sentiment_rate_limit_retry_also_fails(self, mock_sleep, mock_client_cls):
        """Mock first call raising 429, retry also fails; verify sys.exit(1)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        error_429 = Exception("429 RESOURCE_EXHAUSTED")
        error_retry = Exception("Still failing")
        mock_client.models.generate_content.side_effect = [error_429, error_retry]

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.analyse_sentiment(SAMPLE_HEADLINES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("failed after retry", mock_stderr.getvalue())

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_validates_sentiment_values(self, mock_client_cls):
        """Verify returned items have sentiment, confidence, reason keys."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps([
                {"headline": "Test headline", "sentiment": "positive", "confidence": 0.9, "reason": "Good news"},
                {"headline": "Another headline", "sentiment": "negative", "confidence": 0.7, "reason": "Bad news"},
            ])
        )

        result = main.analyse_sentiment([{"title": "Test headline"}, {"title": "Another headline"}])

        for item in result:
            self.assertIn("sentiment", item)
            self.assertIn("confidence", item)
            self.assertIn("reason", item)
            self.assertIn("headline", item)

    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key-456")
    @patch("main.genai.Client")
    def test_analyse_sentiment_empty_headlines_list(self, mock_client_cls):
        """Verify empty headlines list produces valid (empty) prompt and result."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("[]")

        result = main.analyse_sentiment([])

        self.assertEqual(result, [])
        mock_client.models.generate_content.assert_called_once()


# ===================================================================
# Test: display_results
# ===================================================================

class TestDisplayResults(unittest.TestCase):
    """Tests for the display_results() function."""

    def test_display_results_positive_count(self):
        """Mock results with 2 positive items; verify positive count is 2."""
        results = [
            {"headline": "H1", "sentiment": "positive", "confidence": 0.9, "reason": "R1"},
            {"headline": "H2", "sentiment": "positive", "confidence": 0.8, "reason": "R2"},
        ]
        headlines = [{"title": "H1"}, {"title": "H2"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("2 positive", output)
        self.assertIn("0 negative", output)
        self.assertIn("0 neutral", output)

    def test_display_results_negative_count(self):
        """Mock results with 1 negative item; verify negative count is 1."""
        results = [
            {"headline": "H1", "sentiment": "negative", "confidence": 0.7, "reason": "R1"},
        ]
        headlines = [{"title": "H1"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("1 negative", output)
        self.assertIn("0 positive", output)

    def test_display_results_neutral_count(self):
        """Mock results with 3 neutral items; verify neutral count is 3."""
        results = [
            {"headline": "H1", "sentiment": "neutral", "confidence": 0.5, "reason": "R1"},
            {"headline": "H2", "sentiment": "neutral", "confidence": 0.6, "reason": "R2"},
            {"headline": "H3", "sentiment": "neutral", "confidence": 0.4, "reason": "R3"},
        ]
        headlines = [{"title": "H1"}, {"title": "H2"}, {"title": "H3"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("3 neutral", output)
        self.assertIn("0 positive", output)
        self.assertIn("0 negative", output)

    def test_display_results_mixed_sentiments(self):
        """Mock results with mixed sentiments; verify summary line is correct."""
        results = [
            {"headline": "H1", "sentiment": "positive", "confidence": 0.9, "reason": "R1"},
            {"headline": "H2", "sentiment": "negative", "confidence": 0.8, "reason": "R2"},
            {"headline": "H3", "sentiment": "neutral", "confidence": 0.5, "reason": "R3"},
            {"headline": "H4", "sentiment": "positive", "confidence": 0.7, "reason": "R4"},
        ]
        headlines = [{"title": f"H{i}"} for i in range(1, 5)]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("2 positive", output)
        self.assertIn("1 negative", output)
        self.assertIn("1 neutral", output)
        self.assertIn("out of 4 headlines", output)

    def test_display_results_output_format(self):
        """Verify the output contains 'FINANCIAL NEWS SENTIMENT ANALYSIS' header."""
        results = [
            {"headline": "Test", "sentiment": "positive", "confidence": 0.9, "reason": "Good"},
        ]
        headlines = [{"title": "Test"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("FINANCIAL NEWS SENTIMENT ANALYSIS", output)
        self.assertIn("=" * 70, output)
        self.assertIn("[POSITIVE]", output)

    def test_display_results_empty_results(self):
        """Verify empty results list doesn't crash."""
        results = []
        headlines = []

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # Should not raise any exception
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("FINANCIAL NEWS SENTIMENT ANALYSIS", output)
        self.assertIn("0 positive, 0 negative, 0 neutral out of 0 headlines", output)

    def test_display_results_missing_fields_uses_defaults(self):
        """Verify results with missing fields use sensible defaults."""
        results = [
            {"headline": "H1"},  # missing sentiment, confidence, reason
        ]
        headlines = [{"title": "H1"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        # Missing sentiment defaults to "neutral" (falls through else branch)
        self.assertIn("[NEUTRAL]", output)
        self.assertIn("0%", output)  # default confidence 0.0
        self.assertIn("No reason provided", output)

    def test_display_results_uses_headline_from_results(self):
        """Verify headline is taken from results, not fallback to headlines list."""
        results = [
            {"headline": "Result Headline", "sentiment": "positive", "confidence": 0.9, "reason": "R"},
        ]
        headlines = [{"title": "Original Headline"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("Result Headline", output)
        self.assertNotIn("Original Headline", output)

    def test_display_results_fallback_to_headlines_list(self):
        """Verify headline falls back to headlines list when not in results."""
        results = [
            {"sentiment": "positive", "confidence": 0.9, "reason": "R"},
        ]
        headlines = [{"title": "Fallback Headline"}]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.display_results(results, headlines)
            output = mock_stdout.getvalue()

        self.assertIn("Fallback Headline", output)


if __name__ == "__main__":
    unittest.main()
