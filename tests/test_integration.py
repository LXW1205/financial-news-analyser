"""
Integration tests for Financial News Sentiment Analyser.
Tests the full pipeline end-to-end with mocked external APIs.
"""

import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_newsapi_response(articles: list[dict], status: str = "ok") -> MagicMock:
    """Build a mock requests.Response-like object for NewsAPI."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "status": status,
        "articles": articles,
    }
    return mock_response


def _make_gemini_response(text: str) -> MagicMock:
    """Build a mock Gemini response object with a .text attribute."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


def _build_articles(n: int) -> list[dict]:
    """Generate n sample NewsAPI article dicts."""
    return [
        {
            "title": f"Financial Headline {i}",
            "source": {"name": f"Source {i}"},
            "publishedAt": f"2025-01-0{i}T00:00:00Z",
            "url": f"https://example.com/{i}",
        }
        for i in range(1, n + 1)
    ]


def _build_gemini_results(titles: list[str]) -> list[dict]:
    """Generate Gemini-style sentiment results for given titles."""
    sentiments = ["positive", "negative", "neutral"]
    return [
        {
            "headline": title,
            "sentiment": sentiments[i % 3],
            "confidence": 0.5 + (i * 0.1),
            "reason": f"Analysis of {title}",
        }
        for i, title in enumerate(titles)
    ]


# ===================================================================
# Integration Tests
# ===================================================================

class TestFullPipeline(unittest.TestCase):
    """Integration tests for the complete main() pipeline."""

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_full_pipeline_with_mocked_apis(self, mock_get, mock_client_cls):
        """Mock both NewsAPI and Gemini responses end-to-end; verify main() completes without error."""
        # Setup NewsAPI mock
        articles = _build_articles(3)
        mock_get.return_value = _make_newsapi_response(articles)

        # Setup Gemini mock
        titles = [a["title"] for a in articles]
        gemini_results = _build_gemini_results(titles)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps(gemini_results)
        )

        # Run main() — should not raise
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.main()
            output = mock_stdout.getvalue()

        # Verify pipeline completed
        self.assertIn("Fetching financial headlines", output)
        self.assertIn("Fetched 3 headlines", output)
        self.assertIn("Analysing sentiment", output)
        self.assertIn("Analysed 3 headlines", output)
        self.assertIn("FINANCIAL NEWS SENTIMENT ANALYSIS", output)
        self.assertIn("[JSON Output]", output)

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_newsapi_failure(self, mock_get, mock_client_cls):
        """Mock NewsAPI failure; verify pipeline exits gracefully (doesn't crash uncaught)."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Could not connect", mock_stderr.getvalue())

        # Gemini should never be called
        mock_client_cls.assert_not_called()

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_gemini_failure(self, mock_get, mock_client_cls):
        """Mock NewsAPI success but Gemini failure; verify pipeline exits gracefully."""
        # NewsAPI succeeds
        articles = _build_articles(2)
        mock_get.return_value = _make_newsapi_response(articles)

        # Gemini fails
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("Gemini is down")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Gemini API call failed", mock_stderr.getvalue())

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_json_output_valid(self, mock_get, mock_client_cls):
        """Mock both APIs, capture stdout; verify the JSON output section is valid parseable JSON."""
        articles = _build_articles(3)
        mock_get.return_value = _make_newsapi_response(articles)

        titles = [a["title"] for a in articles]
        gemini_results = _build_gemini_results(titles)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps(gemini_results)
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.main()
            output = mock_stdout.getvalue()

        # Extract JSON output section — everything after "[JSON Output]"
        json_marker = "[JSON Output]"
        self.assertIn(json_marker, output)
        json_section = output.split(json_marker, 1)[1].strip()

        # Should be valid JSON
        parsed = json.loads(json_section)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 3)
        for item in parsed:
            self.assertIn("headline", item)
            self.assertIn("sentiment", item)
            self.assertIn("confidence", item)
            self.assertIn("reason", item)

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_headline_count_matches(self, mock_get, mock_client_cls):
        """Mock 5 headlines from NewsAPI; verify Gemini receives exactly 5 and results have 5 items."""
        articles = _build_articles(5)
        mock_get.return_value = _make_newsapi_response(articles)

        titles = [a["title"] for a in articles]
        gemini_results = _build_gemini_results(titles)
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps(gemini_results)
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.main()
            output = mock_stdout.getvalue()

        # Verify Gemini was called exactly once
        mock_client.models.generate_content.assert_called_once()

        # Verify the prompt contains all 5 headlines
        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs["contents"]
        for i in range(1, 6):
            self.assertIn(f"{i}. Financial Headline {i}", prompt)

        # Verify output shows 5 items
        self.assertIn("Fetched 5 headlines", output)
        self.assertIn("Analysed 5 headlines", output)
        self.assertIn("out of 5 headlines", output)

        # Verify JSON output has 5 items
        json_section = output.split("[JSON Output]", 1)[1].strip()
        parsed = json.loads(json_section)
        self.assertEqual(len(parsed), 5)

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_display_summary_correct(self, mock_get, mock_client_cls):
        """Verify the summary line in display output matches the actual sentiment counts."""
        articles = _build_articles(6)
        mock_get.return_value = _make_newsapi_response(articles)

        # Create results with known sentiment distribution: 3 positive, 2 negative, 1 neutral
        titles = [a["title"] for a in articles]
        gemini_results = [
            {"headline": titles[0], "sentiment": "positive", "confidence": 0.9, "reason": "R1"},
            {"headline": titles[1], "sentiment": "positive", "confidence": 0.8, "reason": "R2"},
            {"headline": titles[2], "sentiment": "positive", "confidence": 0.7, "reason": "R3"},
            {"headline": titles[3], "sentiment": "negative", "confidence": 0.85, "reason": "R4"},
            {"headline": titles[4], "sentiment": "negative", "confidence": 0.75, "reason": "R5"},
            {"headline": titles[5], "sentiment": "neutral", "confidence": 0.5, "reason": "R6"},
        ]
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            json.dumps(gemini_results)
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main.main()
            output = mock_stdout.getvalue()

        self.assertIn("3 positive", output)
        self.assertIn("2 negative", output)
        self.assertIn("1 neutral", output)
        self.assertIn("out of 6 headlines", output)

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_empty_articles_from_api(self, mock_get, mock_client_cls):
        """Mock NewsAPI returning empty articles; verify pipeline exits before Gemini call."""
        mock_get.return_value = _make_newsapi_response([])

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("No articles returned", mock_stderr.getvalue())

        mock_client_cls.assert_not_called()

    @patch.object(main, "NEWS_API_KEY", None)
    @patch.object(main, "GOOGLE_API_KEY", "test-gemini-key")
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_missing_news_api_key(self, mock_get, mock_client_cls):
        """Mock missing NEWS_API_KEY; verify pipeline exits before making any API calls."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("NEWS_API_KEY not set", mock_stderr.getvalue())

        mock_get.assert_not_called()
        mock_client_cls.assert_not_called()

    @patch.object(main, "NEWS_API_KEY", "test-news-key")
    @patch.object(main, "GOOGLE_API_KEY", None)
    @patch("main.genai.Client")
    @patch("main.requests.get")
    def test_pipeline_missing_gemini_api_key(self, mock_get, mock_client_cls):
        """Mock missing GOOGLE_API_KEY; verify pipeline fetches headlines but exits at sentiment step."""
        articles = _build_articles(2)
        mock_get.return_value = _make_newsapi_response(articles)

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("GOOGLE_API_KEY not set", mock_stderr.getvalue())

        # NewsAPI was called
        mock_get.assert_called_once()
        # Gemini client was never instantiated
        mock_client_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
