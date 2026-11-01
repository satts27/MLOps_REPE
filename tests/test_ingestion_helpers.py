import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

import pandas as pd

from ingestion.collect_market_data import (
    PROFILE_DEFAULTS,
    finite_value,
    market_documents,
    news_documents,
)


class IngestionHelperTests(unittest.TestCase):
    def test_profile_defaults_are_configured(self):
        self.assertEqual(PROFILE_DEFAULTS["nightly"], {"lookback_days": 5, "news_days": 1})
        self.assertEqual(PROFILE_DEFAULTS["bootstrap"], {"lookback_days": 120, "news_days": 7})

    def test_finite_value_handles_missing_and_numeric_values(self):
        self.assertIsNone(finite_value(None))
        self.assertIsNone(finite_value(float("nan")))
        self.assertEqual(finite_value(12), 12.0)

    def test_market_documents_normalize_values_and_dates(self):
        collected_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            [{
                "trading_date": datetime(2026, 1, 1).date(),
                "open": 10,
                "close": 11.5,
                "adj_close": 11.5,
                "rsi": float("nan"),
            }]
        )

        [document] = market_documents("ABC", frame, collected_at)

        self.assertEqual(document["ticker"], "ABC")
        self.assertEqual(document["open"], 10.0)
        self.assertEqual(document["close"], 11.5)
        self.assertIsNone(document["indicators"]["rsi"])
        self.assertEqual(
            document["trading_date"], datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(document["collected_at"], collected_at)

    def test_news_documents_skip_incomplete_items_and_generate_stable_ids(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, tzinfo=timezone.utc)
        collected_at = datetime(2026, 1, 3, tzinfo=timezone.utc)
        client = Mock()
        client.company_news.return_value = [
            {"headline": "  Earnings update  ", "datetime": 1767225600},
            {"headline": "   ", "datetime": 1767225600},
            {"headline": "No timestamp"},
        ]

        [article] = news_documents("ABC", client, start, end, collected_at)
        [same_article] = news_documents("ABC", client, start, end, collected_at)

        self.assertEqual(article["headline"], "Earnings update")
        self.assertEqual(article["article_id"], same_article["article_id"])
        self.assertEqual(article["published_at"], datetime(2026, 1, 1, tzinfo=timezone.utc))
        client.company_news.assert_called_with("ABC", _from="2026-01-01", to="2026-01-02")


if __name__ == "__main__":
    unittest.main()
