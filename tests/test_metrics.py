import unittest

from flask import Flask

from backend.metrics import instrument_app


class MetricsMiddlewareTests(unittest.TestCase):
    def test_records_api_requests_and_exposes_metrics(self):
        app = Flask(__name__)

        @app.get("/api/health")
        def health():
            return {"status": "ok"}

        instrument_app(app)
        client = app.test_client()

        response = client.get("/api/health")
        missing = client.get("/missing/123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(missing.status_code, 404)
        metrics = client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        metrics_text = metrics.get_data(as_text=True)

        self.assertIn(
            'backend_http_requests_total{method="GET",route="/api/health",status="200"} 1.0',
            metrics_text,
        )
        self.assertIn(
            'backend_http_requests_total{method="GET",route="unmatched",status="404"} 1.0',
            metrics_text,
        )
        self.assertNotIn("/missing/123", metrics_text)


if __name__ == "__main__":
    unittest.main()
