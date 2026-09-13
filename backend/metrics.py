"""HTTP metrics for the single-process Flask backend."""

from time import perf_counter

from flask import g, request
from prometheus_client import Counter, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware


def instrument_app(app):
    requests = Counter(
        "backend_http_requests_total", "Completed HTTP requests.",
        ["method", "route", "status"],
    )
    duration = Histogram(
        "backend_http_request_duration_seconds", "HTTP request duration.",
        ["method", "route"],
        buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
    )

    @app.before_request
    def start_timer():
        g.metrics_started_at = perf_counter()

    @app.after_request
    def record_request(response):
        # Use route templates to avoid a new time series for every ticker/URL.
        route = request.url_rule.rule if request.url_rule else "unmatched"
        requests.labels(request.method, route, str(response.status_code)).inc()
        duration.labels(request.method, route).observe(
            perf_counter() - g.metrics_started_at
        )
        return response

    # Scrapes bypass Flask instrumentation so they do not inflate API traffic.
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})
