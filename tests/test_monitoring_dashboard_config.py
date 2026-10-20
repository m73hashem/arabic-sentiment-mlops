import json
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy" / "monitoring"


def test_prometheus_scrapes_the_existing_api_metrics_endpoint():
    config = (DEPLOY_DIR / "prometheus.yml").read_text(encoding="utf-8")

    assert "job_name: arabic-sentiment-api" in config
    assert "metrics_path: /metrics" in config
    assert 'targets: ["127.0.0.1:8000"]' in config


def test_grafana_dashboard_uses_existing_api_metrics_and_persisted_psi_metric():
    dashboard = json.loads(
        (DEPLOY_DIR / "grafana-dashboard.json").read_text(encoding="utf-8")
    )
    panels = dashboard["panels"]
    queries = [target["expr"] for panel in panels for target in panel["targets"]]

    assert len(panels) == 4
    assert "histogram_quantile(0.95" in queries[0]
    assert "sentiment_prediction_request_duration_seconds_bucket" in queries[0]
    assert "sentiment_prediction_confidence_score_bucket" in queries[1]
    assert "sentiment_prediction_requests_total" in queries[2]
    psi_queries = [target["expr"] for target in panels[3]["targets"]]
    assert 'sentiment_prediction_psi{source="latest"}' in psi_queries
    assert "sentiment_prediction_psi_alert_threshold" in psi_queries
