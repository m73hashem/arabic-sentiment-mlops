import json
from pathlib import Path

import yaml

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy" / "monitoring"
ROOT = DEPLOY_DIR.parents[1]


def test_prometheus_scrapes_the_existing_api_metrics_endpoint():
    config = (DEPLOY_DIR / "prometheus.yml").read_text(encoding="utf-8")

    assert "job_name: arabic-sentiment-api" in config
    assert "metrics_path: /metrics" in config
    assert 'targets: ["api:8000"]' in config


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
    assert all(panel["datasource"]["uid"] == "prometheus" for panel in panels)


def test_compose_starts_api_prometheus_and_provisioned_grafana():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert {"api", "prometheus", "grafana"}.issubset(services)
    assert services["api"]["environment"]["SENTIMENT_PSI_REPORT_PATH"] == (
        "/app/reports/monitoring/drift-report.json"
    )
    assert (
        "./reports/monitoring:/app/reports/monitoring:ro" in services["api"]["volumes"]
    )
    assert services["prometheus"]["image"] == "prom/prometheus:v3.14.0"
    assert services["grafana"]["image"] == "grafana/grafana:13.2.4"
    assert "${GRAFANA_ADMIN_PASSWORD:?Set GRAFANA_ADMIN_PASSWORD in the shell}" == (
        services["grafana"]["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
    )
    assert "127.0.0.1:9090:9090" in services["prometheus"]["ports"]
    assert "127.0.0.1:3000:3000" in services["grafana"]["ports"]

    datasource = yaml.safe_load(
        (
            DEPLOY_DIR / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        ).read_text(encoding="utf-8")
    )["datasources"][0]
    assert datasource["uid"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"

    dashboard_provider = yaml.safe_load(
        (
            DEPLOY_DIR / "grafana" / "provisioning" / "dashboards" / "dashboard.yml"
        ).read_text(encoding="utf-8")
    )["providers"][0]
    assert dashboard_provider["options"]["path"] == "/var/lib/grafana/dashboards"
