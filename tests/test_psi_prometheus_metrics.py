import json

from arabic_sentiment.monitoring import (
    PSI_ALERT_THRESHOLD,
    PSI_SCORE,
    prometheus_payload,
)


def test_persisted_psi_report_updates_exported_score_and_threshold(tmp_path):
    report_path = tmp_path / "drift-report.json"
    report_path.write_text(
        json.dumps(
            {
                "metric": "confidence_score",
                "psi": 0.37,
                "threshold": PSI_ALERT_THRESHOLD,
                "alert": True,
            }
        ),
        encoding="utf-8",
    )

    payload, _ = prometheus_payload(report_path)
    text = payload.decode("utf-8")

    assert 'sentiment_prediction_psi{source="latest"} 0.37' in text
    assert f"sentiment_prediction_psi_alert_threshold {PSI_ALERT_THRESHOLD}" in text
    PSI_SCORE.remove("latest")


def test_missing_or_invalid_report_does_not_leave_a_stale_psi_score(tmp_path):
    report_path = tmp_path / "drift-report.json"
    report_path.write_text(
        json.dumps(
            {
                "metric": "confidence_score",
                "psi": 0.37,
                "threshold": PSI_ALERT_THRESHOLD,
                "alert": True,
            }
        ),
        encoding="utf-8",
    )
    prometheus_payload(report_path)
    report_path.unlink()

    payload, _ = prometheus_payload(report_path)
    text = payload.decode("utf-8")

    assert 'sentiment_prediction_psi{source="latest"}' not in text
    assert f"sentiment_prediction_psi_alert_threshold {PSI_ALERT_THRESHOLD}" in text


def test_psi_threshold_gauge_keeps_the_existing_alert_threshold(tmp_path):
    payload, _ = prometheus_payload(tmp_path / "missing-report.json")

    assert f"sentiment_prediction_psi_alert_threshold {PSI_ALERT_THRESHOLD}" in (
        payload.decode("utf-8")
    )
