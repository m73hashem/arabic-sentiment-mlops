import json

import pytest

from arabic_sentiment.quality_gate import QualityGateError, check_quality, main


def _write_inputs(tmp_path, candidate_score=0.85, baseline_score=0.80):
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(
        json.dumps(
            {
                "validation": {"accuracy": 0.9, "f1_macro": candidate_score},
                "test": {"accuracy": 0.9, "f1_macro": candidate_score},
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric": "f1_macro",
                "split": "test",
                "baseline": {
                    "value": baseline_score,
                    "source": "test fixture only",
                    "model": "test fixture model",
                },
            }
        ),
        encoding="utf-8",
    )
    return candidate_path, baseline_path


def test_quality_gate_accepts_candidate_above_baseline(tmp_path, capsys):
    candidate, baseline = _write_inputs(tmp_path)

    assert main(["--candidate", str(candidate), "--baseline", str(baseline)]) == 0
    assert "QUALITY GATE PASS" in capsys.readouterr().out


def test_quality_gate_rejects_candidate_below_baseline(tmp_path, capsys):
    candidate, baseline = _write_inputs(tmp_path, candidate_score=0.79)

    assert main(["--candidate", str(candidate), "--baseline", str(baseline)]) == 1
    assert "QUALITY GATE FAIL" in capsys.readouterr().err


def test_quality_gate_accepts_equality_with_baseline(tmp_path, capsys):
    candidate, baseline = _write_inputs(tmp_path, candidate_score=0.80)

    assert main(["--candidate", str(candidate), "--baseline", str(baseline)]) == 0
    output = capsys.readouterr().out
    assert "f1_macro=0.8000000000 >= baseline=0.8000000000" in output
    assert "equality with the baseline is accepted" in output


@pytest.mark.parametrize(
    ("candidate_text", "baseline_text", "message"),
    [
        (
            '{"validation":{"f1_macro":0.8},"test":{"f1":0.8}}',
            None,
            "missing test.f1_macro",
        ),
        (
            '{"validation":{"f1_macro":"high"},"test":{"f1_macro":0.8}}',
            None,
            "numeric f1_macro",
        ),
        (
            '{"validation":{"f1_macro":NaN},"test":{"f1_macro":0.8}}',
            None,
            "finite and within",
        ),
        (
            '{"validation":{"f1_macro":0.8},"test":{"f1_macro":1.01}}',
            None,
            "finite and within",
        ),
        (
            '{"validation":{"f1_macro":0.8},"test":{"f1_macro":' + "9" * 400 + "}}",
            None,
            "finite and within",
        ),
        (
            '{"validation":{"f1_macro":0.8},"test":{"f1_macro":0.8}}',
            '{"schema_version":2,"metric":"f1_macro","split":"test","baseline":{"value":0.8,"source":"fixture","model":"fixture"}}',
            "Unsupported quality-gate schema version",
        ),
        (
            "not JSON",
            None,
            "Malformed JSON in candidate metrics",
        ),
    ],
)
def test_quality_gate_rejects_missing_or_invalid_metrics(
    tmp_path, candidate_text, baseline_text, message
):
    candidate, baseline = _write_inputs(tmp_path)
    candidate.write_text(candidate_text, encoding="utf-8")
    if baseline_text is not None:
        baseline.write_text(baseline_text, encoding="utf-8")

    with pytest.raises(QualityGateError, match=message):
        check_quality(candidate, baseline)


def test_quality_gate_fails_if_candidate_metrics_file_is_missing(tmp_path):
    _, baseline = _write_inputs(tmp_path)

    with pytest.raises(QualityGateError, match="Cannot read candidate metrics"):
        check_quality(tmp_path / "absent.json", baseline)
