from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dvc_pipeline_parameters_and_stages_are_declared():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))

    assert params["seed"] == 42
    assert params["data"] == {
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
    }
    assert params["model"]["type"] == "tfidf_logistic_regression"
    assert set(dvc["stages"]) == {"prepare", "train", "evaluate"}
    for stage in dvc["stages"].values():
        assert stage["cmd"]
        assert stage["deps"]
    assert dvc["stages"]["prepare"]["outs"] == ["data/dvc/processed"]
    assert dvc["stages"]["train"]["outs"] == ["models/dvc-reproducibility/model.joblib"]
    assert dvc["stages"]["evaluate"]["metrics"] == ["metrics/dvc-reproducibility.json"]
