"""Track the official baseline and run deterministic CPU-only evaluations."""

import argparse
import json

from .mlflow_tracking import (
    assign_candidate_alias,
    log_evaluation_run,
    log_official_gpu_baseline,
    promote_official_baseline_after_quality_gate,
    verify_project_history,
)
from .serving import MlflowModelService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track/evaluate the official AraBERT baseline"
    )
    parser.add_argument("--config", default="configs/mlflow-experiments.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_project_history(), sort_keys=True), flush=True)
        return

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)
    baseline = log_official_gpu_baseline(config["official_artifact_dir"])
    print(json.dumps({"official_baseline": baseline}, sort_keys=True), flush=True)
    results = []
    for evaluation in config["evaluations"]:
        results.append(
            log_evaluation_run(
                **evaluation,
                official_run_id=baseline["run_id"],
                model_version=baseline["model_version"],
            )
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    candidate = assign_candidate_alias(baseline["model_version"])
    service = MlflowModelService("models:/arabic-sentiment@candidate")
    prediction = service.predict("الخدمة ممتازة والغرفة نظيفة ومريحة")
    promotion = promote_official_baseline_after_quality_gate(
        model_version=candidate,
        model_load_prediction=prediction,
    )
    audit = verify_project_history()
    print(
        json.dumps(
            {
                "candidate_alias_version": candidate,
                "production_promotion": promotion,
                "history_audit": audit,
                "evaluations": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
