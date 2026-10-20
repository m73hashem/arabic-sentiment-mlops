# FastAPI vs BentoML Locust benchmark results

## Scope and method

This is a local, single-host runtime comparison of the single-review `POST /predict` APIs. It is not a production capacity test or a benchmark of BentoML's batch endpoint.

- Date: 2026-10-19 (UTC; trial timestamps are in the raw metadata).
- Model: `aubmindlab/bert-base-arabertv02` official full-dataset GPU inference artifact, exposed by both services as `aubmindlab/bert-base-arabertv02-official-full-gpu-seed-42`.
- Mounted artifact: `models/full-gpu-arabert-inference/`, read-only at `/models/full-gpu-arabert-inference` in both containers.
- `model.safetensors` SHA-256: `1d093e6136126282959f3d7b7fbbf7457f38d3788551366920f75228e660ba93`.
- Workload: six fixed Arabic reviews (short and longer examples, mixed sentiment), same JSON body `{"review": ...}` and same sequence for each service; one closed-loop Locust user, zero wait, spawn rate 1 user/s, 60-second measured duration, 120-second request timeout, and eight sequential warm-up requests excluded from Locust statistics.
- Repetitions: three per service, run sequentially. The services did not share load at the same time. Both used `127.0.0.1:8100` during their respective run.
- Equivalent Docker limits: `--cpus=2` (`NanoCpus=2,000,000,000`), `--memory=2g` and `--memory-swap=2g` (`2,147,483,648` bytes each). CPU percentages below follow Docker's convention where 100% is one CPU; approximately 200% consumes the two-CPU quota.
- Validation: every measured request checked HTTP 200, JSON schema, `negative|positive` label, finite confidence in `[0,1]`, and expected model version. No request failures were recorded.
- Host: Linux, 4 CPUs and about 7.1 GiB RAM; services were limited to 2 CPUs and 2 GiB each. Docker stats were sampled with `docker stats --no-stream`; command overhead yielded roughly 2–2.5 seconds between samples. CPU/memory values are container samples, not host totals.
- Tool versions: Python 3.13.16; Locust 2.46.7; Docker Engine 29.8.0; FastAPI 0.115.12; BentoML 1.4.39; PyTorch 2.14.1+cpu; Transformers 5.18.0.

The workload implementation is in [`benchmarks/locustfile.py`](../../benchmarks/locustfile.py), with the fixed review corpus in [`benchmarks/reviews.py`](../../benchmarks/reviews.py). The repeatable runner and Docker resource sampler are in [`benchmarks/run_benchmark.py`](../../benchmarks/run_benchmark.py). Setup and launch instructions are in [`README.md`](README.md).

## Images and model delivery

| Service | Image used | Image size | Model delivery |
|---|---|---:|---|
| FastAPI | `arabic-sentiment-mlops:locust-benchmark` | 1,717,646,499 bytes | External read-only bind mount |
| BentoML | `arabic-sentiment-mlops-bento:phase4` | 2,416,959,426 bytes | External read-only bind mount |

The FastAPI image was freshly built from the existing repository Dockerfile. The older local `arabic-sentiment-mlops:phase4` image expected `training_config.json`, which is not part of the official inference-only artifact, so it was not used. The Bento image is the successful containerization of Bento `arabic-sentiment:kdcvvqglwcnfslmb`, built from the current Bento service configuration. The earlier Bento tag `arabic-sentiment:3cez6pglvwm3avli` did not containerize successfully with its saved dependency environment and was not used. Bento logs for the tested image confirmed loading all 201/201 model weight tensors. FastAPI and Bento both returned HTTP 200 for the same preflight review, with the same model version and confidence.

The official artifact was absent from image build contexts/layers and only supplied at runtime by the read-only mount. Image size is reported separately from runtime memory; it is not a measure of model memory use.

## Measured results

Latencies are milliseconds. RPS and percentiles are Locust aggregate values for each 60-second run. CPU and memory are means/maxima across the Docker stats samples collected during that run.

| Service | Run | Requests | Failures | RPS | p50 | p95 | p99 | Mean CPU | Peak CPU | Mean memory | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FastAPI | 1 | 495 | 0 | 8.37 | 110 | 190 | 220 | 187.46% | 200.79% | 349.1 MiB | 349.6 MiB |
| FastAPI | 2 | 498 | 0 | 8.41 | 110 | 180 | 210 | 188.12% | 202.16% | 349.1 MiB | 349.6 MiB |
| FastAPI | 3 | 506 | 0 | 8.56 | 110 | 170 | 210 | 188.34% | 200.46% | 349.3 MiB | 349.6 MiB |
| BentoML | 1 | 348 | 0 | 5.87 | 160 | 260 | 360 | 174.04% | 199.94% | 789.6 MiB | 790.4 MiB |
| BentoML | 2 | 435 | 0 | 7.35 | 130 | 210 | 260 | 186.97% | 202.69% | 716.9 MiB | 717.5 MiB |
| BentoML | 3 | 422 | 0 | 7.11 | 130 | 210 | 230 | 185.18% | 200.54% | 715.4 MiB | 715.7 MiB |

Across the three repetitions, arithmetic means were:

| Service | Mean RPS | Mean per-run p50 | Mean per-run p95 | Mean per-run p99 | Total requests / failures | Mean sampled CPU | Mean sampled memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastAPI | 8.45 | 110 ms | 180 ms | 213 ms | 1,499 / 0 | 187.97% | 349.2 MiB |
| BentoML | 6.78 | 140 ms | 227 ms | 283 ms | 1,205 / 0 | 182.07% | 740.6 MiB |

The arithmetic means of the per-run percentiles are descriptive summaries; they are not percentiles recomputed from a merged request-level sample. Per-run Locust CSVs remain the source data. No non-success responses, Locust failures, or request timeouts were recorded in these six measured runs. The raw failure CSVs contain only headers/no failure rows.

Under this specific one-user, two-CPU local workload, FastAPI measured higher throughput and lower p50/p95/p99 latency, with lower sampled container memory. This is a descriptive result for these repetitions, not a claim that FastAPI is generally superior or that either system is production-ready. Bento's first run differed more from its later two; all runs are retained and reported.

## Concurrency and limitations

Pilot trials at one user already showed sampled CPU near the 2-CPU quota (about 200%) for both services. To avoid presenting overloaded higher-concurrency results as reliable, the measured matrix stopped at one user. Thus no 5/10/25/50/100-user scalability curve or maximum sustainable throughput was measured, and a saturation point beyond the observed CPU utilization cannot be identified. No batch endpoint test was performed, and no claim is made that Bento adaptive batching merged requests or improved performance. The local host, one-user closed-loop traffic, and limited repetitions do not represent multi-client production traffic.

The containers ran non-root and non-privileged with read-only model mounts and equal CPU/memory/swap caps. The root filesystem remained writable and additional hardening was not part of this benchmark; this is not a production security assessment.

## Raw artifacts

All measured CSVs and metadata are retained under `raw/`. Each repetition has Locust stats, failures, exceptions, stats history, Docker stats, log, and JSON metadata files. Representative prefixes:

- FastAPI: `raw/fastapi-u1-r1` through `raw/fastapi-u1-r3`
- BentoML: `raw/bentoml-u1-r1` through `raw/bentoml-u1-r3`

Exploratory pilots and smoke attempts are also retained with `pilot-` and `smoke-` prefixes. They are not included in the comparison table. One early FastAPI 2-CPU smoke attempt had a Docker stats monitor exception; its raw output remains available and it is not treated as a completed measured repetition.

Re-run commands and the precise settings are recorded in each `*_metadata.json`. For example:

```bash
PYTHONPATH=. /tmp/arabic-sentiment-locust-venv/bin/python -m benchmarks.run_benchmark \
  --service-name fastapi --container arabic-sentiment-fastapi-bench \
  --host http://127.0.0.1:8100 --users 1 --spawn-rate 1 --duration 60 \
  --output-prefix reports/locust-benchmark/raw/fastapi-u1-r1
```

Use the corresponding Bento container/service name and a distinct output prefix for Bento trials. Do not run both containers under load simultaneously.
