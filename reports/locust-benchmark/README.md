# FastAPI vs BentoML Locust benchmark

This benchmark compares the single-review `POST /predict` endpoint on the
existing FastAPI and BentoML containers. Both must mount the same
`models/full-gpu-arabert-inference/` directory read-only and report the official
model version. The six fixed Arabic input strings are shared by both services;
responses are checked for HTTP 200, the binary label, finite confidence in
`[0, 1]`, and the expected model version.

## Install isolated benchmark tooling

From the repository root, use a separate environment so Locust does not change
the pinned inference dependencies:

```bash
python3.13 -m venv /tmp/arabic-sentiment-locust-venv
/tmp/arabic-sentiment-locust-venv/bin/python -m pip install -r requirements-benchmark.txt
```

## Start one service at a time

The comparison runs services sequentially, never under simultaneous load. Both
containers use the same local artifact mount, `HF_HUB_OFFLINE=1`, CPU quota,
memory limit, host port, and Locust settings. Example FastAPI launch:

```bash
docker run -d --name arabic-sentiment-fastapi-bench \
  --cpus=2 --memory=2g --memory-swap=2g \
  -p 127.0.0.1:8100:8000 \
  --mount type=bind,src="$PWD/models/full-gpu-arabert-inference",dst=/models/full-gpu-arabert-inference,readonly \
  -e SENTIMENT_MODEL_DIR=/models/full-gpu-arabert-inference \
  -e HF_HUB_OFFLINE=1 arabic-sentiment-mlops:locust-benchmark
```

For Bento, use `arabic-sentiment-mlops-bento:phase4` and map
`127.0.0.1:8100:3000` with the same limits and mount. Wait for startup, verify
`/health`, then run the benchmark. Stop/remove the first service before
starting the second. The recorded comparison uses 2 CPUs and 2 GiB memory per
service; this is a local experimental limit, not a production sizing claim.

The FastAPI `:locust-benchmark` image was freshly built from the repository's
existing Dockerfile because the older local `:phase4` image expected a
`training_config.json` absent from the official inference-only artifact. Bento
tag `arabic-sentiment:3cez6pglvwm3avli` could not be containerized with its
saved dependency environment; the successful Bento build used
`arabic-sentiment:kdcvvqglwcnfslmb` and produced the `:phase4` Docker image.
The model directory is mounted read-only in both cases; neither image contains
the model weights.

## Run one trial

`benchmarks/run_benchmark.py` performs eight unmeasured sequential warm-up
predictions, then launches headless Locust and samples Docker CPU/memory with
`docker stats --no-stream` (the local daemon returned samples roughly every
2–2.5 seconds). Every trial writes Locust raw CSVs, Docker stats CSV, log, and
JSON metadata under the requested output prefix.

```bash
PYTHONPATH=. /tmp/arabic-sentiment-locust-venv/bin/python -m benchmarks.run_benchmark \
  --service-name fastapi --container arabic-sentiment-fastapi-bench \
  --host http://127.0.0.1:8100 --users 1 --spawn-rate 1 --duration 60 \
  --output-prefix reports/locust-benchmark/raw/fastapi-u1-r1
```

The completed primary comparison used 1 user, spawn rate 1 user/s, 60 seconds,
and three repetitions per service. Pilot runs reached approximately 200% CPU
for the two-core container at this load, so higher concurrency was not tested.
See [results.md](results.md) for measured values, environment, deviations, and
links to each raw CSV/metadata file. Do not compare Bento's batch endpoint
against single-review FastAPI requests; batch testing is a separate experiment.
