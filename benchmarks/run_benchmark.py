"""Run one reproducible Locust trial and capture Docker resource samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmarks.reviews import MODEL_VERSION, REVIEWS

ROOT = Path(__file__).resolve().parents[1]
LOCUSTFILE = ROOT / "benchmarks" / "locustfile.py"
ARTIFACT = ROOT / "models" / "full-gpu-arabert-inference"


def _docker_inspect(container: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .HostConfig}}", container],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _memory_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9.]+)\s*(B|KiB|MiB|GiB|kB|MB|GB)\s*", value)
    if not match:
        raise ValueError(f"Unrecognized Docker memory value: {value!r}")
    scale = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }
    return int(float(match.group(1)) * scale[match.group(2)])


def _sample_docker_stats(container: str, path: Path, stop: threading.Event) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "timestamp_utc",
                "cpu_percent",
                "memory_usage",
                "memory_bytes",
                "memory_percent",
            ]
        )
        while not stop.is_set():
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
                    container,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    cpu, memory, memory_percent = result.stdout.strip().split("\t")
                    used_memory = memory.split(" / ", maxsplit=1)[0]
                    writer.writerow(
                        [
                            datetime.now(UTC).isoformat(),
                            float(cpu.rstrip("%")),
                            memory,
                            _memory_bytes(used_memory),
                            float(memory_percent.rstrip("%")),
                        ]
                    )
                    output.flush()
                except (ValueError, IndexError):
                    pass
            stop.wait(0.5)


def _wait_for_health(host: str, timeout_seconds: int = 120) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{host.rstrip('/')}/health", timeout=3) as response:
                payload = json.load(response)
            if response.status == 200 and payload.get("status") == "ok":
                return time.monotonic() - started
            last_error = "unexpected health response"
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            last_error = type(exc).__name__
        time.sleep(1)
    raise RuntimeError(f"Service health check timed out: {last_error}")


def _warm_up(host: str, count: int) -> None:
    for index in range(count):
        request_body = json.dumps({"review": REVIEWS[index % len(REVIEWS)]}).encode()
        request = Request(
            f"{host.rstrip('/')}/predict",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Warm-up prediction failed: {type(exc).__name__}"
            ) from exc
        if (
            result.get("label") not in {"negative", "positive"}
            or result.get("model_version") != MODEL_VERSION
        ):
            raise RuntimeError(
                "Warm-up response did not match the official model contract"
            )


def _parse_aggregate(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("Name") == "Aggregated":
                return row
    raise RuntimeError(f"Locust aggregate row is missing from {path}")


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", required=True, help="Service host, e.g. http://127.0.0.1:8100"
    )
    parser.add_argument(
        "--container", required=True, help="Running Docker container name"
    )
    parser.add_argument("--users", required=True, type=int)
    parser.add_argument("--spawn-rate", required=True, type=float)
    parser.add_argument(
        "--duration", required=True, type=int, help="Measured duration in seconds"
    )
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--service-name", required=True, choices=["fastapi", "bentoml"])
    args = parser.parse_args()
    if (
        args.users < 1
        or args.spawn_rate <= 0
        or args.duration < 1
        or args.warmup_requests < 0
    ):
        parser.error(
            "users/duration must be positive, spawn-rate positive, and warmup nonnegative"
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    docker_config = _docker_inspect(args.container)
    artifact_hash = hashlib.sha256(
        (ARTIFACT / "model.safetensors").read_bytes()
    ).hexdigest()
    health_wait_seconds = _wait_for_health(args.host)
    warmup_started = time.monotonic()
    _warm_up(args.host, args.warmup_requests)
    warmup_seconds = time.monotonic() - warmup_started

    prefix = str(args.output_prefix)
    csv_file = Path(f"{prefix}_stats.csv")
    docker_stats_file = Path(f"{prefix}_docker_stats.csv")
    log_file = Path(f"{prefix}.log")
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        args.host,
        "--users",
        str(args.users),
        "--spawn-rate",
        str(args.spawn_rate),
        "--run-time",
        f"{args.duration}s",
        "--stop-timeout",
        "120",
        "--csv",
        prefix,
        "--csv-full-history",
        "--loglevel",
        "WARNING",
    ]
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_sample_docker_stats,
        args=(args.container, docker_stats_file, stop_monitor),
        daemon=True,
    )
    env = os.environ.copy()
    env["LOCUST_EXPECTED_MODEL_VERSION"] = MODEL_VERSION
    started_at = datetime.now(UTC)
    monitor.start()
    try:
        with log_file.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        stop_monitor.set()
        monitor.join(timeout=3)
    completed_at = datetime.now(UTC)
    aggregate = _parse_aggregate(csv_file)
    metadata = {
        "service": args.service_name,
        "host": args.host,
        "container": args.container,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds_requested": args.duration,
        "users": args.users,
        "spawn_rate_users_per_second": args.spawn_rate,
        "warmup_requests_excluded_from_locust_stats": args.warmup_requests,
        "health_wait_seconds": round(health_wait_seconds, 3),
        "warmup_duration_seconds": round(warmup_seconds, 3),
        "model_version": MODEL_VERSION,
        "model_safetensors_sha256": artifact_hash,
        "docker_host_config": {
            "cpu_quota_nano_cpus": docker_config.get("NanoCpus"),
            "memory_limit_bytes": docker_config.get("Memory"),
            "memory_swap_limit_bytes": docker_config.get("MemorySwap"),
        },
        "locust_command": command,
        "locust_exit_code": result.returncode,
        "locust_aggregate": aggregate,
        "files": {
            "stats_csv": csv_file.name,
            "docker_stats_csv": docker_stats_file.name,
            "locust_log": log_file.name,
        },
    }
    metadata_path = Path(f"{prefix}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(run())
