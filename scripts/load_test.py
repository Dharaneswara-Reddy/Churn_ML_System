"""
Measure API throughput and latency under concurrent load.

Why this exists
---------------
The README previously carried latency claims that nothing had measured. Numbers
that nobody produced are worse than no numbers: they get quoted in capacity
planning, and the first time anyone checks is during an incident.

This drives a **real uvicorn process over real HTTP** rather than
``TestClient``. TestClient runs the app in-process on the caller's event loop, so
it measures handler time and hides everything capacity actually depends on:
connection handling, the ASGI server's own scheduling, and the GIL contention
between request parsing and the ``asyncio.to_thread`` inference calls.

Usage
-----
    # start a server yourself, or let the script do it:
    .venv/bin/python scripts/load_test.py --duration 20 --concurrency 16

    # against an already-running instance:
    .venv/bin/python scripts/load_test.py --url http://localhost:8000 --no-spawn

Reported percentiles are of *end-to-end client-observed* latency, which is what a
caller experiences — not the ``latency_seconds`` the API reports for its own
inference step.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_payload(base_url: str) -> dict:
    """
    Construct a valid request from the live OpenAPI document.

    Hardcoding a payload would silently stop exercising the model the moment the
    champion's feature schema changed — the load test would then be measuring the
    422 path at full speed and reporting it as throughput.
    """
    with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=10) as response:  # noqa: S310
        spec = json.load(response)

    schema = spec["components"]["schemas"]["DynamicPredictionRequest"]
    required = schema.get("required", list(schema["properties"]))

    sample = {
        "Tenure Months": 2,
        "Monthly Charges": 70.7,
        "Total Charges": 151.65,
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Payment Method": "Electronic check",
        "Gender": "Male",
    }

    row = {}
    for name in required:
        if name in sample:
            row[name] = sample[name]
            continue
        kind = schema["properties"][name].get("type", "string")
        row[name] = {"integer": 1, "number": 1.0, "boolean": False}.get(kind, "No")
    return row


def _post(url: str, body: bytes, api_key: str | None) -> tuple[float, int]:
    request = urllib.request.Request(url, data=body, method="POST")  # noqa: S310
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("X-API-Key", api_key)

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    except Exception:
        status = 0
    return time.perf_counter() - started, status


def run_phase(
    base_url: str, payload: dict, concurrency: int, duration: float, api_key: str | None
) -> dict:
    body = json.dumps(payload).encode()
    url = f"{base_url}/predict"
    deadline = time.perf_counter() + duration

    latencies: list[float] = []
    statuses: dict[int, int] = {}

    def worker() -> tuple[list[float], dict[int, int]]:
        mine: list[float] = []
        codes: dict[int, int] = {}
        while time.perf_counter() < deadline:
            elapsed, status = _post(url, body, api_key)
            mine.append(elapsed)
            codes[status] = codes.get(status, 0) + 1
        return mine, codes

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for mine, codes in pool.map(lambda _: worker(), range(concurrency)):
            latencies.extend(mine)
            for code, count in codes.items():
                statuses[code] = statuses.get(code, 0) + count
    wall = time.perf_counter() - wall_start

    ok = statuses.get(200, 0)
    ordered = sorted(latencies)

    def pct(p: float) -> float:
        if not ordered:
            return float("nan")
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "successful": ok,
        "statuses": statuses,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(latencies) / wall, 1) if wall else 0.0,
        "mean_ms": round(statistics.mean(ordered) * 1000, 2) if ordered else None,
        "p50_ms": round(pct(0.50) * 1000, 2),
        "p95_ms": round(pct(0.95) * 1000, 2),
        "p99_ms": round(pct(0.99) * 1000, 2),
        "max_ms": round(ordered[-1] * 1000, 2) if ordered else None,
    }


def wait_for_ready(base_url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8111")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="Concurrency levels to sweep; one phase each.",
    )
    parser.add_argument("--workers", type=int, default=1, help="uvicorn worker processes")
    parser.add_argument("--no-spawn", action="store_true", help="use an already-running server")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("CHURN_API_KEY") or None
    server = None

    if not args.no_spawn:
        port = args.url.rsplit(":", 1)[-1]
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "src"),
            # Rate limiting would make this a measurement of slowapi, not the model.
            "CHURN_DISABLE_RATE_LIMIT": "1",
            "CHURN_MLFLOW_ENABLED": "0",
        }
        env.setdefault("CHURN_ALLOW_ANONYMOUS", "1")
        server = subprocess.Popen(  # noqa: S603
            [
                sys.executable, "-m", "uvicorn", "churn_system.api.api:app",
                "--host", "127.0.0.1", "--port", port,
                "--workers", str(args.workers), "--log-level", "warning",
            ],
            cwd=REPO_ROOT,
            env=env,
        )

    try:
        if not wait_for_ready(args.url):
            print("Server did not become healthy in time.", file=sys.stderr)
            return 1

        payload = build_payload(args.url)
        print(f"Payload: {len(payload)} fields — {sorted(payload)[:4]}...\n")

        results = []
        for level in args.concurrency:
            # A short warm-up so the first phase does not absorb the cold model
            # load and JIT-less import costs that every later phase avoids.
            run_phase(args.url, payload, level, 1.0, api_key)
            result = run_phase(args.url, payload, level, args.duration, api_key)
            results.append(result)
            print(
                f"concurrency={result['concurrency']:>3}  "
                f"rps={result['throughput_rps']:>8.1f}  "
                f"p50={result['p50_ms']:>7.2f}ms  "
                f"p95={result['p95_ms']:>7.2f}ms  "
                f"p99={result['p99_ms']:>7.2f}ms  "
                f"ok={result['successful']}/{result['requests']}"
            )

        report = {
            "uvicorn_workers": args.workers,
            "duration_seconds_per_phase": args.duration,
            "cpu_count": os.cpu_count(),
            "phases": results,
        }
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2))
            print(f"\nWrote {args.json_out}")
        return 0
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
