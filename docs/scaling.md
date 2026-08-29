# Scaling and Performance

> Every number on this page was measured on this repository with
> [`scripts/load_test.py`](../scripts/load_test.py) and the benchmarks described
> below. Nothing here is estimated. Where a number is environment-specific, the
> environment is stated.

**Measurement environment**: 12-core x86-64 Linux, Python 3.10, one host, load
generator running on the same machine as the server (so it competes for the same
CPUs — see [Reading the numbers](#reading-the-numbers)).

**Model under test**: RandomForestClassifier, 150 trees, `max_depth=8`,
`n_jobs=None`, 19 features.

---

## Where the time goes

Profiled in-process, 200 iterations, single row:

| Stage | Median |
|---|---:|
| `pd.DataFrame([row])` | 0.49 ms |
| `build_features` | 1.24 ms |
| `+ validate_inference_data` | 1.61 ms |
| `predict_proba` | **10.63 ms** |
| **Full serving path** | **12.80 ms** |

`predict_proba` is 81% of the work. Feature building and validation together cost
less than 2 ms, so optimising them cannot move the number meaningfully — the model
is the cost.

Over HTTP the same request costs ~26 ms p50, so roughly 13 ms is HTTP handling,
ASGI dispatch and the event-write hand-off.

## Single-prediction capacity

`POST /predict`, 8-second phases after a warm-up, client-observed latency:

| uvicorn workers | Peak throughput | p50 @ c=1 | p50 @ c=32 | p95 @ c=32 |
|---:|---:|---:|---:|---:|
| 1 | 32 rps | 25.7 ms | 959 ms | 1669 ms |
| 2 | 52 rps | 28.0 ms | 645 ms | 955 ms |
| 4 | 81 rps | 29.7 ms | 380 ms | 710 ms |
| 8 | **110 rps** | 30.1 ms | 266 ms | 504 ms |

Two things this table establishes:

**A single process is capped at roughly `1 / latency`.** 32 rps against a 12.8 ms
inference is not a coincidence — it is one prediction at a time. `predict_proba`
holds the GIL for the traversal, so `asyncio.to_thread` moves the work off the
event loop (which is what keeps `/health` responsive under load) but does not make
two predictions run at once. **Concurrency does not increase throughput on one
process; it only increases queueing**, which is exactly what the latency column
shows: p50 grows linearly with concurrency while rps stays flat.

**Capacity scales with processes, not threads.** Add uvicorn workers, or add
replicas — see [deploy/kubernetes](../deploy/kubernetes/).

## Batch capacity

`POST /predict/batch` amortises the fixed per-call cost across rows, and is by far
the cheapest way to move volume:

| Batch size | Total | Per row | Rows/s |
|---:|---:|---:|---:|
| 1 | 29.4 ms | 29.35 ms | 34 |
| 10 | 39.2 ms | 3.92 ms | 255 |
| 50 | 72.2 ms | 1.44 ms | 693 |
| 100 | 55.0 ms | 0.55 ms | **1819** |

A 100-row batch is **~53x** the per-row throughput of 100 single requests on the
same single worker. If a caller has more than one row, batching is the single
biggest lever available.

### Why chunking was removed

`/predict/batch` used to split the batch into `BATCH_CHUNK_SIZE` chunks (default
25) and fan them out with `asyncio.gather` + `asyncio.to_thread`, documented as
achieving parallelism "while one chunk is waiting on GIL release". Measured on a
100-row batch, one worker, 25 reps with outliers trimmed:

Two independent runs, reported as ranges because run-to-run variance is about 10%
and never reorders the rows:

| `CHURN_BATCH_CHUNK_SIZE` | p50 | Rows/s |
|---:|---:|---:|
| 10 | 266–279 ms | 358–376 |
| 25 *(old default)* | 128–129 ms | 775–780 |
| 50 | 77–88 ms | 1136–1299 |
| 100 *(new default — one chunk)* | **54–55 ms** | **1819–1847** |

Monotonic in both runs: every split costs throughput. The premise was wrong — `predict_proba`
holds the GIL for the entire call, so the chunks never overlap, while each chunk
repays the full fixed cost of DataFrame construction, validation and reindexing.
The default is now the whole batch. The knob remains as a **memory** bound for
deployments that raise `CHURN_MAX_BATCH_SIZE` well beyond 100; it is not a
throughput control.

## Tuning knobs, with measured guidance

| Setting | Default | What it actually does |
|---|---|---|
| uvicorn `--workers` | 1 | The main throughput lever. ~32 rps per worker; budget one core each and leave headroom for the scheduler and worker containers. |
| `CHURN_BATCH_CHUNK_SIZE` | `= CHURN_MAX_BATCH_SIZE` | Memory bound per inference call. Lowering it *reduces* throughput (table above). |
| `CHURN_MAX_BATCH_SIZE` | 100 | Request ceiling. Raising it raises peak memory per call, since the default chunk size follows it. |
| `pool_size` / `max_overflow` | 10 / 5 | Per-process PostgreSQL connections. Budget **15 per API replica**, plus workers, plus the scheduler, against the server's `max_connections`. |
| `EVENT_WRITER_WORKERS` | 4 | Threads draining fire-and-forget event writes. Event writing is I/O-bound, so these genuinely do overlap — unlike inference threads. |
| `CHURN_OUTBOX_LEASE_SECONDS` | `max(6 x poll, 60)` | How long a crashed worker's claimed rows stay stuck before another worker retries them. |

## What the event store costs

Measured at 4 workers, concurrency 8–32:

| Event store | Peak throughput |
|---|---:|
| SQLite | 73 rps |
| PostgreSQL | 68 rps |

The event store is **not** the serving bottleneck at this scale — PostgreSQL is
marginally slower purely from network round-trips. Move to PostgreSQL for
correctness (multiple writers, multiple hosts, cluster-wide leader election), not
for speed.

## Horizontal scaling

| Component | Replicas | Why |
|---|---|---|
| `api` | 2–10 (HPA on CPU) | Stateless. Per-process state is the read-only model bundle, identical everywhere. |
| `worker` | N | Outbox claiming is a committed compare-and-set with a lease, verified against a real PostgreSQL server under 8-way contention in [`tests/test_postgres_integration.py`](../tests/test_postgres_integration.py). |
| `scheduler` | **1** | Retrains, rewrites the shared drift baseline, promotes. The advisory lock makes a second replica *safe* (it stands by and exits), not useful. |
| `postgres` | 1, or managed | Single writer. |

Two hard prerequisites for more than one API replica:

1. **PostgreSQL, not SQLite.** SQLite is one writer and one file. Two replicas on
   one host give `database is locked`; on two hosts they are simply two different
   databases.
2. **A shared model volume.** Every replica must serve the same champion, or
   identical requests get different answers depending on which replica the load
   balancer picked. Promotion reload is posted to the *headless* Service so every
   replica reloads — through the load-balanced Service exactly one would.

See [`docker-compose.scale.yml`](../docker-compose.scale.yml) and
[`deploy/kubernetes/`](../deploy/kubernetes/).

## Reading the numbers

* The load generator runs on the same 12 cores as the server, so it competes for
  CPU. Worker scaling is therefore **sublinear here** (8 workers gives 3.4x, not
  8x). On separate hosts, expect closer to linear until the event store or the
  network saturates.
* Latencies are **client-observed end-to-end**, not the `latency_seconds` the API
  reports for its own inference step. The former is what a caller experiences.
* Rate limiting is disabled during measurement (`CHURN_DISABLE_RATE_LIMIT=1`).
  With it enabled you are measuring slowapi, not the model.
* Every phase is preceded by a 1-second warm-up so the first phase does not absorb
  the cold model load.

Reproduce with:

```bash
.venv/bin/python scripts/load_test.py --duration 8 --concurrency 1 8 32 --workers 4
```

## Known ceilings

* **~110 rps single-prediction on one 12-core host.** Beyond that, add replicas.
* **`predict_proba` is 81% of serving cost.** A smaller forest is the only lever
  that moves it much; 150 trees at depth 8 was chosen for PR-AUC, and cutting it
  is a model-quality decision, not a performance one.
* **`n_jobs=None` on the classifier is deliberate.** Setting `n_jobs>1` would make
  each *individual* prediction use several cores, which helps a lone request and
  hurts under concurrency by oversubscribing every worker at once.
