---
type: Service
title: Outbox Worker
description: Asynchronous outbox event consumer with distributed concurrency safety via SQL row-level locking and graceful shutdown.
tags: [workers, outbox-pattern, distributed-systems, async, concurrency]
timestamp: 2026-06-30T00:00:00Z
---

The outbox worker is a background daemon that polls unprocessed events from the [event store](events.md) and delivers them to external systems. It implements the consumer side of the Transactional Outbox Pattern, decoupling long-running operations from the [API](api.md) request-response loop.

# Transactional Outbox Pattern

1. When a prediction request is received, the API writes both the prediction result and an outbox record to the database in a single atomic transaction (see [event store](events.md)).
2. The outbox worker polls the `outbox_events` table, processes events asynchronously, and marks them as processed.

This ensures reliable event delivery without distributed two-phase commits. Future integration with message brokers (Kafka, SQS, SNS) requires no changes to the API code — only the worker's delivery function changes.

# Processing Flow

- **`run_worker()`** — launches the polling loop and registers signal handlers for `SIGTERM`/`SIGINT`.
- **`_claim_and_process_batch()`** — queries unprocessed events and executes multi-threaded batch processing via a `ThreadPoolExecutor`.
- **`_process_single_event()`** — handles payload delivery. In local environments, this logs the event. In production, this publishes to Amazon SNS, Amazon SQS, or Apache Kafka.

# Distributed Concurrency Safety

When scaling horizontally (multiple worker instances), double-processing must be prevented. The worker uses SQL row-level locking:

```sql
SELECT * FROM outbox_events
WHERE processed_at IS NULL
ORDER BY created_at ASC
LIMIT 50
FOR UPDATE SKIP LOCKED;
```

- **`FOR UPDATE`** — locks the selected rows so no other transaction can modify or lock them.
- **`SKIP LOCKED`** — instructs the database (PostgreSQL) to skip rows already locked by other worker instances instead of blocking. This allows workers to run fully in parallel on disjoint chunks of work.

# Graceful Shutdown

To prevent data loss during deployments or scaling:

1. The worker listens for `SIGTERM` and `SIGINT` signals.
2. Upon receipt, it sets a threading `Event` and halts the polling loop.
3. The worker waits for all threads in the current `ThreadPoolExecutor` batch to finish (draining the current batch) before closing database connections and exiting.
