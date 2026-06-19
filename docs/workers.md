# Outbox Worker Package (`workers/`)

The `workers` package implements background execution runtimes and asynchronous consumers to decouple long-running operations from the main HTTP API request-response loop.

## Transactional Outbox Pattern

To ensure reliable communication with external services without using distributed two-phase commits, the system uses the **Transactional Outbox Pattern**.

1. When a prediction request is received, the API writes both the prediction result and an audit record to the `outbox_events` table **within a single database transaction**.
2. A separate background worker polls the `outbox_events` table, processes the events asynchronously, and flags them as processed.

```
+-------------+
|   API Pod   |
+-------------+
       | (Atomic SQL Transaction)
       v
 +---------------------------------------+
 | Database                              |
 |   - predictions                       |
 |   - outbox_events  <------------------+--+
 +---------------------------------------+  |
                                            | (SELECT FOR UPDATE
                                            |  SKIP LOCKED)
                                            v
                                     +---------------+
                                     | Outbox Worker |
                                     +---------------+
                                            | (Publish)
                                            v
                                     [ Message Broker ]
                                     (SNS / SQS / Kafka)
```

---

## File and Component Index

### `outbox_consumer.py`

Poller and processor daemon designed to run as a standalone process (or horizontally scaled containers).

#### Key Functions
*   **`run_worker()`**: Launches the polling loop. Registers signal handlers for `SIGTERM` and `SIGINT` to trigger a graceful shutdown sequence.
*   **`_claim_and_process_batch()`**: Queries the database for unprocessed events. Executes a multi-threaded batch processing loop via a `ThreadPoolExecutor`.
*   **`_process_single_event()`**: Handles payload delivery. In local environments, this logs the event. In production setups, this publishes to Amazon SNS, Amazon SQS, or Apache Kafka.

---

## Distributed Systems Coordination

### row-level locking (SKIP LOCKED)
When scaling horizontally, running multiple instances of the worker could cause double-processing of events. To prevent this without deploying a complex coordinator service (like ZooKeeper), the worker uses SQL row-level locking:

```sql
SELECT * FROM outbox_events 
WHERE processed_at IS NULL 
ORDER BY created_at ASC 
LIMIT 50 
FOR UPDATE SKIP LOCKED;
```

*   **`FOR UPDATE`**: Locks the selected rows so no other transaction can modify or lock them.
*   **`SKIP LOCKED`**: Instructs database engines (like PostgreSQL) to skip any rows already locked by other worker instances instead of blocking. This allows workers to run fully in parallel on disjoint chunks of work.

### Graceful Shutdown
To prevent data loss or partially processed tasks during deployments or scaling actions:
*   The worker listens for `SIGTERM` and `SIGINT` signals.
*   Upon receipt, it sets a threading `Event` and halts the polling loop.
*   The worker waits for all threads in the current `ThreadPoolExecutor` batch to complete processing (draining the current batch) before closing database connections and exiting.
