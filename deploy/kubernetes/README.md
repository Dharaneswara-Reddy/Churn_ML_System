# Kubernetes deployment

Applies in order:

```bash
kubectl apply -f namespace.yaml
kubectl create secret generic churn-secrets -n churn \
  --from-literal=api-key="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  --from-literal=admin-api-key="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  --from-literal=artifact-signing-key="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  --from-literal=subject-key-salt="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  --from-literal=database-url="postgresql+psycopg://churn:CHANGEME@postgres:5432/churn_events"
kubectl apply -f configmap.yaml -f migrate-job.yaml
kubectl apply -f api-deployment.yaml -f api-service.yaml -f api-hpa.yaml -f api-pdb.yaml
kubectl apply -f worker-deployment.yaml -f scheduler-deployment.yaml
```

## What is and is not horizontally scalable

| Component | Replicas | Why |
|---|---|---|
| `api` | 2 – 10, HPA on CPU | Stateless. The only per-process state is the model bundle, which is read-only and identical on every replica. |
| `worker` | N | Outbox claiming is a committed compare-and-set with a lease. Verified against a real PostgreSQL server under 8-way contention in `tests/test_postgres_integration.py`. |
| `scheduler` | **1** | Not stateless: it retrains, rewrites the shared drift baseline and promotes. A second replica is *safe* — the PostgreSQL advisory lock makes it stand by and exit — but does no work. `replicas: 1` with `strategy: Recreate` is deliberate. |
| `postgres` | 1 (or managed) | Single writer. Use a managed service in production; the manifest here is for evaluation. |

## Probes

`/health` is a liveness probe: process is up. `/ready` is a readiness probe and runs
a **real one-row prediction** through the full serving path, so a replica with a
missing, unsigned or unloadable bundle is pulled out of the Service rather than
returning 500s to clients.

Because `/ready` does real work it is given a longer `periodSeconds` than
`/health`. Never point liveness at `/ready`: a model that fails to load would then
restart the pod in a loop instead of simply leaving it out of rotation.

## The model bundle

`api` mounts the bundle **read-only**. That is the other half of signature
verification: the API process should not be able to write the artifact it is about
to unpickle.

`ReadWriteMany` is required because the scheduler writes the bundle while several
API replicas read it. If your cluster has no RWX storage class, the alternative is
to bake the model into the image and promote by rolling a new image — slower, but
it removes the shared volume entirely.
