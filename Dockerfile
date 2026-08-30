# API service — mount ./models at runtime with production artifacts
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # Which peers may set X-Forwarded-For. Deliberately NOT "*": rate limiting keys
    # on the client address, so trusting every peer would let any caller forge an
    # address and get an unlimited bucket. Override with your ingress CIDR.
    FORWARDED_ALLOW_IPS=127.0.0.1

# Pick up base-image security patches published since the tag was built, then drop
# the apt lists so they do not ship in the layer.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. Running as root next to a writable ./data bind mount
# means any RCE in the web stack lands with full access to the customer dataset.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

# The postgres extra is included because SQLite cannot back more than one
# process writing concurrently, and this image runs alongside the outbox worker.
# Without it the container starts and then fails at import with a missing
# psycopg the moment CHURN_EVENT_STORE_DATABASE_URL names PostgreSQL.
RUN pip install --no-cache-dir ".[postgres]" \
    && chown -R appuser:appuser /app

# Create the mount points inside the image, owned by the runtime user.
#
# Docker copies the image's ownership and contents into a *named* volume the
# first time it is mounted — but only if the path already exists in the image.
# Without these, the volume is created root-owned, the unprivileged process
# cannot open its own log file, and the container crash-loops on
# PermissionError before it ever binds a port.
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R appuser:appuser /app/logs /app/data /app/models

USER appuser

EXPOSE 8000

# Baked into the image so the check travels with it — a compose-only healthcheck is
# lost under `docker run` and under Kubernetes. Targets /ready, not /health:
# /health is deliberately always-200 while the process lives, so using it would
# mark an instance with a missing or corrupt model.pkl as healthy and let
# dependent services start against it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "churn_system.api.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers"]
