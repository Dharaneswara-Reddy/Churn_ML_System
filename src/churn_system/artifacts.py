"""Helpers for model artifact paths, bundle validation, and bundle signing.

A deployable model bundle is a directory containing ``model.pkl`` + ``metadata.json``.
Because promotion automatically triggers a hot-reload of the serving process (and a
compromised training job can write into that same directory tree — see
docker-compose.yml, where ``train``/``scheduler`` mount ``./models`` read-write while
``api`` mounts it read-only), the bundle's integrity and provenance must be verifiable
*before* anything unpickles ``model.pkl``. ``sign_model_bundle`` / ``verify_bundle_signature``
provide that: an HMAC-SHA256 over a SHA-256 digest of the model bytes and the canonical
metadata, keyed by ``CHURN_ARTIFACT_SIGNING_KEY``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from churn_system.config.config import CONFIG
from churn_system.logging.logger import get_logger

logger = get_logger(__name__, CONFIG["logging"]["lifecycle"])

VERSION_PATTERN = re.compile(r"^churn_model_\d{8}_\d{6}$")

# --- signing -------------------------------------------------------------------

SIGNATURE_FILENAME = "signature.json"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
SIGNING_KEY_ENV = "CHURN_ARTIFACT_SIGNING_KEY"
# Explicit, mirrors the CHURN_ALLOW_ANONYMOUS pattern used for API auth: absent this,
# an unset signing key must refuse verification rather than silently pass.
ALLOW_UNSIGNED_ENV = "CHURN_ALLOW_UNSIGNED_ARTIFACTS"


class ArtifactSignatureError(Exception):
    """Raised when a model bundle's signature is missing, malformed, or does not verify."""


def _signature_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / SIGNATURE_FILENAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_metadata_bytes(metadata_path: Path) -> bytes:
    """
    Canonical serialisation of metadata.json for hashing.

    Sorted keys and fixed separators so re-formatting metadata.json (whitespace,
    key order) never changes the digest — only the data does.
    """
    data = load_metadata(metadata_path)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signing_key() -> bytes | None:
    key = os.environ.get(SIGNING_KEY_ENV)
    if key is None or not key.strip():
        return None
    return key.encode("utf-8")


def _unsigned_allowed() -> bool:
    return os.environ.get(ALLOW_UNSIGNED_ENV, "").strip() == "1"


def compute_bundle_digest(bundle_dir: Path, *, model_filename: str = "model.pkl") -> dict[str, str]:
    """Return the SHA-256 hex digests of the model file and the canonical metadata."""
    bundle_dir = Path(bundle_dir)
    model_path = bundle_dir / model_filename
    metadata_path = bundle_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    return {
        "model_sha256": _sha256_file(model_path),
        "metadata_sha256": hashlib.sha256(_canonical_metadata_bytes(metadata_path)).hexdigest(),
    }


def _compute_and_write_signature(bundle_dir: Path, *, model_filename: str = "model.pkl") -> Path:
    """
    Compute and write the HMAC-SHA256 signature file for a bundle.

    Raises ``RuntimeError`` when no signing key is configured — signing is a
    deliberate act and must never silently no-op.
    """
    key = _signing_key()
    if key is None:
        raise RuntimeError(
            f"{SIGNING_KEY_ENV} is not set — cannot sign model bundle at {bundle_dir}."
        )

    bundle_dir = Path(bundle_dir)
    digest = compute_bundle_digest(bundle_dir, model_filename=model_filename)
    message = f"{digest['model_sha256']}:{digest['metadata_sha256']}".encode()
    signature = hmac.new(key, message, hashlib.sha256).hexdigest()

    payload = {
        "algorithm": SIGNATURE_ALGORITHM,
        "model_filename": model_filename,
        "model_sha256": digest["model_sha256"],
        "metadata_sha256": digest["metadata_sha256"],
        "signature": signature,
    }
    sig_path = _signature_path(bundle_dir)
    sig_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sig_path


def sign_model_bundle(bundle_dir: Path, *, model_filename: str = "model.pkl") -> Path:
    """
    Sign a model bundle in place, writing ``signature.json`` next to ``model.pkl``.

    Requires ``CHURN_ARTIFACT_SIGNING_KEY``. Called at promotion time (see
    ``lifecycle/promote.py`` and ``lifecycle/rollback.py``) so the signature always
    reflects the bytes that actually landed in the serving directory, independent of
    whether the training pipeline that produced the experiment signed anything.
    """
    return _compute_and_write_signature(Path(bundle_dir), model_filename=model_filename)


def verify_bundle_signature(bundle_dir: Path) -> None:
    """
    Verify a model bundle's HMAC-SHA256 signature. Raises on any failure, returns
    ``None`` on success.

    This is the gate the serving layer calls before ``pickle.load``-ing ``model.pkl``
    (see ``serving/model_registry.py``). It fails closed: with no signing key
    configured, verification refuses rather than silently passing, unless the
    explicit opt-out ``CHURN_ALLOW_UNSIGNED_ARTIFACTS=1`` is set (mirrors
    ``CHURN_ALLOW_ANONYMOUS`` for API auth). That opt-out only covers the "no key
    configured" case — once a key is configured, a bundle with a missing or invalid
    signature is always rejected.
    """
    bundle_dir = Path(bundle_dir)
    key = _signing_key()

    if key is None:
        if _unsigned_allowed():
            logger.warning(
                "Signature verification skipped for %s (%s unset, %s=1).",
                bundle_dir,
                SIGNING_KEY_ENV,
                ALLOW_UNSIGNED_ENV,
            )
            return
        raise ArtifactSignatureError(
            f"{SIGNING_KEY_ENV} is not set and {ALLOW_UNSIGNED_ENV} is not '1' — "
            f"refusing to load an unverified model bundle: {bundle_dir}"
        )

    sig_path = _signature_path(bundle_dir)
    if not sig_path.exists():
        raise ArtifactSignatureError(f"Missing signature file for bundle: {sig_path}")

    try:
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactSignatureError(f"Could not read signature file: {sig_path}") from exc

    if not isinstance(payload, dict):
        raise ArtifactSignatureError(f"Signature file must be a JSON object: {sig_path}")

    expected_model_hash = payload.get("model_sha256")
    expected_metadata_hash = payload.get("metadata_sha256")
    expected_signature = payload.get("signature")
    model_filename = payload.get("model_filename", "model.pkl")

    if not (
        isinstance(expected_model_hash, str)
        and isinstance(expected_metadata_hash, str)
        and isinstance(expected_signature, str)
        and expected_model_hash
        and expected_metadata_hash
        and expected_signature
    ):
        raise ArtifactSignatureError(f"Signature file is malformed: {sig_path}")

    try:
        digest = compute_bundle_digest(bundle_dir, model_filename=model_filename)
    except FileNotFoundError as exc:
        raise ArtifactSignatureError(str(exc)) from exc

    if not hmac.compare_digest(digest["model_sha256"], expected_model_hash):
        raise ArtifactSignatureError(
            f"Model checksum mismatch — {model_filename} was modified after signing: {bundle_dir}"
        )
    if not hmac.compare_digest(digest["metadata_sha256"], expected_metadata_hash):
        raise ArtifactSignatureError(
            f"Metadata checksum mismatch — metadata.json was modified after signing: {bundle_dir}"
        )

    message = f"{expected_model_hash}:{expected_metadata_hash}".encode()
    recomputed_signature = hmac.new(key, message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(recomputed_signature, expected_signature):
        raise ArtifactSignatureError(
            f"Signature verification failed (wrong signing key or tampered signature): {bundle_dir}"
        )


# --- paths -----------------------------------------------------------------


def _cfg(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if config is not None else CONFIG


def production_model_path(config: dict[str, Any] | None = None) -> Path:
    return Path(_cfg(config)["paths"]["production_model"])


def production_model_dir(config: dict[str, Any] | None = None) -> Path:
    return production_model_path(config).parent


def production_metadata_path(config: dict[str, Any] | None = None) -> Path:
    return production_model_dir(config) / "metadata.json"


def experiments_dir(config: dict[str, Any] | None = None) -> Path:
    return Path(_cfg(config)["paths"]["experiments_dir"])


def experiment_dir(version: str, config: dict[str, Any] | None = None) -> Path:
    return experiments_dir(config) / version


def latest_experiment_dir(config: dict[str, Any] | None = None) -> Path | None:
    """
    Return the newest complete experiment bundle, or None when there is none.

    Versions are named ``churn_model_YYYYMMDD_HHMMSS``, so a lexicographic sort is
    chronological. Directories without a metadata.json are half-written and skipped.
    """
    root = experiments_dir(config)
    if not root.is_dir():
        return None
    versions = sorted(
        d
        for d in root.glob("churn_model_*")
        if d.is_dir() and VERSION_PATTERN.match(d.name) and (d / "metadata.json").exists()
    )
    return versions[-1] if versions else None


# --- concurrency-safe bundle swap --------------------------------------------

_SWAP_THREAD_LOCKS: dict[str, threading.Lock] = {}
_SWAP_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(target: Path) -> threading.Lock:
    """
    A per-target ``threading.Lock``, so two threads in this process serialize before
    either touches the filesystem lock at all.

    Keyed by the resolved parent directory plus the bundle name, not by an object
    identity, so unrelated targets never contend and the same logical target always
    maps to the same lock regardless of how many ``Path`` instances refer to it.
    """
    key = str(target.parent.resolve()) + "/" + target.name
    with _SWAP_THREAD_LOCKS_GUARD:
        return _SWAP_THREAD_LOCKS.setdefault(key, threading.Lock())


@contextlib.contextmanager
def _bundle_swap_lock(target: Path):
    """
    Serialize the whole promote/rollback critical section for ``target``.

    Two locks, nested: a ``threading.Lock`` for fast in-process exclusion, and an
    ``fcntl.flock`` on a sentinel file in ``target.parent`` for exclusion across
    separate processes (e.g. a scheduler and a manual promotion running at once).
    flock is per open-file-description, so distinct opens — even from the same
    process — correctly block each other; it is also released automatically by the
    kernel if the holding process dies, so a crashed swap never leaves the lock
    stuck.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock_for(target)
    lock_path = target.parent / f".{target.name}.lock"

    with thread_lock, open(lock_path, "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def swap_model_bundle(source: Path, target: Path, *, sign: bool = False) -> None:
    """
    Replace the bundle at ``target`` with a copy of ``source``, atomically.

    The copy lands in a uniquely-named sibling staging directory first (created via
    ``tempfile.mkdtemp``, so two concurrent swaps never collide on a fixed name), and
    the whole operation runs inside ``_bundle_swap_lock`` so concurrent
    promote/rollback calls for the same ``target`` are fully serialized rather than
    racing each other's staging/retired directories. Only two directory renames touch
    ``target``, and the outgoing bundle is retained until the new one is in place — a
    crash mid-swap therefore leaves a complete bundle at either ``target`` or its
    retired sibling, never a half-copied one. Any temp directories left behind by a
    failed attempt are removed before returning.

    When ``sign`` is True, a fresh HMAC signature is computed for the bytes that
    actually landed at ``target`` (see ``sign_model_bundle``) before the lock is
    released — so no other swap can observe a partially-signed bundle, and a bundle
    that fails to sign (no signing key, and the unsigned opt-out is not set) never
    goes live: the previous bundle is restored and the failure is raised.
    """
    source = Path(source)
    target = Path(target)

    if not source.is_dir():
        raise FileNotFoundError(f"Model bundle source not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)

    with _bundle_swap_lock(target):
        staging = Path(
            tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.incoming-")
        )
        retired: Path | None = None
        try:
            # mkdtemp already created the directory; copytree requires the
            # destination to not exist yet.
            staging.rmdir()
            shutil.copytree(source, staging)

            if target.exists():
                retired = Path(
                    tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.retired-")
                )
                retired.rmdir()
                target.rename(retired)

            try:
                staging.rename(target)
            except OSError:
                # Put the outgoing bundle back rather than leaving production empty.
                if retired is not None and retired.exists() and not target.exists():
                    retired.rename(target)
                raise

            if sign:
                try:
                    _compute_and_write_signature(target)
                except RuntimeError:
                    if _unsigned_allowed():
                        logger.warning(
                            "Bundle swapped into %s left unsigned (%s not set, "
                            "%s=1).",
                            target,
                            SIGNING_KEY_ENV,
                            ALLOW_UNSIGNED_ENV,
                        )
                    else:
                        # Fail closed: never leave an unsigned bundle live. Restore
                        # whatever was serving before this swap, or remove the
                        # target entirely when there was nothing to restore.
                        if retired is not None and retired.exists():
                            shutil.rmtree(target, ignore_errors=True)
                            retired.rename(target)
                        else:
                            shutil.rmtree(target, ignore_errors=True)
                        raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        shutil.rmtree(retired, ignore_errors=True) if retired is not None else None


def metadata_path_for_model(model_path: Path) -> Path:
    return model_path.parent / "metadata.json"


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    return data


def validate_model_bundle(
    model_path: Path,
    *,
    metadata_path: Path | None = None,
    require_model: bool = True,
) -> dict[str, Any]:
    """
    Validate the serving contract for a model artifact bundle.

    A deployable bundle is a model pickle plus sibling metadata.json. Metadata
    must carry a non-empty, ordered feature schema because serving depends on it.
    Metadata alone is never sufficient, though: when a model file is required, its
    signature is verified first (see ``verify_bundle_signature``) so a tampered or
    unsigned bundle is rejected before anything downstream trusts its metadata.
    """

    if require_model and not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    resolved_metadata_path = metadata_path or metadata_path_for_model(model_path)
    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {resolved_metadata_path}")

    if require_model:
        verify_bundle_signature(model_path.parent)

    metadata = load_metadata(resolved_metadata_path)
    feature_schema = metadata.get("feature_schema")
    if not isinstance(feature_schema, list) or not feature_schema:
        raise ValueError("metadata.json must contain a non-empty feature_schema list")
    if not all(isinstance(feature, str) and feature for feature in feature_schema):
        raise ValueError("feature_schema entries must be non-empty strings")

    feature_count = metadata.get("feature_count")
    if feature_count is not None and int(feature_count) != len(feature_schema):
        raise ValueError(
            "metadata feature_count does not match feature_schema length "
            f"({feature_count} != {len(feature_schema)})"
        )

    metrics = metadata.get("metrics", {})
    if metrics is not None and not isinstance(metrics, dict):
        raise ValueError("metadata metrics must be an object when present")

    return metadata
