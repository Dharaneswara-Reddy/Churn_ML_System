"""
Migration mechanics that are backend-independent.

Live PostgreSQL behaviour (JSONB types, drift against the ORM) is covered in
``test_postgres_integration.py``; what is here holds on SQLite too.
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

import pytest


class TestAlembicDoesNotSilenceApplicationLogging:
    """
    ``alembic/env.py`` calls ``logging.config.fileConfig``, whose
    ``disable_existing_loggers`` parameter defaults to **True**. That is harmless
    for ``alembic upgrade`` on the command line, where nothing else is running.

    Run a migration *in-process* — a test, or a startup hook that migrates before
    serving — and it silently disables every logger that already existed. The
    application keeps working and simply stops logging, with no error anywhere. It
    was found because running the migration suite before the scheduler suite made
    a passing log-assertion fail.
    """

    def test_env_py_preserves_existing_loggers(self):
        source = Path("alembic/env.py").read_text(encoding="utf-8")

        assert "disable_existing_loggers=False" in source, (
            "alembic/env.py must pass disable_existing_loggers=False to fileConfig, "
            "or an in-process migration silently disables application logging."
        )

    def test_running_a_migration_leaves_a_logger_enabled(self, tmp_path, monkeypatch):
        """
        The behavioural version of the check above: create a logger, run a real
        migration in-process, and confirm the logger still emits.
        """
        from alembic import command
        from alembic.config import Config

        db_path = tmp_path / "migration_logging.db"
        monkeypatch.setenv("CHURN_EVENT_STORE_DATABASE_URL", f"sqlite:///{db_path}")

        canary = logging.getLogger("churn_system.canary.migration_logging")
        canary.setLevel(logging.INFO)
        canary.propagate = True

        command.upgrade(Config("alembic.ini"), "head")

        assert not canary.disabled, (
            "Running a migration in-process disabled an existing application "
            "logger — logging would silently stop for the rest of the process."
        )
        assert canary.isEnabledFor(logging.INFO)


class TestMigrationChain:
    def _revisions(self) -> dict[str, str | None]:
        chain: dict[str, str | None] = {}
        for path in sorted(Path("alembic/versions").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            revision = down = None
            for line in text.splitlines():
                if line.startswith("revision = "):
                    revision = line.split("=", 1)[1].strip().strip('"\'')
                elif line.startswith("down_revision = "):
                    raw = line.split("=", 1)[1].strip()
                    down = None if raw == "None" else raw.strip('"\'')
            if revision:
                chain[revision] = down
        return chain

    def test_there_is_exactly_one_head(self):
        """
        Two heads make ``alembic upgrade head`` ambiguous and it refuses to run —
        a deploy-time failure that no unit test would otherwise catch.
        """
        chain = self._revisions()
        parents = {down for down in chain.values() if down is not None}
        heads = [rev for rev in chain if rev not in parents]

        assert len(heads) == 1, f"Expected a single migration head, found {heads}"

    def test_there_is_exactly_one_base(self):
        chain = self._revisions()
        bases = [rev for rev, down in chain.items() if down is None]

        assert len(bases) == 1, f"Expected a single base revision, found {bases}"

    def test_every_parent_exists(self):
        chain = self._revisions()

        for revision, down in chain.items():
            if down is not None:
                assert down in chain, (
                    f"Migration {revision} declares down_revision={down!r}, which "
                    "does not exist — the chain is broken."
                )

    def test_every_migration_is_reversible(self):
        """
        A migration without a real ``downgrade`` cannot be rolled back, which turns
        a bad deploy into a restore-from-backup.
        """
        for path in sorted(Path("alembic/versions").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert "def downgrade()" in text, f"{path.name} has no downgrade()"
            body = text.split("def downgrade()", 1)[1]
            assert "pass" not in body.split("\n")[1:3], (
                f"{path.name} has an empty downgrade()"
            )


class TestAlembicConfiguration:
    def test_the_url_comes_from_application_config_not_the_ini(self):
        """
        If alembic.ini carried its own URL, ``alembic upgrade head`` could migrate
        a different database from the one the service reads and writes — and
        nothing would report the mismatch.
        """
        parser = configparser.ConfigParser()
        parser.read("alembic.ini")

        ini_url = parser.get("alembic", "sqlalchemy.url", fallback="").strip()
        assert ini_url in ("", "driver://user:pass@localhost/dbname"), (
            f"alembic.ini pins sqlalchemy.url to {ini_url!r}; env.py must be the "
            "single source of the database URL."
        )
        assert "sqlalchemy.url" in Path("alembic/env.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("revision", ["0001", "0002", "0003", "0004"])
def test_expected_revisions_are_present(revision):
    """Guards against a migration being deleted rather than superseded."""
    files = list(Path("alembic/versions").glob(f"{revision}_*.py"))

    assert files, f"Migration {revision} is missing from alembic/versions/"
