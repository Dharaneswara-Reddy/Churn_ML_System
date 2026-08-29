"""
Alembic environment.

Pulls the database URL from the application's own configuration rather than
alembic.ini, so `alembic upgrade head` can never migrate a different database from
the one the service reads and writes.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from churn_system.config.config import load_config
from churn_system.events.db import Base

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers defaults to True, which silences every logger that
    # already exists in the process. That is harmless for `alembic upgrade` on the
    # command line, where nothing else is running — but running a migration
    # in-process (a test, or a startup hook that migrates before serving) would
    # then silently disable the application's own loggers for the rest of that
    # process's life. The failure mode is that logging simply stops, with no error.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    cfg = load_config()
    return str(
        cfg.get("event_store", {}).get("database_url", "sqlite:///./data/churn_events.db")
    )


config.set_main_option("sqlalchemy.url", _database_url())


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER columns in place; batch mode recreates the table.
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
