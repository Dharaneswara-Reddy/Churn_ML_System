"""Store JSON payloads as JSONB on PostgreSQL.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-30

``events/db.py`` declares its payload columns as
``JSON().with_variant(JSONB, "postgresql")``, and the README describes the
PostgreSQL deployment as using JSONB — indexable, binary, type-checked at write.
The migrations, however, hardcoded ``sa.JSON()``, which PostgreSQL renders as the
plain ``json`` type.

Because the schema is created by Alembic and not by ``create_all`` in any real
deployment, the variant never took effect: a live PostgreSQL server had ``json``
columns, and Alembic's own autogenerate reported the two columns as drifted from
the models. This was invisible to compile-only checking, which asks the dialect
what it *would* emit for the model rather than what the database actually holds.

The practical consequences of ``json`` over ``jsonb``:

* No GIN indexing, so a containment query over ``features`` cannot be indexed.
* Text is re-parsed on every read instead of being stored pre-parsed.
* Duplicate keys and insignificant whitespace are preserved verbatim rather than
  normalised, so two logically identical payloads do not compare equal.

The cast is exact — every value already in the column is valid JSON, so
``USING payload::jsonb`` cannot fail — but it does rewrite both tables, which
takes an ACCESS EXCLUSIVE lock for the duration. On a large events table run this
during a maintenance window.

No-op on SQLite, which has no JSONB type; ``JSON`` there is already the only
representation, which is exactly what the variant expresses.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# (table, column) pairs declared as JSONType in events/db.py.
_JSON_COLUMNS = (
    ("prediction_events", "features"),
    ("outbox_events", "payload"),
)


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgresql():
        return

    for table, column in _JSON_COLUMNS:
        op.alter_column(
            table,
            column,
            type_=postgresql.JSONB(astext_type=None),
            existing_type=postgresql.JSON(astext_type=None),
            existing_nullable=False,
            postgresql_using=f"{column}::jsonb",
        )


def downgrade() -> None:
    if not _is_postgresql():
        return

    for table, column in _JSON_COLUMNS:
        op.alter_column(
            table,
            column,
            type_=postgresql.JSON(astext_type=None),
            existing_type=postgresql.JSONB(astext_type=None),
            existing_nullable=False,
            postgresql_using=f"{column}::json",
        )
