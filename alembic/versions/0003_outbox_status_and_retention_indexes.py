"""Add explicit outbox status, dead-letter reason, and retention indexes.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-27

Two problems this closes:

* An event that exhausted its retry budget sat at ``attempts == MAX_ATTEMPTS``
  with ``processed_at IS NULL`` — byte-for-byte indistinguishable from a normal
  pending row. A "pending events" query or metric silently counted permanently
  dead work as live backlog, and nothing surfaced the failure.
* The claim and retention queries filtered on columns with no supporting
  composite index, so their cost grew with the total historical row count rather
  than with the number of rows actually pending.

Existing rows are backfilled from ``processed_at`` so the new column is correct
for data written before this migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _indexes(table: str) -> set[str]:
    return {i["name"] for i in sa.inspect(op.get_bind()).get_indexes(table)}


def upgrade() -> None:
    outbox_columns = _columns("outbox_events")

    with op.batch_alter_table("outbox_events") as batch:
        if "status" not in outbox_columns:
            batch.add_column(
                sa.Column(
                    "status",
                    sa.String(16),
                    nullable=False,
                    server_default="PENDING",
                )
            )
        if "last_error" not in outbox_columns:
            batch.add_column(sa.Column("last_error", sa.String(512), nullable=True))

    # Backfill: rows written before this migration carry no status.
    if "status" not in outbox_columns:
        op.execute(
            "UPDATE outbox_events SET status = 'PROCESSED' "
            "WHERE processed_at IS NOT NULL"
        )
        # Exhausted-but-unprocessed rows are exactly the invisible dead letters
        # this migration exists to make visible.
        op.execute(
            "UPDATE outbox_events SET status = 'DEAD_LETTER' "
            "WHERE processed_at IS NULL AND attempts >= 5"
        )
        op.execute(
            "UPDATE outbox_events SET status = 'PENDING' "
            "WHERE processed_at IS NULL AND attempts < 5"
        )

    existing = _indexes("outbox_events")
    if "ix_outbox_events_status" not in existing:
        op.create_index("ix_outbox_events_status", "outbox_events", ["status"])
    if "ix_outbox_claim" not in existing:
        op.create_index(
            "ix_outbox_claim",
            "outbox_events",
            ["status", "attempts", "claimed_at", "created_at"],
        )
    if "ix_outbox_retention" not in existing:
        op.create_index(
            "ix_outbox_retention", "outbox_events", ["status", "processed_at"]
        )

    prediction_indexes = _indexes("prediction_events")
    if "ix_prediction_events_retention" not in prediction_indexes:
        op.create_index(
            "ix_prediction_events_retention",
            "prediction_events",
            ["created_at", "label"],
        )


def downgrade() -> None:
    for name, table in (
        ("ix_prediction_events_retention", "prediction_events"),
        ("ix_outbox_retention", "outbox_events"),
        ("ix_outbox_claim", "outbox_events"),
        ("ix_outbox_events_status", "outbox_events"),
    ):
        if name in _indexes(table):
            op.drop_index(name, table_name=table)

    with op.batch_alter_table("outbox_events") as batch:
        batch.drop_column("last_error")
        batch.drop_column("status")
