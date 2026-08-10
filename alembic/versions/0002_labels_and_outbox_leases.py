"""Add ground-truth labels, subject keys, and outbox lease columns.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-10

Two capabilities that could not be added by ``create_all`` — it only creates
missing tables and never alters existing ones:

* ``prediction_events.label`` / ``labeled_at`` / ``subject_key`` — ground truth
  collection and pseudonymous erasure.
* ``outbox_events.claimed_at`` / ``attempts`` — lease-based claiming, replacing the
  ``FOR UPDATE SKIP LOCKED`` approach that SQLite silently ignored.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    prediction_columns = _columns("prediction_events")

    with op.batch_alter_table("prediction_events") as batch:
        if "label" not in prediction_columns:
            batch.add_column(sa.Column("label", sa.Integer(), nullable=True))
        if "labeled_at" not in prediction_columns:
            batch.add_column(
                sa.Column("labeled_at", sa.DateTime(timezone=True), nullable=True)
            )
        if "subject_key" not in prediction_columns:
            batch.add_column(sa.Column("subject_key", sa.String(64), nullable=True))

    if "label" not in prediction_columns:
        op.create_index("ix_prediction_events_label", "prediction_events", ["label"])
    if "subject_key" not in prediction_columns:
        op.create_index(
            "ix_prediction_events_subject_key", "prediction_events", ["subject_key"]
        )

    outbox_columns = _columns("outbox_events")

    with op.batch_alter_table("outbox_events") as batch:
        if "claimed_at" not in outbox_columns:
            batch.add_column(
                sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True)
            )
        if "attempts" not in outbox_columns:
            batch.add_column(
                sa.Column(
                    "attempts", sa.Integer(), nullable=False, server_default="0"
                )
            )

    if "claimed_at" not in outbox_columns:
        op.create_index("ix_outbox_events_claimed_at", "outbox_events", ["claimed_at"])


def downgrade() -> None:
    op.drop_index("ix_outbox_events_claimed_at", table_name="outbox_events")
    with op.batch_alter_table("outbox_events") as batch:
        batch.drop_column("attempts")
        batch.drop_column("claimed_at")

    op.drop_index("ix_prediction_events_subject_key", table_name="prediction_events")
    op.drop_index("ix_prediction_events_label", table_name="prediction_events")
    with op.batch_alter_table("prediction_events") as batch:
        batch.drop_column("subject_key")
        batch.drop_column("labeled_at")
        batch.drop_column("label")
