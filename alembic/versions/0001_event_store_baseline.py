"""Event store baseline: prediction and outbox tables.

Revision ID: 0001
Revises:
Create Date: 2026-08-10

Baseline matching the schema that ``Base.metadata.create_all`` produced before
migrations existed. Written idempotently so it can be stamped onto a database that
already has these tables.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("prediction_events"):
        op.create_table(
            "prediction_events",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("request_id", sa.String(64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("model_version", sa.String(64), nullable=True),
            sa.Column("probability", sa.Float(), nullable=False),
            sa.Column("prediction", sa.Integer(), nullable=False),
            sa.Column("latency_seconds", sa.Float(), nullable=False),
            sa.Column("features", sa.JSON(), nullable=False),
        )
        op.create_index(
            "ix_prediction_events_request_id", "prediction_events", ["request_id"]
        )
        op.create_index(
            "ix_prediction_events_created_at", "prediction_events", ["created_at"]
        )

    if not _has_table("outbox_events"):
        op.create_table(
            "outbox_events",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("event_type", sa.String(64), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_outbox_events_created_at", "outbox_events", ["created_at"])
        op.create_index("ix_outbox_events_event_type", "outbox_events", ["event_type"])


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("prediction_events")
