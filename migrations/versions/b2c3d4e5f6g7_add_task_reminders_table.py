"""add task reminders table

Revision ID: b2c3d4e5f6g7
Revises: 95cdd51c35cc
Create Date: 2025-11-21 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6g7'
down_revision = '95cdd51c35cc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create task_reminders table for scheduled memory notifications.

    This feature enables users to schedule reminders for tasks and
    receive proactive notifications when tasks become due.
    """

    # Create enum types for status and priority
    op.execute("""
        CREATE TYPE reminderstatus AS ENUM (
            'pending', 'reminded', 'completed', 'cancelled'
        )
    """)

    op.execute("""
        CREATE TYPE reminderpriority AS ENUM (
            'low', 'medium', 'high'
        )
    """)

    # Create task_reminders table
    op.create_table(
        'task_reminders',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('due_date', sa.DateTime),
        sa.Column('reminder_date', sa.DateTime, nullable=False),
        sa.Column('status', sa.Enum('pending', 'reminded', 'completed', 'cancelled',
                                    name='reminderstatus'), nullable=False, server_default='pending'),
        sa.Column('priority', sa.Enum('low', 'medium', 'high',
                                      name='reminderpriority'), nullable=False, server_default='medium'),
        sa.Column('associated_memory_id', UUID(as_uuid=True), sa.ForeignKey('memories.id'), nullable=True),
        sa.Column('extra_data', JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column('reminded_at', sa.DateTime),
        sa.Column('completed_at', sa.DateTime),
    )

    # Create indexes for efficient querying
    op.create_index(
        'idx_reminder_user_status',
        'task_reminders',
        ['user_id', 'status']
    )

    op.create_index(
        'idx_reminder_due_date',
        'task_reminders',
        ['due_date']
    )

    op.create_index(
        'idx_reminder_reminder_date',
        'task_reminders',
        ['reminder_date']
    )

    op.create_index(
        'idx_reminder_memory',
        'task_reminders',
        ['associated_memory_id']
    )

    # Add trigger to update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_task_reminder_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER task_reminder_updated_at_trigger
        BEFORE UPDATE ON task_reminders
        FOR EACH ROW
        EXECUTE FUNCTION update_task_reminder_updated_at();
    """)


def downgrade() -> None:
    """
    Remove task_reminders table and related objects.
    """
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS task_reminder_updated_at_trigger ON task_reminders")
    op.execute("DROP FUNCTION IF EXISTS update_task_reminder_updated_at()")

    # Drop indexes (will be dropped with table, but being explicit)
    op.drop_index('idx_reminder_memory', table_name='task_reminders')
    op.drop_index('idx_reminder_reminder_date', table_name='task_reminders')
    op.drop_index('idx_reminder_due_date', table_name='task_reminders')
    op.drop_index('idx_reminder_user_status', table_name='task_reminders')

    # Drop table
    op.drop_table('task_reminders')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS reminderstatus")
    op.execute("DROP TYPE IF EXISTS reminderpriority")
