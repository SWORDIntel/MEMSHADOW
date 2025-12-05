"""Add DSMILSYSTEM memory schema

Revision ID: f1g2h3i4j5k6
Revises: a1b2c3d4e5f6
Create Date: 2025-01-XX

This migration creates the DSMILSYSTEM-aligned memory schema with:
- Layer semantics (2-9)
- Device semantics (0-103)
- Clearance tokens
- Multi-tier storage
- Event correlation IDs
- ROE metadata
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'f1g2h3i4j5k6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    """Create DSMILSYSTEM memory table"""
    
    # Create memories_dsmil table
    op.create_table(
        'memories_dsmil',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # DSMILSYSTEM semantics
        sa.Column('layer_id', sa.Integer(), nullable=False),
        sa.Column('device_id', sa.Integer(), nullable=False),
        sa.Column('clearance_token', sa.String(128), nullable=False),
        
        # User context
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('users.id'), nullable=False),
        
        # Content
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('embedding', Vector(2048)),  # 2048d INT8 quantized
        sa.Column('tags', postgresql.JSONB(), nullable=False, server_default='[]'),
        
        # Metadata
        sa.Column('extra_data', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('roe_metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('correlation_id', sa.String(128)),
        
        # Storage tier
        sa.Column('tier', sa.Enum('hot', 'warm', 'cold', name='memorytier'), 
                  nullable=False, server_default='cold'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('accessed_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('promoted_at', sa.DateTime()),
        sa.Column('demoted_at', sa.DateTime()),
        
        # Constraints
        sa.CheckConstraint('layer_id >= 2 AND layer_id <= 9', name='check_layer_range'),
        sa.CheckConstraint('device_id >= 0 AND device_id <= 103', name='check_device_range'),
        sa.CheckConstraint('char_length(content) >= 1', name='check_content_not_empty'),
    )
    
    # Create indexes
    op.create_index('idx_layer_device', 'memories_dsmil', ['layer_id', 'device_id'])
    op.create_index('idx_clearance_token', 'memories_dsmil', ['clearance_token'])
    op.create_index('idx_correlation_id', 'memories_dsmil', ['correlation_id'])
    op.create_index('idx_tier', 'memories_dsmil', ['tier'])
    op.create_index('idx_user_created', 'memories_dsmil', ['user_id', 'created_at'])
    op.create_index('idx_content_hash', 'memories_dsmil', ['content_hash'])
    op.create_index('idx_extra_data_gin', 'memories_dsmil', ['extra_data'], 
                    postgresql_using='gin')
    op.create_index('idx_tags_gin', 'memories_dsmil', ['tags'], 
                    postgresql_using='gin')


def downgrade():
    """Drop DSMILSYSTEM memory table"""
    
    # Drop indexes
    op.drop_index('idx_tags_gin', 'memories_dsmil')
    op.drop_index('idx_extra_data_gin', 'memories_dsmil')
    op.drop_index('idx_content_hash', 'memories_dsmil')
    op.drop_index('idx_user_created', 'memories_dsmil')
    op.drop_index('idx_tier', 'memories_dsmil')
    op.drop_index('idx_correlation_id', 'memories_dsmil')
    op.drop_index('idx_clearance_token', 'memories_dsmil')
    op.drop_index('idx_layer_device', 'memories_dsmil')
    
    # Drop table
    op.drop_table('memories_dsmil')
    
    # Drop enum type
    op.execute("DROP TYPE IF EXISTS memorytier")
