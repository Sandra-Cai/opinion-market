"""Initial migration

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_premium', sa.Boolean(), nullable=False, default=False),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('notification_settings', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    op.create_index(op.f('ix_users_created_at'), 'users', ['created_at'], unique=False)

    # Create markets table
    op.create_table('markets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('category', sa.Enum('POLITICS', 'SPORTS', 'ECONOMICS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'OTHER', name='marketcategory'), nullable=False),
        sa.Column('outcome_a', sa.String(length=100), nullable=False),
        sa.Column('outcome_b', sa.String(length=100), nullable=False),
        sa.Column('creator_id', sa.Integer(), nullable=False),
        sa.Column('closes_at', sa.DateTime(), nullable=False),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('resolution_criteria', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('OPEN', 'CLOSED', 'RESOLVED', 'CANCELLED', name='marketstatus'), nullable=False, default='OPEN'),
        sa.Column('price_a', sa.Float(), nullable=False, default=0.5),
        sa.Column('price_b', sa.Float(), nullable=False, default=0.5),
        sa.Column('volume_total', sa.Float(), nullable=False, default=0.0),
        sa.Column('volume_24h', sa.Float(), nullable=False, default=0.0),
        sa.Column('trending_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('sentiment_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('liquidity_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('risk_score', sa.Float(), nullable=False, default=0.0),
        sa.Column('initial_liquidity', sa.Float(), nullable=False, default=1000.0),
        sa.Column('trading_fee', sa.Float(), nullable=False, default=0.02),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['creator_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_markets_category'), 'markets', ['category'], unique=False)
    op.create_index(op.f('ix_markets_created_at'), 'markets', ['created_at'], unique=False)
    op.create_index(op.f('ix_markets_status'), 'markets', ['status'], unique=False)
    op.create_index(op.f('ix_markets_trending_score'), 'markets', ['trending_score'], unique=False)

    # Create orders table
    op.create_table('orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('market_id', sa.Integer(), nullable=False),
        sa.Column('order_type', sa.Enum('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', name='ordertype'), nullable=False),
        sa.Column('trade_type', sa.Enum('BUY', 'SELL', name='tradetype'), nullable=False),
        sa.Column('outcome', sa.Enum('OUTCOME_A', 'OUTCOME_B', name='tradeoutcome'), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('stop_price', sa.Float(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED', name='orderstatus'), nullable=False, default='PENDING'),
        sa.Column('time_in_force', sa.Enum('GTC', 'IOC', 'FOK', name='timeinforce'), nullable=False, default='GTC'),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('filled_amount', sa.Float(), nullable=False, default=0.0),
        sa.Column('average_price', sa.Float(), nullable=True),
        sa.Column('total_fees', sa.Float(), nullable=False, default=0.0),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('filled_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_orders_market_id'), 'orders', ['market_id'], unique=False)
    op.create_index(op.f('ix_orders_status'), 'orders', ['status'], unique=False)
    op.create_index(op.f('ix_orders_user_id'), 'orders', ['user_id'], unique=False)
    op.create_index(op.f('ix_orders_created_at'), 'orders', ['created_at'], unique=False)

    # Create trades table
    op.create_table('trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('trade_type', sa.Enum('BUY', 'SELL', name='tradetype'), nullable=False),
        sa.Column('outcome', sa.Enum('OUTCOME_A', 'OUTCOME_B', name='tradeoutcome'), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('price_a', sa.Float(), nullable=False),
        sa.Column('price_b', sa.Float(), nullable=False),
        sa.Column('price_per_share', sa.Float(), nullable=False),
        sa.Column('total_value', sa.Float(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'EXECUTED', 'COMPLETED', 'CANCELLED', 'FAILED', 'PARTIALLY_FILLED', name='tradestatus'), nullable=False, default='COMPLETED'),
        sa.Column('fee', sa.Float(), nullable=False, default=0.0),
        sa.Column('fee_amount', sa.Float(), nullable=False, default=0.0),
        sa.Column('price_impact', sa.Float(), nullable=False, default=0.0),
        sa.Column('slippage', sa.Float(), nullable=False, default=0.0),
        sa.Column('order_id', sa.Integer(), nullable=True),
        sa.Column('fill_amount', sa.Float(), nullable=True),
        sa.Column('fill_price', sa.Float(), nullable=True),
        sa.Column('market_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('trade_hash', sa.String(length=255), nullable=True),
        sa.Column('gas_fee', sa.Float(), nullable=False, default=0.0),
        sa.Column('additional_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('executed_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_trades_market_id'), 'trades', ['market_id'], unique=False)
    op.create_index(op.f('ix_trades_status'), 'trades', ['status'], unique=False)
    op.create_index(op.f('ix_trades_trade_hash'), 'trades', ['trade_hash'], unique=True)
    op.create_index(op.f('ix_trades_user_id'), 'trades', ['user_id'], unique=False)
    op.create_index(op.f('ix_trades_created_at'), 'trades', ['created_at'], unique=False)

    # Create indexes for performance
    op.create_index('idx_markets_creator_status', 'markets', ['creator_id', 'status'], unique=False)
    op.create_index('idx_trades_user_market', 'trades', ['user_id', 'market_id'], unique=False)
    op.create_index('idx_orders_user_status', 'orders', ['user_id', 'status'], unique=False)
    op.create_index('idx_markets_category_status', 'markets', ['category', 'status'], unique=False)

    # Add check constraints
    op.create_check_constraint('check_positive_amount', 'trades', 'amount > 0')
    op.create_check_constraint('check_price_range', 'trades', 'price_per_share >= 0 AND price_per_share <= 1')
    op.create_check_constraint('check_positive_total_value', 'trades', 'total_value > 0')
    op.create_check_constraint('check_non_negative_fee', 'trades', 'fee >= 0')
    op.create_check_constraint('check_non_negative_price_impact', 'trades', 'price_impact >= 0')

    # Add triggers for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    op.execute("""
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_markets_updated_at BEFORE UPDATE ON markets
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

    op.execute("""
        CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_orders_updated_at ON orders;")
    op.execute("DROP TRIGGER IF EXISTS update_markets_updated_at ON markets;")
    op.execute("DROP TRIGGER IF EXISTS update_users_updated_at ON users;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # Drop check constraints
    op.drop_constraint('check_non_negative_price_impact', 'trades', type_='check')
    op.drop_constraint('check_non_negative_fee', 'trades', type_='check')
    op.drop_constraint('check_positive_total_value', 'trades', type_='check')
    op.drop_constraint('check_price_range', 'trades', type_='check')
    op.drop_constraint('check_positive_amount', 'trades', type_='check')

    # Drop indexes
    op.drop_index('idx_markets_category_status', table_name='markets')
    op.drop_index('idx_orders_user_status', table_name='orders')
    op.drop_index('idx_trades_user_market', table_name='trades')
    op.drop_index('idx_markets_creator_status', table_name='markets')

    # Drop tables
    op.drop_table('trades')
    op.drop_table('orders')
    op.drop_table('markets')
    op.drop_table('users')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS tradestatus;")
    op.execute("DROP TYPE IF EXISTS tradeoutcome;")
    op.execute("DROP TYPE IF EXISTS tradetype;")
    op.execute("DROP TYPE IF EXISTS timeinforce;")
    op.execute("DROP TYPE IF EXISTS orderstatus;")
    op.execute("DROP TYPE IF EXISTS marketstatus;")
    op.execute("DROP TYPE IF EXISTS marketcategory;")
    op.execute("DROP TYPE IF EXISTS ordertype;")
