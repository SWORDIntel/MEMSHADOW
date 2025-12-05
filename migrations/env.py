import os
from logging.config import fileConfig
from dotenv import load_dotenv

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Load .env file from the project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from app.db.postgres import Base
from app.core.config import settings  # Import Pydantic settings
# Import all models here so that autogenerate can see them
from app.models.user import User
from app.models.memory import Memory
from app.models.auth import WebAuthnCredential
from app.models.audit import AuditEvent
from app.models.chimera import Lure, TriggerEvent
# Import DSMILSYSTEM model
try:
    from app.models.memory_dsmil import Memory as MemoryDSMIL
except ImportError:
    pass  # Model may not exist in older versions

target_metadata = Base.metadata

# Overwrite the sqlalchemy.url from the alembic.ini file
# with a synchronous one for Alembic's operations. This is crucial
# because Alembic runs synchronously and requires a sync driver like psycopg2,
# while the main application uses an async driver (asyncpg).
if settings.POSTGRES_USER:
    sync_db_url = (
        f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_SERVER}/{settings.POSTGRES_DB}"
    )
    config.set_main_option('sqlalchemy.url', sync_db_url)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
