from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.migrations import get_migration_database_url, to_sqlalchemy_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def resolve_database_url() -> str:
    configured_database_url = config.attributes.get("database_url")
    if configured_database_url:
        return str(configured_database_url)
    return get_migration_database_url()


def run_migrations_offline() -> None:
    database_url = to_sqlalchemy_url(resolve_database_url())
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(resolve_database_url()))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
