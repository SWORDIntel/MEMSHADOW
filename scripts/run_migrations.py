import os
import sys

# --- Diagnostic Block ---
print("--- Starting Diagnostic ---")
# Print the current working directory from the script's perspective
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")

# Define the project root and list its contents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Calculated Project Root: {project_root}")
print(f"--- Listing contents of {project_root} ---")
try:
    for item in os.listdir(project_root):
        print(item)
except FileNotFoundError:
    print(f"Error: Directory not found: {project_root}")
print("--- End of Directory Listing ---")

# --- Foolproof Environment Loading ---
from dotenv import load_dotenv
sys.path.insert(0, project_root)
dotenv_path = os.path.join(project_root, '.env')

if os.path.exists(dotenv_path):
    print(f"SUCCESS: Found .env file at: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"FATAL: .env file not found at {dotenv_path}. Cannot proceed.")
    sys.exit(1)
# --- End of Loading Logic ---


# Now that the environment is loaded, we can safely import application modules.
from alembic.config import Config
from alembic import command
from app.core.config import settings

def run_migrations():
    """
    Applies Alembic migrations programmatically, ensuring the environment is pre-loaded.
    """
    print("--- Running database migrations ---")

    alembic_ini_path = os.path.join(project_root, 'alembic.ini')

    if not os.path.exists(alembic_ini_path):
        print(f"Error: alembic.ini not found at {alembic_ini_path}")
        return

    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option('sqlalchemy.url', str(settings.DATABASE_URL))

    print("Applying migrations to 'head'...")
    try:
        command.upgrade(alembic_cfg, "head")
        print("--- Migrations applied successfully ---")
    except Exception as e:
        print(f"An error occurred during migration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_migrations()