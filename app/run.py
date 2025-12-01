import subprocess
import sys
import os
from dotenv import load_dotenv

def main():
    """
    Centralized script runner.
    Loads the .env file and then executes the given command.
    """
    # Programmatically add the project root to the Python path.
    # This makes the script resilient to `sudo` stripping PYTHONPATH.
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python run.py <command> [args...]")
        print("Commands: migrate, upgrade, test")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # The PYTHONPATH env var is no longer needed here for the subprocess
    env = {**os.environ}

    if command == "migrate":
        if not args:
            print("Error: Migration message is required.")
            print("Usage: python run.py migrate \"Your migration message\"")
            sys.exit(1)
        cmd = ["alembic", "revision", "--autogenerate", "-m", args[0]]
    elif command == "upgrade":
        cmd = ["alembic", "upgrade", "head"]
    elif command == "test":
        cmd = ["pytest", *args]
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()