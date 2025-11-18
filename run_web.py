#!/usr/bin/env python3
"""
MEMSHADOW Web Interface Launcher
Starts the FastAPI web server for MEMSHADOW management
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="MEMSHADOW Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    # Ensure directories exist
    (project_root / "config").mkdir(exist_ok=True)
    (project_root / "app" / "web" / "static" / "css").mkdir(parents=True, exist_ok=True)
    (project_root / "app" / "web" / "static" / "js").mkdir(parents=True, exist_ok=True)
    (project_root / "app" / "web" / "templates").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMSHADOW WEB INTERFACE")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Auto-reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print("=" * 80)
    print()
    print("Starting server...")
    print(f"Dashboard: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/")
    print(f"API Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/api/docs")
    print()
    print("Press CTRL+C to stop")
    print("=" * 80)

    try:
        import uvicorn
        from app.web.api import app

        uvicorn.run(
            "app.web.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    except ImportError as e:
        print(f"\nError: Missing dependencies. Please install requirements:")
        print(f"  pip install -r requirements-web.txt")
        print(f"\nDetails: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\nError starting web interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
