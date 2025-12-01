#!/bin/bash
# MEMSHADOW Setup Script for New Features
# Run this script to set up the new memlayer-inspired features

set -e

echo "================================================"
echo "MEMSHADOW Setup - Memlayer-Inspired Features"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the MEMSHADOW root directory"
    exit 1
fi

echo "✓ Directory check passed"
echo ""

# Step 1: Install SDK (optional)
echo "Step 1: Install SDK (Optional)"
echo "------------------------------------"
read -p "Do you want to install the memshadow-sdk package? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd sdk
    pip install -e .
    cd ..
    echo "✓ SDK installed successfully"
else
    echo "⊘ SDK installation skipped"
fi

echo ""

# Step 2: Celery Beat Configuration
echo "Step 2: Celery Beat Configuration"
echo "------------------------------------"
echo "Task reminders require Celery Beat to be running."
echo ""
echo "To start Celery Beat (in a separate terminal):"
echo "  celery -A app.workers.celery_app beat --loglevel=info"
echo ""
echo "To start Celery Worker (in another terminal):"
echo "  celery -A app.workers.celery_app worker --loglevel=info"
echo ""

# Step 3: Quick Test
echo "Step 3: Quick Test"
echo "------------------------------------"
read -p "Do you want to run a quick integration test? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "tests/test_integration.py" ]; then
        python tests/test_integration.py
    else
        echo "⚠ tests/test_integration.py not found. Skipping test."
    fi
else
    echo "⊘ Quick test skipped"
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "New Features Available:"
echo "  ✓ Operation Modes (LOCAL/ONLINE/LIGHTWEIGHT) - Configured via install.sh or .env"
echo "  ✓ Task Reminders with scheduled notifications"
echo "  ✓ Python SDK with OpenAI & Anthropic wrappers"
echo "  ✓ ChromaDB batch operations"
echo ""
echo "API Endpoints Added:"
echo "  POST   /api/v1/reminders/         - Create reminder"
echo "  GET    /api/v1/reminders/         - List reminders"
echo "  GET    /api/v1/reminders/{id}     - Get reminder"
echo "  PATCH  /api/v1/reminders/{id}     - Update reminder"
echo "  POST   /api/v1/reminders/{id}/complete"
echo "  DELETE /api/v1/reminders/{id}     - Delete reminder"
echo ""
echo "Next Steps:"
echo "  1. Start the FastAPI server: uvicorn app.main:app --reload"
echo "  2. Start Celery worker & beat (see Step 2 above)"
echo "  3. Try the SDK examples in sdk/examples/"
echo ""
echo "Documentation:"
echo "  - SDK README: sdk/README.md"
echo "  - API docs: http://localhost:8000/docs"
echo ""