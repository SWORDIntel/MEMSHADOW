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

# Step 1: Database Migrations
echo "Step 1: Running Database Migrations"
echo "------------------------------------"
echo "This will create the task_reminders table..."
echo ""

# Check if database is accessible
if ! command -v alembic &> /dev/null; then
    echo "⚠ Alembic not found. Installing..."
    pip install alembic
fi

# Run migrations
echo "Running: alembic upgrade head"
alembic upgrade head

if [ $? -eq 0 ]; then
    echo "✓ Migrations completed successfully"
else
    echo "❌ Migration failed. Please check your database connection."
    exit 1
fi

echo ""

# Step 2: Install SDK (optional)
echo "Step 2: Install SDK (Optional)"
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

# Step 3: Environment Variables
echo "Step 3: Environment Variables"
echo "------------------------------------"
echo "The following new environment variable is available:"
echo ""
echo "MEMORY_OPERATION_MODE (default: local)"
echo "  Options: local, online, lightweight"
echo "  - local: Full enrichment, all features"
echo "  - online: Balanced speed/features"
echo "  - lightweight: Minimal processing"
echo ""

if [ -f ".env" ]; then
    if grep -q "MEMORY_OPERATION_MODE" .env; then
        echo "✓ MEMORY_OPERATION_MODE already set in .env"
    else
        echo "Adding MEMORY_OPERATION_MODE to .env..."
        echo "" >> .env
        echo "# Memory Operation Mode (local/online/lightweight)" >> .env
        echo "MEMORY_OPERATION_MODE=local" >> .env
        echo "✓ Added to .env"
    fi
else
    echo "⚠ No .env file found. Please create one and add:"
    echo "MEMORY_OPERATION_MODE=local"
fi

echo ""

# Step 4: Celery Beat Configuration
echo "Step 4: Celery Beat Configuration"
echo "------------------------------------"
echo "Task reminders require Celery Beat to be running."
echo ""
echo "To start Celery Beat (in a separate terminal):"
echo "  celery -A app.workers.celery_app beat --loglevel=info"
echo ""
echo "To start Celery Worker (in another terminal):"
echo "  celery -A app.workers.celery_app worker --loglevel=info"
echo ""

# Step 5: Quick Test
echo "Step 5: Quick Test"
echo "------------------------------------"
read -p "Do you want to run a quick integration test? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "test_integration.py" ]; then
        python test_integration.py
    else
        echo "⚠ test_integration.py not found. Skipping test."
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
echo "  ✓ Operation Modes (LOCAL/ONLINE/LIGHTWEIGHT)"
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
echo "  2. Start Celery worker & beat (see Step 4 above)"
echo "  3. Try the SDK examples in sdk/examples/"
echo ""
echo "Documentation:"
echo "  - SDK README: sdk/README.md"
echo "  - API docs: http://localhost:8000/docs"
echo ""
