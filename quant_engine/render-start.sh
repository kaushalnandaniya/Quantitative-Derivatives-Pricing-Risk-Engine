#!/bin/bash

# Exit on any error
set -e

echo "Running migrations..."
alembic upgrade head || echo "No migrations found or migrations failed. Continuing..."

echo "Starting Celery worker in the background..."
celery -A core.celery_app worker --loglevel=info &

echo "Starting FastAPI server..."
# Render sets the PORT environment variable automatically
uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
