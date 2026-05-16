#!/bin/bash

# Exit on any error (but don't fail on optional services)
set -e

echo "Running migrations..."
alembic upgrade head || echo "No migrations found or migrations failed. Continuing..."

# Start Celery worker only if Redis is configured
if [ -n "$REDIS_URL" ]; then
  echo "Starting Celery worker in the background..."
  celery -A services.tasks:celery_app worker --loglevel=info --concurrency=1 &
else
  echo "REDIS_URL not set — skipping Celery worker (tasks will run synchronously)"
fi

echo "Starting FastAPI server..."
# Render sets the PORT environment variable automatically
uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
