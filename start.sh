#!/bin/sh
# Use PORT environment variable provided by Render, or default to 8000
# Use PORT environment variable (Hugging Face sets this, or expects 7860)
PORT="${PORT:-7860}"

# Use Gunicorn with Uvicorn workers for production
# Workers defaults to 1 for safety on low-memory instances (1-2GB RAM).
# Set WEB_CONCURRENCY env var to increase workers on larger instances.
WORKERS="${WEB_CONCURRENCY:-1}"

# Timeout = 120s (Face matching can be slow on cold start or weak CPUs)
TIMEOUT=120

# Limit thread pool for AI tasks (prevents CPU thrashing if multiple workers are used)
# Default to 2 threads per worker if not set, or leave it to main.py default (cpu_count)
# MAX_AI_THREADS="${MAX_AI_THREADS:-2}" 
# export MAX_AI_THREADS

echo "DEBUG: System thinks PORT is set to: '$PORT'"
echo "Starting Gunicorn on port $PORT with $WORKERS workers..."

exec gunicorn main:app \
    --workers "$WORKERS" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:"$PORT" \
    --timeout "$TIMEOUT" \
    --access-logfile - \
    --error-logfile -
