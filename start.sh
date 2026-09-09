#!/bin/bash
set -e

export VECTORDB_HOST=0.0.0.0
export VECTORDB_PORT="${PORT:-10000}"

echo "=========================================="
echo "Starting NeuroIndex"
echo "=========================================="
echo "Host: $VECTORDB_HOST"
echo "Port: $VECTORDB_PORT"

python3 main.py
