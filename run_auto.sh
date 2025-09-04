#!/usr/bin/env bash
set -euo pipefail

# Detect GPU availability via nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  COMPOSE_FILE="docker-compose.yml"
  echo "GPU detected. Using ${COMPOSE_FILE}."
else
  COMPOSE_FILE="docker-compose.cpu.yml"
  echo "No GPU detected. Using ${COMPOSE_FILE}."
fi

if [ "$#" -gt 0 ]; then
  # Pass through custom docker compose commands, e.g., ./run_auto.sh down
  docker compose -f "${COMPOSE_FILE}" "$@"
else
  # Default: build and start detached
  docker compose -f "${COMPOSE_FILE}" build
  docker compose -f "${COMPOSE_FILE}" up -d
fi


