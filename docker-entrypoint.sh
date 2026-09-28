#!/bin/sh
# docker-entrypoint.sh
#
# Accepts an optional first argument to select the run mode:
#
#   interactive  (default) — boots SapBERT, then opens a read-eval loop
#                            asking for drug term + NCT IDs on stdin
#   api          — boots SapBERT and starts the FastAPI HTTP server on $SERVER_PORT
#
# Examples:
#   docker run -it <image>                    # interactive mode
#   docker run -d -p 8000:8000 <image> api    # API server mode

set -e

MODE="${1:-interactive}"

case "$MODE" in
  api)
    echo "Starting in API server mode (port ${SERVER_PORT:-8000})..."
    exec python -c "
from dotenv import load_dotenv
load_dotenv()

from src.schemas.models import AppConfig
from src.api.routes import Server
import uvicorn, os

config = AppConfig.from_env()

# Boot SapBERT background thread
from src.api.sapbert_server import SapBERTServer
SapBERTServer(config)

# Wire and serve FastAPI app
server = Server(config)
uvicorn.run(server.server, host='0.0.0.0', port=config.server.server_port)
"
    ;;

  interactive|*)
    echo "Starting in interactive mode..."
    exec python main.py
    ;;
esac
