# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1: Install Python dependencies
# ============================================================================
FROM python:3.13-slim AS deps

WORKDIR /install

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive

# Install standard HTTPS certificates.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first to improve Docker layer caching.
COPY pyproject.toml README.md ./

# Install uv and CPU-only PyTorch.
#
# The insecure-host exceptions are limited to the official PyTorch download
# domains because the corporate network is presenting an unrecognized issuer.
#
# This does not install CUDA or NVIDIA dependencies.
RUN python -m pip install --no-cache-dir uv \
    && uv pip install \
        --system \
        --no-cache \
        --allow-insecure-host download.pytorch.org \
        --allow-insecure-host download-r2.pytorch.org \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.14.0"

# Install the remaining Python dependencies from PyPI.
RUN uv pip install \
        --system \
        --system-certs \
        --no-cache \
        --index-url https://pypi.org/simple \
        "transformers>=4.40.0" \
        "fastapi>=0.141.1" \
        "uvicorn>=0.52.4" \
        "openai>=3.8.0" \
        "python-dotenv>=1.2.3" \
        "qdrant-client>=1.9.1,<1.19.0" \
        "pydantic>=2.0" \
        "safetensors" \
        "tqdm" \
        "httpx" \
        "sse-starlette>=1.6.5" \
        "huggingface-hub>=0.24" \
        "openpyxl>=3.1.5"


# ============================================================================
# Stage 2: Build the Vue frontend (see frontend/)
# ============================================================================
FROM node:22-slim AS frontend

WORKDIR /build

# Install frontend dependencies from the lockfile first for better layer caching.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Build the SPA. vite.config.js emits the bundle to dist/.
COPY frontend/ ./
RUN npm run build


# ============================================================================
# Stage 3: Runtime image
# ============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_DIR=/app/models \
    SAPBERT_PORT=7997 \
    SERVER_PORT=8000 \
    SAPBERT_MODEL=sapbert \
    SAPBERT_BASE_URL=http://localhost:7997/v1 \
    SAPBERT_API_KEY=none


# libgomp1 is required by CPU-only PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python packages and executables installed in the dependency stage.
COPY --from=deps /usr/local/lib/python3.13/site-packages/ \
    /usr/local/lib/python3.13/site-packages/

COPY --from=deps /usr/local/bin/ \
    /usr/local/bin/

# Copy the application source.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py ./main.py
COPY pyproject.toml README.md ./

# Copy the built Vue frontend bundle, which FastAPI serves from frontend/dist.
COPY --from=frontend /build/dist ./frontend/dist

# Install the local application package without reinstalling dependencies.
RUN python -m pip install \
    --no-cache-dir \
    --no-deps \
    -e .

# Create application data and model directories.
RUN mkdir -p /app/data /app/models

# Copy the container entrypoint.
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Remove Windows CRLF characters and make the entrypoint executable.
RUN sed -i 's/\r$//' /docker-entrypoint.sh \
    && chmod +x /docker-entrypoint.sh

VOLUME ["/app/models"]

EXPOSE 7997
EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
