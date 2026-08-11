# --------------------------------------------------------------------------- #
# Stage 1: build dependencies and train the model
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --upgrade pip && pip install .

RUN python -m scripts.run_experiments --quick
RUN python -c "from src.artifacts import load_bundle; b = load_bundle(); print('artifact ok:', b.manifest.version)"

# --------------------------------------------------------------------------- #
# Stage 2: runtime (Unified API + UI)
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser entrypoint.sh ./

COPY --from=builder --chown=appuser:appuser /build/artifacts/ ./artifacts/

RUN mkdir -p /app/audit /app/reports && chown -R appuser:appuser /app/audit /app/reports
RUN chmod +x /app/entrypoint.sh

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
