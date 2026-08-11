# Credit risk decision service.
#
# Two stages. The builder installs dependencies and trains a model; the runtime
# carries only what is needed to serve. The trained artifact is copied across,
# so the image that starts is the image that was validated -- no training at
# container start, and no chance of a model appearing that nothing checked.

# --------------------------------------------------------------------------- #
# Stage 1: build dependencies and train the model
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# LightGBM needs libgomp at build and run time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
# Uncomment the following line if your training script reads data from a local folder:
# COPY data/ ./data/

RUN pip install --upgrade pip && pip install .

# Train inside the image. The artifact is therefore reproducible from the
# Dockerfile alone and is validated before the runtime stage ever sees it.
RUN python -m scripts.run_experiments --quick

# Fail the build if the artifact cannot be loaded. Catching a model/calibrator
# mismatch here costs a red build; catching it in production costs a year of
# wrongly priced lending.
RUN python -c "from src.artifacts import load_bundle; b = load_bundle(); print('artifact ok:', b.manifest.version)"

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. A service that writes an append-only audit log has no
# business running as root.
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

# The model and its calibrator travel together, from the same training run.
COPY --from=builder --chown=appuser:appuser /build/artifacts/ ./artifacts/

RUN mkdir -p /app/audit /app/reports && chown -R appuser:appuser /app/audit /app/reports

USER appuser
EXPOSE 8080

# The service refuses to start on a model/calibrator mismatch, so an unhealthy
# container is a genuine signal rather than a slow start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
