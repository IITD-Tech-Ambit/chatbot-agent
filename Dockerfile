# ─── Stage 1: dependency builder ────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# System deps needed only to compile wheels (e.g. hiredis)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install project + runtime extras into an isolated prefix so we can copy it cleanly
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir gunicorn \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps -e .


# ─── Stage 2: lean runtime image ─────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Don't write .pyc files; force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/usr/local/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

# Non-root user for least-privilege execution
RUN addgroup --system appgroup \
 && adduser --system --ingroup appgroup --no-create-home appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/gunicorn                  /usr/local/bin/gunicorn
COPY --from=builder /usr/local/bin/uvicorn                   /usr/local/bin/uvicorn

# Copy application source
COPY src/ ./src/

# Drop to non-root
USER appuser

EXPOSE 3003

# -u clears any HTTP_PROXY/HTTPS_PROXY that may be present container-wide
# (e.g. for outbound xAI/Groq calls, see LLM_HTTP_PROXY_URL in groq_client.py)
# before this call: Python's urllib.request auto-honors those vars by
# default, and a proxy can't/won't forward a request to a loopback address —
# same class of false-negative health check bug fixed in auth-service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:3003/health')"

# gunicorn + UvicornWorker gives graceful shutdown, worker recycling, and
# process-level isolation while keeping the async event loop intact.
# WEB_CONCURRENCY can be overridden at deploy time (default: 2).
CMD ["sh", "-c", \
     "gunicorn agent.main:app \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers ${WEB_CONCURRENCY:-2} \
        --bind 0.0.0.0:3003 \
        --timeout 120 \
        --graceful-timeout 30 \
        --keep-alive 5 \
        --log-level ${LOG_LEVEL:-info} \
        --access-logfile - \
        --error-logfile -"]
