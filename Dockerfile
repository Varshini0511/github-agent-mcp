# GitHub Agent -- container image for the Streamlit UI (default) or the API.
#
#   docker build -t github-agent .
#   docker run --env-file .env -p 8501:8501 github-agent                 # web UI
#   docker run --env-file .env -p 8000:8000 github-agent \
#       uvicorn api:app --host 0.0.0.0 --port 8000                        # HTTP API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (cached layer unless requirements change).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as a non-root user; state (memory db, vector index) lives in /data.
RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /data && chown -R agent:agent /app /data
USER agent
ENV CHECKPOINT_DB=/data/agent_memory.sqlite \
    VECTOR_DIR=/data/vector_store

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
