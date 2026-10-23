# Streamlit front end. It talks to the API over HTTP and contains no model code.
FROM python:3.12.8-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-app.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements-app.txt

FROM python:3.12.8-slim-bookworm
RUN useradd --create-home --uid 10001 app && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app ./app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 MVI_API_URL=http://api:8000 \
    MVI_RESULTS_DIR=/results MVI_DATA_DIR=/data/mvtec_ad
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
