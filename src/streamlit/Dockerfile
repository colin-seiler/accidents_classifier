# src/streamlit/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ COPY THE WHOLE src TREE (same idea as API)
COPY .. /app/src

# Data directory (mounted)
RUN mkdir -p /app/data

ENV PYTHONPATH=/app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]