FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full src tree (needed for pickle)
COPY . /app/src

RUN mkdir -p /app/models

ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]