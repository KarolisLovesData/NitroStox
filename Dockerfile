# Cloud Run image for the NitroStox Streamlit app.
# Expected layout (repo root): Dockerfile, requirements.txt, src/app.py, src/upgraded_nitrostox.py, src/sec_rag.py
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached between code-only changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ .

# Cloud Run sets $PORT (8080 by default).
EXPOSE 8080
CMD exec streamlit run app.py \
    --server.port="${PORT:-8080}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
