FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional, larger: richer static analysis and semantic embeddings
# RUN pip install --no-cache-dir -r requirements-optional.txt
COPY pyproject.toml ./
COPY codelens ./codelens
COPY evaluation ./evaluation
COPY finetune ./finetune
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "codelens.server:app", "--host", "0.0.0.0", "--port", "8000"]
