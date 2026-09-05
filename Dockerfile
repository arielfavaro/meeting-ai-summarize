FROM python:3.10-slim

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependências de sistema essenciais (FFmpeg, libsndfile, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências Python
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copiar código da aplicação
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Criar diretórios de dados
RUN mkdir -p /app/data/uploads /app/data/processed /app/data/exports /app/data/models_cache

# Variáveis de ambiente padrão no container
ENV DATA_DIR=/app/data \
    HF_HOME=/app/data/models_cache \
    PYTHONPATH=/app

EXPOSE 8000

# Checagem de saúde do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Inicializar servidor web e API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
