#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " MeetingAI - Transcrição, Separação de Pessoas e Atas de Reunião"
echo "====================================================================="
echo ""

if ! docker info > /dev/null 2>&1; then
    echo "[ERRO] O daemon do Docker não está rodando!"
    echo "Inicie o serviço do Docker e tente novamente."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "[INFO] Criando arquivo .env a partir de .env.example..."
    cp .env.example .env
fi

echo "[INFO] Construindo e subindo containers Docker..."
docker compose up --build -d

echo ""
echo "====================================================================="
echo "[SUCESSO] Sistema em execução com sucesso!"
echo ""
echo "Interface Web disponível em:"
echo "👉 http://localhost:8000"
echo ""
echo "Para baixar o modelo de LLM recomendado (Llama 3.2):"
echo "  docker compose exec ollama ollama pull llama3.2:3b"
echo ""
echo "Logs em tempo real:"
echo "  docker compose logs -f app"
echo ""
echo "Para parar os containers:"
echo "  docker compose down"
echo "====================================================================="
echo ""

# Abrir navegador se disponível
if which xdg-open > /dev/null; then
    xdg-open http://localhost:8000 &
elif which open > /dev/null; then
    open http://localhost:8000 &
fi
