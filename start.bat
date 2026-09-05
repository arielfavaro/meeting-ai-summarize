@echo off
chcp 65001 > nul
echo =====================================================================
echo  MeetingAI - Transcrição, Separação de Pessoas e Atas de Reunião
echo =====================================================================
echo.

:: Verificar se Docker está rodando
docker info > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] O Docker Desktop não parece estar rodando!
    echo Por favor, inicie o Docker Desktop e tente novamente.
    echo.
    pause
    exit /b 1
)

:: Copiar .env se não existir
if not exist ".env" (
    echo [INFO] Criando arquivo .env a partir de .env.example...
    copy ".env.example" ".env" > nul
)

echo [INFO] Construindo e iniciando containers Docker...
echo Isso pode levar alguns minutos no primeiro download de imagens.
echo.

docker compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Falha ao iniciar os containers do Docker!
    echo Verifique os logs com: docker compose logs
    echo.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCESSO] Sistema em execução com sucesso!
echo.
echo Interface Web disponível em:
echo 👉 http://localhost:8000
echo.
echo Para baixar o modelo de LLM local recomendado (caso ainda não tenha):
echo   docker compose exec ollama ollama pull llama3.2:3b
echo.
echo Para acompanhar os logs em tempo real:
echo   docker compose logs -f app
echo.
echo Para parar o sistema:
echo   docker compose down
echo =====================================================================
echo.

:: Abrir navegador automaticamente
start http://localhost:8000

pause
