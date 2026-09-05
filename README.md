# 🎙️ MeetingAI - Transcrição, Diarização e Atas com IA 100% Local (PT-BR)

> **Privacidade Total:** Grave, transcreva, separe os interlocutores e gere Atas de Reunião executivas diretamente na sua máquina via **Docker**, sem enviar nenhum dado para a nuvem.

---

## ✨ Principais Funcionalidades

- **🎧 Entrada Flexível de Áudio e Vídeo:**
  - Envie gravações prévias em múltiplos formatos (`MP3`, `WAV`, `M4A`, `OGG`, `MP4`, `WEBM`).
  - **Grave ao vivo pelo navegador** com microfone integrado e visualização em tempo real das ondas sonoras.
- **👥 Separação de Interlocutores (Diarização):**
  - Identifica automaticamente quem está falando em cada trecho (`Locutor 1`, `Locutor 2`, etc.).
  - Suporte ao modelo oficial **Pyannote.audio 3.1** (com token gratuito HuggingFace).
  - **Diarizador Fallback Nativo:** se você não configurar token do HuggingFace, o sistema utiliza agrupamento acústico próprio e não falha.
  - **Renomeação de Oradores:** edite os nomes na interface (ex: `Locutor 1` ➔ `Carlos`, `Locutor 2` ➔ `Dra. Mariana`) e regere a ata com um clique.
- **⚡ Transcrição Otimizada em Português (Faster-Whisper):**
  - Utiliza CTranslate2 para transcrição até 4x mais veloz que o Whisper tradicional com baixo consumo de memória.
  - Suporta modelos de `tiny` a `large-v3`.
- **📋 Geração de Atas de Reunião com LLM Local (Ollama):**
  - Integração nativa com **Ollama** (`llama3.2:3b`, `qwen2.5:7b`, `mistral`, etc.).
  - Produz Atas estruturadas contendo:
    - **Resumo Executivo** de 2 minutos.
    - **Tópicos Principais e Discussões** detalhadas.
    - **Decisões Tomadas** registradas formalmente.
    - **Matriz de Ações / Tarefas:** Quem, O quê, Prazo e Status.
    - **Pontos em Aberto / Próximos Passos**.
  - **Gerador Estruturado de Contingência:** caso o modelo do Ollama ainda não tenha sido baixado, a ata é construída por análise heurística sem travar o usuário.
- **📄 Exportação Profissional:**
  - Exportação em **Word (.docx)** com formatação corporativa e tabela de tarefas.
  - Exportação em **Markdown (.md)** com formatação GitHub.
  - Exportação em **Texto puro (.txt)** e visualização pronta para **Impressão / PDF**.
- **🎯 Player de Áudio Interativo e Sincronizado:**
  - Clique em qualquer balão de fala da transcrição para pular o áudio diretamente para o segundo exato (`timestamp`).
  - Velocidades de reprodução de `1x`, `1.25x` e `1.5x`.
- **🗄️ Histórico Completo:**
  - Reuniões salvas localmente em banco SQLite para consulta e re-exportação a qualquer momento.

---

## 🏛️ Arquitetura do Sistema

```
meeting-ai-summarize/
├── docker-compose.yml           # Orquestração do App + Ollama
├── docker-compose.gpu.yml       # Override para placas NVIDIA CUDA
├── Dockerfile                   # Container com Python, FFmpeg e IA
├── .env.example                 # Guia de variáveis de ambiente
├── start.bat                    # Inicializador automático para Windows
├── start.sh                     # Inicializador automático para Linux/macOS
│
├── backend/                     # API FastAPI e Motores de IA
│   ├── main.py                  # Endpoints REST e streaming SSE
│   ├── config.py                # Configurações centralizadas
│   ├── database.py              # Armazenamento SQLite local
│   ├── models/schemas.py        # Validação com Pydantic
│   └── services/
│       ├── audio_service.py     # Conversão 16kHz mono com FFmpeg
│       ├── transcription.py     # Faster-Whisper (PT-BR)
│       ├── diarization.py       # Pyannote + Fallback acústico VAD
│       ├── alignment.py         # Fusão fala x orador
│       ├── summarizer.py        # Prompt engineering para Atas no Ollama
│       └── exporter.py          # Gerador DOCX, Markdown e TXT
│
├── frontend/                    # Interface Web Moderna (SPA)
│   ├── index.html               # Layout com abas, dropzone e player
│   ├── css/styles.css           # Tema escuro elegante e glassmorphism
│   └── js/
│       ├── app.js               # Gerenciador de estado e fluxo
│       ├── player.js            # Player interativo sincronizado
│       ├── recorder.js          # Gravador de microfone com Web Audio API
│       └── settings.js          # Diagnóstico de conexão do Ollama
│
└── tests/                       # Testes de integração e validação
    ├── test_audio_pipeline.py   # Testes automatizados do pipeline
    └── generate_test_audio.py   # Gerador de áudio sintético WAV
```

---

## 🚀 Como Iniciar

### Pré-requisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando.

### 1. Início Rápido com 1 Clique

#### No Windows:
Basta dar dois cliques no arquivo:
```cmd
start.bat
```

#### No Linux / macOS:
```bash
chmod +x start.sh
./start.sh
```

Ou execute manualmente pelo terminal:
```bash
docker compose up --build -d
```

A interface web estará disponível imediatamente em:
👉 **http://localhost:8000**

---

### 2. Baixar o Modelo do Ollama (LLM Local)

No primeiro uso, baixe o modelo de linguagem recomendado (leve e excelente em português):

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

Outras excelentes opções em português:
```bash
# Alta precisão de escrita e raciocínio (7B)
docker compose exec ollama ollama pull qwen2.5:7b

# Modelo Mistral para reuniões de negócios
docker compose exec ollama ollama pull mistral:7b
```

*(O modelo fica salvo permanentemente no volume do Docker `ollama_models`).*

---

## 🎮 Aceleração por Placa de Vídeo (NVIDIA GPU / CUDA)

Se sua máquina possui uma GPU NVIDIA com drivers e o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalados, inicie com o override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```
Isso habilitará execução com `WHISPER_DEVICE=cuda` e computação em ponto flutuante `float16`.

---

## ⚙️ Conectar ao Ollama Já Instalado no Host (Opcional)

Se você já possui o Ollama rodando nativamente no seu Windows/Mac, crie um arquivo `.env` e configure:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```
Assim o container se comunicará diretamente com o Ollama da sua máquina.

---

## 🔑 Configurar Pyannote Oficial (Opcional)

Por padrão, o MeetingAI já traz um **Diarizador Fallback Acústico** que agrupa as vozes sem precisar de nenhuma chave.

Caso queira a precisão do Pyannote 3.1 oficial:
1. Crie uma conta gratuita no [Hugging Face](https://huggingface.co).
2. Aceite os termos do modelo: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
3. Crie um token de leitura em: [Configurações de Tokens](https://huggingface.co/settings/tokens).
4. Insira seu token no arquivo `.env`:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   ou preencha na aba **Ajustes** da interface web.

---

## 🧪 Verificação e Testes

Para validar a integridade de todos os componentes do pipeline (banco, alinhamento, exportações e sumarizador):

```bash
docker compose exec app python -m unittest tests/test_audio_pipeline.py
```

---

## 📡 Documentação da API REST

Quando a aplicação estiver rodando, a documentação Swagger interativa pode ser acessada em:
👉 **http://localhost:8000/docs**

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Envio de arquivos de áudio/vídeo |
| `POST` | `/api/process` | Dispara esteira de diarização, transcrição e ata |
| `GET` | `/api/jobs/{id}` | Consulta status da tarefa |
| `GET` | `/api/jobs/{id}/stream` | Atualização em tempo real via Server-Sent Events |
| `GET` | `/api/meetings` | Histórico de reuniões gravadas |
| `GET` | `/api/meetings/{id}` | Detalhes completos da reunião |
| `PUT` | `/api/meetings/{id}/speakers` | Renomeia interlocutores e atualiza a ata |
| `POST` | `/api/meetings/{id}/regenerate-summary` | Regera a ata com outro modelo de IA |
| `GET` | `/api/meetings/{id}/export/{format}` | Exporta para `.docx`, `.md` ou `.txt` |
| `GET` | `/api/audio/{filename}` | Streaming de áudio para o player |
| `GET` | `/api/models/ollama` | Lista modelos instalados no Ollama |
| `GET` | `/api/health` | Diagnóstico geral dos serviços |

---

## 🔒 Privacidade & Segurança

- **100% Local:** Todo o processamento de áudio, transcrição e inferência do LLM é executado estritamente na sua máquina dentro dos containers Docker.
- **Sem Telemetria:** Nenhuma gravação, voz ou texto de reunião sai da sua rede.
