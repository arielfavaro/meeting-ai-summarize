# 🎙️ MeetingAI - Transcrição, Diarização e Atas com IA 100% Local (PT-BR)

> **Privacidade Total:** Grave, transcreva, separe os interlocutores e gere Atas de Reunião executivas diretamente na sua máquina via **Docker**, sem enviar nenhum dado para a nuvem.

---

## ✨ Principais Funcionalidades

- **🎧 Entrada Flexível e Pipeline Multi-Faixa Inteligente:**
  - Envie gravações prévias em múltiplos formatos (`MP3`, `WAV`, `M4A`, `OGG`, `MP4`, `WEBM`).
  - **Isolamento de Canais Dedicados (OBS / Zoom / Meet):** Analisa e descarta automaticamente faixas inativas/mudas (-91 dB). Canais dedicados de microfone são processados isoladamente como 1 locutor garantido, enquanto a faixa compartilhada de desktop é diarizada para os demais participantes, intercalando todas as falas em ordem cronológica contínua.
  - **Grave ao vivo pelo navegador** com microfone integrado e visualização em tempo real das ondas sonoras.
- **👥 Separação de Interlocutores (Diarização Neural 100% Local e Offline):**
  - **Motor Neural SpeechBrain (ECAPA-TDNN):** Extração de assinaturas vocais densas de 192 dimensões aceleradas por GPU CUDA, sem requerer token ou cadastro no Hugging Face.
  - **Silero VAD Neural:** Detecção de fala humana profunda que descarta digitação, ar-condicionado e cliques.
  - **Suavização de Micro-Clusters:** Elimina locutores fantasmas causados por tosses ou interjeições rápidas.
  - **Inferência Automática de Nomes Reais:** O Ollama infere e sugere os nomes reais dos participantes diretamente das saudações no diálogo (*"Oi Ariel"*, *"Com certeza, Carlos"*).
  - **Renomeação e Reorganização:** Edite os nomes na interface com 1 clique e regere a ata a qualquer momento.
- **⚡ Transcrição Otimizada em Português (Faster-Whisper):**
  - Utiliza CTranslate2 com precisão `float16` na GPU CUDA ou `int8` na CPU.
  - **`initial_prompt` em PT-BR:** Pontuação, acentuação e termos técnicos corporativos refinados.
  - **`word_timestamps=True` e Alinhamento Cirúrgico:** Timestamps em nível de palavra para divisão exata na troca de oradores.
  - **Anti-Alucinação:** `condition_on_previous_text=False` e `hallucination_silence_threshold=2.0`, eliminando repetições em loop em reuniões longas.
- **📋 Geração de Atas de Reunião com LLM Local (Ollama):**
  - Integração nativa com **Ollama** (`qwen2.5:14b`, `gemma4:12b`, `llama3.1:8b`, etc.).
  - **Janela de Contexto Adaptativa (`OLLAMA_NUM_CTX`):** Dimensiona automaticamente a memória de 4.096 até 32.768+ tokens de acordo com o tamanho real da conversa, garantindo síntese completa de reuniões de até 2 horas e meia sem perda de conteúdo.
  - Produz Atas executivas completas:
    - **Resumo Executivo** estruturado.
    - **Tópicos Principais e Discussões** detalhadas.
    - **Decisões Tomadas** registradas formalmente.
    - **Matriz de Ações / Tarefas:** Responsável, Ação, Prazo e Status.
    - **Pontos em Aberto / Próximos Passos**.
  - **Gerador Estruturado de Contingência:** Caso o modelo do Ollama ainda não tenha sido baixado, a ata é gerada por análise heurística local sem travar.
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
│       ├── audio_service.py     # Conversão 16kHz mono com FFmpeg e amix multifaixa
│       ├── transcription.py     # Faster-Whisper 100% offline (PT-BR)
│       ├── diarization.py       # Motor de Diarização 100% Local (VAD + Biometria)
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

### 2. Baixar / Gerenciar Modelos do Ollama (LLM Local)

O sistema vem configurado por padrão com o **Gemma 4 (12B)**, modelo de última geração com alta inteligência executiva em português e 128k de contexto.

Você pode baixar e alternar modelos diretamente pela aba **⚙️ Ajustes** da interface web, ou pelo terminal:

```bash
# Modelo Padrão Ativo (Gemma 4 12B)
docker compose exec ollama ollama pull gemma4:12b

# Outras excelentes opções em português:
docker compose exec ollama ollama pull qwen2.5:14b   # Alta precisão em matriz de tarefas
docker compose exec ollama ollama pull gemma2:27b    # Topo de linha com síntese executiva profunda
docker compose exec ollama ollama pull llama3.1:8b    # Equilibrado e rápido
docker compose exec ollama ollama pull llama3.2:3b    # Leve para testes rápidos
```

*(Todos os modelos ficam salvos permanentemente no seu disco local no diretório persistente `./data/ollama`).*

---

## 🎮 Aceleração por Placa de Vídeo (NVIDIA GPU / CUDA)

Se sua máquina possui uma GPU NVIDIA com drivers e o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) instalados, inicie com o arquivo de aceleração:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```
Isso habilitará automaticamente:
* **Faster-Whisper:** Execução com `WHISPER_DEVICE=cuda` e precisão `float16` nos Tensor Cores da GPU.
* **Ollama (LLM):** Aceleração GPU completa, carregando os pesos da rede neural na VRAM para geração quase instantânea da ata.
* **Diarização Neural (SpeechBrain ECAPA-TDNN):** Extração de assinaturas biométricas de voz processadas em lotes acelerados por CUDA (`cuda:0`).

---

## ⚙️ Variáveis de Ambiente e Configurações (`.env`)

Todas as opções do MeetingAI são configuradas centralizadamente via variáveis de ambiente no arquivo `.env` (criado a partir de `.env.example`).

### Tabela Geral de Parâmetros

| Variável | Padrão | Valores Suportados | Descrição |
| :--- | :--- | :--- | :--- |
| `OLLAMA_NUM_CTX` | `32768` | `4096` a `131072` | **Janela máxima de contexto do LLM em tokens.** Essencial para reuniões longas. |
| `OLLAMA_MODEL` | `gemma4:12b` | `gemma4:12b`, `qwen2.5:14b`, `llama3.1:8b`, etc. | Modelo de LLM local utilizado para sumarizar e extrair a ata de reunião. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL válida do Ollama | Endereço do serviço Ollama (no Docker ou host nativo). |
| `WHISPER_MODEL_SIZE` | `large-v3` | `tiny`, `base`, `small`, `medium`, `large-v3` | Tamanho do modelo Whisper. `medium` e `large-v3` são recomendados para PT-BR. |
| `WHISPER_DEVICE` | `cpu` (ou `cuda`) | `cpu`, `cuda` | Dispositivo de hardware para a transcrição. |
| `WHISPER_COMPUTE_TYPE`| `int8` (ou `float16`)| `int8`, `float16`, `float32` | Precisão numérica: `float16` para GPU, `int8` para CPU. |
| `DEFAULT_LANGUAGE` | `pt` | Código ISO (`pt`, `en`, `es`, `auto`) | Idioma padrão das reuniões. |
| `MAX_FILE_SIZE_MB` | `500` | Inteiro (MB) | Limite de tamanho de arquivo aceito no upload. |
| `ENABLE_PYANNOTE` | `false` | `true`, `false` | Ativa o pipeline Pyannote legado se um `HF_TOKEN` for fornecido. |
| `HF_TOKEN` | `""` | Token Hugging Face | Token opcional do Hugging Face (desnecessário para o SpeechBrain). |

---

### 🧠 O que é o `OLLAMA_NUM_CTX` e como ajustar?

O parâmetro `OLLAMA_NUM_CTX` define o **tamanho máximo da janela de contexto** (em tokens) que o Ollama reserva na memória ao processar a transcrição da reunião.

#### 1. Por que esse parâmetro é crucial?
* Uma gravação de **1 hora e 20 minutos** gera entre 12.000 e 20.000 palavras transcritas em português (~16.000 a 25.000 tokens).
* Por padrão, o Ollama original utiliza apenas **2.048 ou 4.096 tokens**. Se o contexto não for ampliado, o modelo simplesmente corta o texto e ignora a segunda metade da reunião!
* Com `OLLAMA_NUM_CTX=32768`, o modelo lê **100% da reunião**, garantindo que todas as decisões, tarefas e discussões até o último minuto sejam incluídas na ata.

#### 2. Como o MeetingAI dimensiona o contexto de forma inteligente?
O MeetingAI implementa um **dimensionamento adaptativo**:
$$\text{num\_ctx} = \min(\text{OLLAMA\_NUM\_CTX}, \max(4096, \text{tokens\_estimados} + 2500))$$
* **Reuniões curtas (5 a 15 min):** O sistema aloca apenas 4.096 tokens, economizando VRAM e gerando a ata em segundos.
* **Reuniões longas (1h a 2h30):** O sistema expande a janela dinamicamente até o limite configurado em `OLLAMA_NUM_CTX` (32.768).

#### 3. Guia de Ajuste por Hardware / Placa de Vídeo (VRAM)

Você pode ajustar `OLLAMA_NUM_CTX` no seu arquivo `.env` de acordo com a sua GPU:

| VRAM Disponível | Modelo Recomendado | `OLLAMA_NUM_CTX` | Duração Máxima de Reunião |
| :--- | :--- | :--- | :--- |
| **8 GB VRAM ou CPU** | `llama3.2:3b` / `qwen2.5:7b` | `8192` | Reuniões de até 35 a 45 minutos |
| **12 GB VRAM** (ex: RTX 3060) | `qwen2.5:7b` / `gemma4:12b` | `16384` | Reuniões de até 1h15m |
| **16 GB VRAM** (ex: RTX 4060 Ti / 5060 Ti) | `qwen2.5:14b` / `gemma4:12b` | `32768` *(Padrão)* | Reuniões de até 2h30m |
| **24 GB VRAM** (ex: RTX 3090 / 4090) | `qwen2.5:14b` / `gemma2:27b` | `65536` | Reuniões de até 5 horas |

---

## 🎙️ Diarização Neural 100% Local & Offline

O MeetingAI utiliza um pipeline moderno em cascata:
1. **SpeechBrain ECAPA-TDNN:** Rede neural de ponta com extração de 192 embeddings vocais invariantes a volume e ruído.
2. **Silero VAD Neural:** Detecção de atividade vocal que ignora respirações, estalos e ruídos ambientes.
3. **Pós-processamento de Micro-Clusters:** Suaviza fatias curtas espúrias para evitar "locutores fantasmas".
4. **Isolamento Multi-Faixa:** Em vídeos com múltiplos canais (OBS Studio), o canal de microfone do apresentador é isolado como 1 orador único, e a chamada desktop é diarizada separadamente para os demais participantes.

Não é necessário criar conta no Hugging Face nem informar tokens externos para usufruir da separação completa de oradores.

---

## 🧪 Verificação e Testes Automatizados

Para executar os testes de validação do pipeline de áudio e das otimizações de precisão:

```bash
# Executar suíte completa de testes unitários e de integração
docker compose exec app python -m unittest discover tests

# Executar teste específico das novas otimizações (Silero VAD, word_timestamps e alinhamento cirúrgico)
docker compose exec app python /app/tests/test_improvements.py
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
