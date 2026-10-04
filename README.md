# 🎙️ MeetingAI - Transcrição, Diarização e Atas com IA 100% Local (PT-BR)

> **Privacidade Total:** Grave, transcreva, separe os interlocutores e gere Atas de Reunião executivas diretamente na sua máquina via **Docker**, sem enviar nenhum dado para a nuvem.

---

## ✨ Principais Funcionalidades

- **🎧 Entrada Flexível e Pipeline Multi-Faixa Inteligente:**
  - Envie gravações prévias em múltiplos formatos (`MP3`, `WAV`, `M4A`, `OGG`, `MP4`, `WEBM`).
  - **Canal de microfone = 1 locutor e eco removido:** em gravações multi-faixa, a faixa dominada por uma voz é tratada como um locutor só (ruído e vazamento não viram "locutores fantasmas"), e falas que aparecem em duas faixas ao mesmo tempo (microfone captando o alto-falante) são deduplicadas.
  - **Isolamento de Canais Dedicados (OBS / Zoom / Meet):** Analisa e descarta automaticamente faixas inativas/mudas (-91 dB). Canais dedicados de microfone são processados isoladamente como 1 locutor garantido, enquanto a faixa compartilhada de desktop é diarizada para os demais participantes, intercalando todas as falas em ordem cronológica contínua.
  - **Grave ao vivo pelo navegador** com microfone integrado e visualização em tempo real das ondas sonoras.
- **👥 Separação de Interlocutores (Diarização Neural 100% Local e Offline):**
  - **pyannote community-1 (padrão quando baixado):** Segmentação neural com detecção de fala sobreposta e saída "exclusiva" (um locutor por instante) pensada para casar com o Whisper. Carregado de diretório local, com telemetria desligada.
  - **Motores de reserva:** SpeechBrain ECAPA-TDNN e motor acústico (MFCC + pitch), escolhidos automaticamente se o anterior não estiver disponível.
  - **Atribuição por Frase:** O locutor é decidido por frase (voto ponderado pela duração), não palavra a palavra — uma palavra na borda de um turno não parte mais a frase. Interrupções reais (várias palavras, mais de 1 s) continuam separadas.
  - **Correção Manual:** Mesclar locutores, mover um trecho para outro locutor (ou para um novo) e refazer a separação informando o número de pessoas — a transcrição é reaproveitada e os nomes confirmados são mantidos.
  - **Silero VAD Neural:** Detecção de fala humana profunda que descarta digitação, ar-condicionado e cliques.
  - **Suavização de Micro-Clusters:** Elimina locutores fantasmas causados por tosses ou interjeições rápidas.
  - **Inferência de Nomes com Evidência:** O LLM sugere nomes (auto-apresentação: *"aqui é a Mariana"*; ou chamado pelo nome: *"Carlos, pode falar?"* seguido da resposta do Carlos). As sugestões só são aplicadas após validação determinística (nome presente na transcrição, quem é chamado é quem responde, confiança mínima, nomes únicos). Sugestões fracas aparecem na interface para você confirmar.
  - **IDs Estáveis de Locutor:** A transcrição e a ata guardam o rótulo (`Locutor 2`); os nomes são aplicados na exibição. Renomear atualiza transcrição, ata e exportações na hora, sem chamar o LLM, e nomes definidos por você nunca são sobrescritos pela IA.
- **⚡ Transcrição Otimizada em Português (Faster-Whisper):**
  - Utiliza CTranslate2 com precisão `float16` na GPU CUDA ou `int8` na CPU.
  - **`initial_prompt` em PT-BR:** Pontuação, acentuação e termos técnicos corporativos refinados.
  - **`word_timestamps=True` e Alinhamento Cirúrgico:** Timestamps em nível de palavra para divisão exata na troca de oradores.
  - **Anti-Alucinação:** `condition_on_previous_text=False` e `hallucination_silence_threshold=2.0`, eliminando repetições em loop em reuniões longas.
- **📋 Geração de Atas de Reunião com LLM Local (Ollama):**
  - Integração nativa com **Ollama** (`qwen2.5:14b`, `gemma4:12b`, `llama3.1:8b`, etc.).
  - **Contexto da Reunião:** Objetivo/pauta, tipo de reunião (daily, planejamento, retrospectiva, 1:1, comercial, técnica...), participantes esperados e glossário. Nomes e termos também alimentam o `initial_prompt` do Whisper.
  - **Saída Estruturada (JSON Schema):** O esquema Pydantic é enviado no `format` do Ollama (geração restrita por gramática) e validado na volta, com nova tentativa informando o erro ao modelo.
  - **Evidências:** Cada objetivo, decisão, tarefa e ponto em aberto aponta os trechos da transcrição (`#ID`) que o sustentam — clique no horário para ouvir. Itens sem evidência são sinalizados.
  - **Sem Retrabalho do LLM:** a resposta chega em streaming; se o modelo entra em laço (espaços sem fim ou repetição), a geração é abortada em segundos, o JSON já gerado é aproveitado quando tem conteúdo útil e a nova tentativa usa penalidade de repetição. Um bloco problemático fica de fora com aviso, sem derrubar a ata; se a consolidação falhar, a ata é montada a partir dos blocos.
  - **Janela Planejada Antes da Chamada + Map-Reduce:** o schema da ata tem limites de itens e caracteres, então o tamanho máximo da resposta é conhecido de antemão; `num_ctx` = prompt + pior caso da resposta (em degraus de 2048, estável entre tentativas e blocos para o Ollama não recarregar o modelo). Reuniões que não cabem em `OLLAMA_NUM_CTX` são analisadas em blocos e consolidadas numa ata única.
  - Produz Atas executivas completas:
    - **Objetivos da Reunião** com status (atingido / parcial / não atingido).
    - **Resumo Executivo** estruturado.
    - **Tópicos Principais e Discussões** detalhadas.
    - **Decisões Tomadas** registradas formalmente.
    - **Matriz de Ações / Tarefas:** Responsável, Ação, Prazo (com data absoluta: *"sexta-feira"* → `09/10/2026`) e Status.
    - **Pontos em Aberto / Próximos Passos** e **Riscos / Impedimentos**.
  - **Gerador de Contingência Transparente:** Sem LLM disponível, a ata é gerada por palavras-chave e marcada como contingência na interface e nas exportações.
- **📄 Exportação Profissional:**
  - Exportação em **Word (.docx)** com formatação corporativa e tabela de tarefas.
  - Exportação em **Markdown (.md)** com formatação GitHub.
  - Exportação em **Texto puro (.txt)** e visualização pronta para **Impressão / PDF**.
- **🎯 Player de Áudio Interativo e Sincronizado:**
  - Clique em qualquer balão de fala da transcrição para pular o áudio diretamente para o segundo exato (`timestamp`).
  - Velocidades de reprodução de `1x`, `1.25x` e `1.5x`.
- **🗄️ Histórico Completo:**
  - Reuniões salvas localmente em banco SQLite para consulta e re-exportação a qualquer momento.
  - Excluir uma reunião apaga também o áudio, as faixas e os checkpoints dela no disco.
- **🛟 Robustez:**
  - Processamentos gravados no banco: recarregar a página reconecta ao andamento; um reinício do servidor marca o job como interrompido, com botão **Tentar novamente**.
  - **Checkpoints por etapa** (faixas, diarização, transcrição): tentar novamente ou re-diarizar não refaz o que já foi concluído.
  - Em GPU, diarização e Whisper são liberados da VRAM antes da ata (Ollama), evitando estouro de memória.

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
│   ├── dependencies.py          # Composition root (injeção de dependências via Depends)
│   ├── config.py                # Configurações centralizadas (pydantic-settings)
│   ├── database.py              # Repositório SQLite com migrações versionadas
│   ├── models/schemas.py        # Domínio e DTOs (Pydantic)
│   ├── prompts/                 # Prompts versionados da ata (+ orientações por tipo de reunião)
│   ├── scripts/download_models.py  # Download único dos modelos (depois tudo roda offline)
│   └── services/
│       ├── pipeline.py          # Orquestração do processamento (não bloqueia o event loop)
│       ├── jobs.py              # Jobs em memória + SQLite, log em tempo real e fila (semáforo)
│       ├── file_store.py        # Upload em streaming e validação de caminhos
│       ├── audio_service.py     # Conversão 16kHz mono com FFmpeg e amix multifaixa
│       ├── transcription.py     # Faster-Whisper 100% offline (PT-BR)
│       ├── diarization/         # Motores plugáveis: pyannote → SpeechBrain → acústico (offline)
│       ├── artifacts.py         # Checkpoints das etapas pesadas (retomar / re-diarizar)
│       ├── speaker_editing.py   # Mesclar, reatribuir trechos, manter nomes ao re-diarizar
│       ├── alignment.py         # Fusão fala x orador
│       ├── speakers.py          # IDs estáveis x nomes de exibição (apresentação)
│       ├── exporter.py          # Gerador DOCX, Markdown e TXT
│       ├── llm/                 # Contrato LLMClient + cliente Ollama
│       └── minutes/             # Ata: gerador (single-pass/map-reduce), verificação,
│                                #      prazos, nomes de locutores, renderização
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
└── tests/                       # Testes (rodam sem modelos de IA, usando dublês)
    ├── test_minutes.py          # Ata estruturada, map-reduce, retries, prazos, nomes
    ├── test_api.py              # Segurança de arquivos, renomeação, migração de dados legados
    ├── test_pipeline.py         # Pipeline ponta a ponta, fila e event loop
    ├── test_alignment.py        # Atribuição de locutor por frase
    ├── test_audio_pipeline.py   # Alinhamento, exportações, diarização local
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
| `WHISPER_MODEL_SIZE` | `medium` | `tiny`, `base`, `small`, `medium`, `large-v3` | Tamanho do modelo Whisper. `medium` e `large-v3` são recomendados para PT-BR. |
| `WHISPER_DEVICE` | `cpu` (ou `cuda`) | `cpu`, `cuda` | Dispositivo de hardware para a transcrição. |
| `WHISPER_COMPUTE_TYPE`| `int8` (ou `float16`)| `int8`, `float16`, `float32` | Precisão numérica: `float16` para GPU, `int8` para CPU. |
| `DEFAULT_LANGUAGE` | `pt` | Código ISO (`pt`, `en`, `es`, `auto`) | Idioma padrão das reuniões. |
| `MAX_FILE_SIZE_MB` | `0` | Inteiro (MB) | Limite de tamanho de arquivo aceito no upload. `0` = sem limite (padrão). |
| `OLLAMA_NUM_PREDICT` | `4096` | Inteiro | Piso de tokens para a resposta do LLM. O valor efetivo é o pior caso do schema da ata (limites de itens e caracteres), calculado antes da chamada. |
| `MINUTES_CHUNK_TOKENS` | `6000` | Inteiro | Tamanho de cada bloco quando a reunião não cabe em `OLLAMA_NUM_CTX` (map-reduce). |
| `SPEAKER_NAME_MIN_CONFIDENCE` | `0.75` | `0` a `1` | Confiança mínima para aplicar automaticamente um nome inferido. |
| `MAX_CONCURRENT_JOBS` | `1` | Inteiro | Processamentos pesados simultâneos (os demais aguardam na fila). |
| `CORS_ORIGINS` | `["*"]` | Lista JSON | Origens permitidas para chamadas à API. |
| `DIARIZATION_ENGINE` | `auto` | `auto`, `pyannote`, `speechbrain`, `acoustic` | Motor de diarização preferido (os seguintes servem de reserva). |
| `HF_HUB_OFFLINE` | `1` | `0`, `1` | `1` = modelos só do disco, sem rede. Use `0` apenas no comando de download. |
| `HF_TOKEN` | `""` | Token Hugging Face | Usado **somente** pelo script de download do pyannote; nunca em processamento. |
| `OLLAMA_KEEP_ALIVE` | `5m` | Duração | Tempo que o Ollama mantém o modelo carregado após a ata. |
| `RELEASE_MODELS_AFTER_USE` | automático | `true`, `false` | Libera diarização/Whisper da memória entre etapas (automático: só com CUDA). |
| `MULTITRACK_DOMINANT_SHARE` | `0.8` | `0` a `1` | Em gravações multi-faixa, faixa em que uma voz responde por essa fração da fala vira 1 locutor (canal de microfone). `0` desativa. |
| `MULTITRACK_DEDUPE_ECHO` | `true` | `true`, `false` | Remove falas duplicadas entre faixas (microfone captando o alto-falante). |
| `APP_TIMEZONE` | `America/Sao_Paulo` | Fuso IANA | Fuso dos logs, da data da ata e dos prazos relativos ("amanhã", "sexta"). |

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

## 📦 Modelos 100% Offline (download único)

Por padrão o container roda com `HF_HUB_OFFLINE=1`: os modelos são lidos **somente** do disco (`./data/models_cache`) e nenhuma biblioteca acessa a rede. Para baixar os modelos uma única vez:

```bash
# Whisper + SpeechBrain (públicos)
docker compose run --rm -e HF_HUB_OFFLINE=0 app python -m backend.scripts.download_models --whisper medium

# + pyannote community-1 (gratuito; aceite os termos em
#   https://huggingface.co/pyannote/speaker-diarization-community-1 e use um token de leitura)
docker compose run --rm -e HF_HUB_OFFLINE=0 app python -m backend.scripts.download_models --whisper medium,large-v3 --hf-token hf_xxx
```

O token é usado só nesse comando e não é gravado. Depois disso, tudo roda offline. Se um modelo escolhido na interface não estiver no disco, o processamento falha com a mensagem de como baixá-lo (nada é baixado em segundo plano). `GET /api/diarization/engines` mostra quais motores de diarização estão disponíveis.

### 🔐 pyannote: por que pede para aceitar termos? Ele é obrigatório?

- **Não é obrigatório.** Sem o pyannote, a aplicação usa automaticamente o **SpeechBrain** (público, sem termos, também 100% offline). O pyannote só melhora a separação de locutores: detecta fala sobreposta, conta melhor as pessoas e entrega um locutor por instante, o que casa melhor com a transcrição do Whisper.
- **O termo vale só para o download, não para o uso.** Os autores (que também mantêm o serviço comercial pyannoteAI) publicam o modelo como *gated* no Hugging Face: para baixar os arquivos é preciso estar logado e aceitar compartilhar nome/empresa com eles. A licença do modelo é **CC-BY-4.0** (aberta, permite uso comercial, exige apenas atribuição aos autores).
- **Em processamento é 100% offline:** o pipeline é carregado de `data/models_cache/pyannote-speaker-diarization-community-1`, sem token e sem rede. A telemetria opcional do pyannote 4 fica desligada (`PYANNOTE_METRICS_ENABLED=0` no `docker-compose.yml` e `set_telemetry_metrics(False)` no código), e o container roda com `HF_HUB_OFFLINE=1`.

#### Opção A — pelo script (usa o token apenas no download)

1. Crie uma conta em [huggingface.co](https://huggingface.co) e aceite os termos em [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Gere um token de **leitura** em *Settings → Access Tokens*.
3. Rode:
   ```bash
   docker compose run --rm -e HF_HUB_OFFLINE=0 app python -m backend.scripts.download_models --whisper medium --hf-token hf_xxx
   ```

#### Opção B — sem colocar token na aplicação

1. Aceite os termos pelo navegador (passo 1 acima).
2. Em qualquer máquina com internet, clone o repositório oficial (pede o usuário e um token do Hugging Face como senha):
   ```bash
   git lfs install
   git clone https://huggingface.co/pyannote/speaker-diarization-community-1
   ```
3. Copie a pasta para `data/models_cache/pyannote-speaker-diarization-community-1` (precisa conter o `config.yaml`).
4. Reinicie o app (`docker compose restart app`). Com `DIARIZATION_ENGINE=auto`, o pyannote passa a ser usado; o log do processamento mostra o motor escolhido.

> ⚠️ Existem cópias do modelo **sem** o termo publicadas por terceiros no Hugging Face (a licença permite redistribuição). Não recomendamos usá-las: não são oficiais e um arquivo de modelo adulterado pode executar código ao ser carregado. Use sempre o repositório oficial `pyannote/speaker-diarization-community-1`.

---

## 🎙️ Diarização Neural 100% Local & Offline

Motores plugáveis, escolhidos por `DIARIZATION_ENGINE` (padrão `auto`, na ordem abaixo; se um não estiver disponível ou falhar, o próximo assume):
1. **pyannote `speaker-diarization-community-1`:** segmentação neural com fala sobreposta, contagem de locutores e saída exclusiva (um locutor por instante). Telemetria desligada (`PYANNOTE_METRICS_ENABLED=0`).
2. **SpeechBrain ECAPA-TDNN + Silero VAD:** embeddings de 192 dimensões e agrupamento hierárquico.
3. **Motor acústico:** MFCC + pitch + clustering, sem modelos baixados.

Depois da diarização, o **alinhamento por frase** decide o locutor de cada frase pelo voto ponderado das palavras. **Isolamento multi-faixa:** em gravações com várias faixas (OBS Studio), cada faixa é diarizada separadamente e as falas são intercaladas.

---

## 🧪 Verificação e Testes Automatizados

Para executar os testes de validação do pipeline de áudio e das otimizações de precisão:

```bash
# Executar suíte completa de testes unitários e de integração
docker compose exec app python -m unittest discover tests

# Executar só os testes da ata estruturada (não precisam de GPU nem de modelos baixados)
docker compose exec app python -m unittest discover tests -p "test_minutes.py"
```

---

## 📡 Documentação da API REST

Quando a aplicação estiver rodando, a documentação Swagger interativa pode ser acessada em:
👉 **http://localhost:8000/docs**

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Envio de arquivos de áudio/vídeo |
| `POST` | `/api/process` | Dispara esteira de diarização, transcrição e ata (aceita `objective`, `meeting_type`, `participants`, `glossary`) |
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
| `GET` | `/api/jobs/active` | Processamento em andamento (para reconectar a interface) |
| `POST` | `/api/jobs/{id}/retry` | Refaz um processamento que falhou, reaproveitando etapas concluídas |
| `POST` | `/api/meetings/{id}/rediarize` | Refaz a separação de locutores (ex.: `{"num_speakers": 3}`) reaproveitando a transcrição |
| `POST` | `/api/meetings/{id}/speakers/merge` | Mescla locutores (`source_ids` → `target_id`) |
| `POST` | `/api/meetings/{id}/segments/reassign` | Move trechos para outro locutor ou para um novo |
| `GET` | `/api/diarization/engines` | Motores de diarização disponíveis e motivo dos indisponíveis |

---

## 🔒 Privacidade & Segurança

- **100% Local:** Todo o processamento de áudio, transcrição e inferência do LLM é executado estritamente na sua máquina dentro dos containers Docker.
- **Sem Telemetria:** Nenhuma gravação, voz ou texto de reunião sai da sua rede. Telemetria do Hugging Face e do pyannote desligada; a interface não carrega fontes nem scripts de CDNs externos.
- **Modo Offline Estrito:** `HF_HUB_OFFLINE=1` por padrão; modelos só do disco local.
