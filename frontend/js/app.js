/**
 * APLICAÇÃO PRINCIPAL - CONTROLE DE ESTADO, ABAS E FLUXO DE PROCESSAMENTO
 */

/** Escapa texto antes de interpolar em HTML (conteúdo vem do LLM, da transcrição e do usuário). */
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** ISO 8601 -> "dd/mm/aaaa hh:mm". Valores legados (já formatados) são devolvidos como estão. */
function formatDateTime(value) {
  if (!value) return "--";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
}

function formatDateOnly(value) {
  if (!value) return "";
  const d = new Date(`${value}T12:00:00`);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleDateString("pt-BR");
}

const OBJECTIVE_STATUS = {
  atingido: "✅ Atingido",
  parcial: "🟡 Parcial",
  nao_atingido: "❌ Não atingido",
  indefinido: "⚪ Indefinido",
};
const OBJECTIVE_ORIGIN = { declarado: "declarado na reunião", inferido: "inferido", informado: "informado por você" };

class MeetingApp {
  constructor() {
    this.selectedFile = null;
    this.currentMeeting = null;
    this.activeJobId = null;
    this.eventSource = null;
    this.speakerColors = [
      "#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#f97316"
    ];

    this._initElements();
    this._setupDropzone();
    this._checkHealth();
    this.loadMeetingsHistory();
    this._resumeActiveJob();
  }

  /** Reconecta a um processamento em andamento (ex.: página recarregada no meio da transcrição). */
  async _resumeActiveJob() {
    try {
      const resp = await fetch("/api/jobs/active");
      if (!resp.ok) return;
      const job = await resp.json();
      if (!job || !job.job_id) return;
      this.switchTab("processing");
      this._resetProgressUI();
      this._appendLog("Reconectado ao processamento em andamento.", "success");
      this.activeJobId = job.job_id;
      this._updateProgressUI(job);
      this._listenToJobEvents(job.job_id);
    } catch (e) {
      console.warn("Não foi possível verificar processamentos em andamento:", e);
    }
  }

  _initElements() {
    this.fileInput = document.getElementById("file-input");
    this.dropzone = document.getElementById("audio-dropzone");
    this.fileSelectedCard = document.getElementById("file-selected-card");
    this.selectedFileName = document.getElementById("selected-file-name");
    this.progressFill = document.getElementById("job-progress-fill");
    this.currentStepLabel = document.getElementById("job-current-step-label");
    this.logBox = document.getElementById("job-log-box");
  }

  switchTab(tabName) {
    document.querySelectorAll(".tab-panel").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll(".nav-tab-btn").forEach((el) => el.classList.remove("active"));

    const targetPanel = document.getElementById(`tab-${tabName}`);
    const targetBtn = document.getElementById(`btn-tab-${tabName}`);

    if (targetPanel) targetPanel.classList.add("active");
    if (targetBtn) targetBtn.classList.add("active");

    if (tabName === "history") {
      this.loadMeetingsHistory();
    }
  }

  _setupDropzone() {
    if (!this.dropzone) return;

    ["dragenter", "dragover"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.remove("dragover");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this._handleSelectedFile(e.dataTransfer.files[0]);
      }
    });

    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this._handleSelectedFile(e.target.files[0]);
      }
    });
  }

  _handleSelectedFile(file) {
    this.selectedFile = file;
    this.selectedFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    this.fileSelectedCard.style.display = "flex";
    this.dropzone.style.display = "none";

    const titleInput = document.getElementById("input-meeting-title");
    if (!titleInput.value) {
      const cleanName = file.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, " ");
      titleInput.value = cleanName;
    }
  }

  setRecordedAudio(blob, filename) {
    const file = new File([blob], filename, { type: blob.type });
    this._handleSelectedFile(file);
    this.showToast("Áudio gravado pronto para processar!");
  }

  clearSelectedFile() {
    this.selectedFile = null;
    this.fileInput.value = "";
    this.fileSelectedCard.style.display = "none";
    this.dropzone.style.display = "flex";
  }

  async _checkHealth() {
    await settingsManager.testOllamaConnection();
  }

  async startProcessing() {
    if (!this.selectedFile) {
      alert("Por favor, selecione um arquivo de áudio ou faça uma gravação pelo microfone.");
      return;
    }

    try {
      this.switchTab("processing");
      this._resetProgressUI();
      this._appendLog(`Iniciando upload de '${this.selectedFile.name}'...`);

      // 1. Upload do Arquivo
      const formData = new FormData();
      formData.append("file", this.selectedFile);

      const uploadResp = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResp.ok) {
        const err = await uploadResp.json().catch(() => ({}));
        throw new Error(err.detail || "Falha no upload do arquivo.");
      }
      const uploadData = await uploadResp.json();
      this._appendLog(`Upload concluído! ID: ${uploadData.file_id} (Duração: ${uploadData.duration.toFixed(1)}s)`);

      // 2. Iniciar Processamento
      const title = document.getElementById("input-meeting-title").value.trim();
      const whisperModel = document.getElementById("select-whisper-model").value;
      const ollamaModel = document.getElementById("select-ollama-model").value;
      const minSpeakers = document.getElementById("input-num-speakers").value;
      const customPrompt = document.getElementById("input-custom-prompt").value.trim();
      const hfToken = settingsManager.getHfToken();

      const processForm = new FormData();
      processForm.append("file_id", uploadData.file_id);
      processForm.append("title", title || "Reunião");
      processForm.append("whisper_model", whisperModel);
      processForm.append("language", "pt");
      if (ollamaModel) processForm.append("ollama_model", ollamaModel);
      if (minSpeakers) {
        processForm.append("min_speakers", minSpeakers);
        processForm.append("max_speakers", minSpeakers);
      }
      if (customPrompt) processForm.append("custom_prompt", customPrompt);
      if (hfToken) processForm.append("hf_token", hfToken);

      // Contexto da reunião
      const objective = document.getElementById("input-meeting-objective").value.trim();
      const participants = document.getElementById("input-participants").value.trim();
      const glossary = document.getElementById("input-glossary").value.trim();
      processForm.append("meeting_type", document.getElementById("select-meeting-type").value || "geral");
      if (objective) processForm.append("objective", objective);
      if (participants) processForm.append("participants", participants);
      if (glossary) processForm.append("glossary", glossary);

      const startResp = await fetch("/api/process", {
        method: "POST",
        body: processForm,
      });

      if (!startResp.ok) {
        const err = await startResp.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "Erro ao disparar pipeline.");
      }
      const startData = await startResp.json();
      this.activeJobId = startData.job_id;

      this._listenToJobEvents(startData.job_id);
    } catch (e) {
      console.error(e);
      alert(`Erro: ${e.message}`);
      this.switchTab("new");
    }
  }

  _listenToJobEvents(jobId) {
    if (this.eventSource) this.eventSource.close();
    this.activeJobId = jobId;
    this._hideJobActions();

    this.eventSource = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);
    this.eventSource.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data);
        this._updateProgressUI(job);
        if (job.status === "completed" || job.status === "failed") {
          this.eventSource.close();
          this._onJobFinished(job);
        }
      } catch (err) {
        console.error("Erro no SSE:", err);
      }
    };

    this.eventSource.onerror = () => {
      // Fallback para polling a cada 2s se SSE falhar
      this.eventSource.close();
      this._pollJobStatus(jobId);
    };
  }

  async _pollJobStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        if (!res.ok) return;
        const job = await res.json();
        this._updateProgressUI(job);
        if (job.status === "completed" || job.status === "failed") {
          clearInterval(interval);
          this._onJobFinished(job);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 2000);
  }

  async _onJobFinished(job) {
    if (job.status === "failed") {
      this._showJobActions(job);
      this.showToast("Falha no processamento. Veja o log e use \"Tentar novamente\".", 5000);
      return;
    }
    let meeting = job.result;
    if (!meeting && job.meeting_id) {
      // Job concluído enquanto a página estava fechada: o resultado vem do banco
      const resp = await fetch(`/api/meetings/${encodeURIComponent(job.meeting_id)}`);
      if (resp.ok) meeting = await resp.json();
    }
    if (!meeting) return;
    this.showToast(job.kind === "rediarize" ? "🎉 Separação de locutores refeita e ata atualizada!"
                                            : "🎉 Reunião processada e Ata gerada com sucesso!");
    this._renderMeetingData(meeting);
    setTimeout(() => this.switchTab(job.kind === "rediarize" ? "transcript" : "minutes"), 600);
  }

  _showJobActions(job) {
    const box = document.getElementById("job-actions");
    if (!box) return;
    box.style.display = "flex";
    document.getElementById("job-error-text").textContent = job.error || "Erro desconhecido.";
    const btn = document.getElementById("btn-retry-job");
    btn.disabled = !job.request;
    btn.onclick = () => this.retryJob(job.job_id);
  }

  _hideJobActions() {
    const box = document.getElementById("job-actions");
    if (box) box.style.display = "none";
  }

  /** Refaz um job que falhou; as etapas já concluídas (faixas, diarização, transcrição) são reaproveitadas. */
  async retryJob(jobId) {
    try {
      const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/retry`, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Não foi possível tentar novamente.");
      this._resetProgressUI();
      this._appendLog("Retomando o processamento (etapas concluídas serão reaproveitadas)...", "info");
      this._listenToJobEvents(data.job_id);
    } catch (e) {
      alert(`Erro: ${e.message}`);
    }
  }

  _resetProgressUI() {
    this.progressFill.style.width = "0%";
    this.currentStepLabel.textContent = "Preparando modelos locais...";
    this.logBox.innerHTML = "";
    this._hideJobActions();
    this.lastLogSeq = 0;
    this.lastStepIdx = 1;
    this.lastRenderedStep = "";
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step-${i}`);
      if (step) step.className = "step-item";
    }
  }

  _updateProgressUI(job) {
    this.progressFill.style.width = `${job.progress}%`;
    if (job.current_step) {
      this.currentStepLabel.textContent = job.current_step;
    }

    // Log em tempo real do backend: renderiza só as entradas novas (seq > última vista).
    // O backend mantém as últimas 150 linhas, por isso usamos a sequência e não o tamanho da lista.
    if (Array.isArray(job.logs) && job.logs.length > 0) {
      job.logs
        .filter((entry) => entry.seq > (this.lastLogSeq || 0))
        .forEach((entry) => {
          // Horário do navegador (o servidor pode estar em outro fuso, ex.: container em UTC)
          const time = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString("pt-BR") : entry.time;
          this._appendLogLine(`[${time}] ${entry.message}`, entry.level);
          this.lastLogSeq = entry.seq;
        });
    } else if (job.current_step && job.current_step !== this.lastRenderedStep) {
      this._appendLog(`[Progresso ${job.progress}%] ${job.current_step}`);
      this.lastRenderedStep = job.current_step;
    }

    const statusMap = {
      preprocessing: 1,
      diarizing: 2,
      transcribing: 3,
      aligning: 4,
      summarizing: 5,
      completed: 6,
    };

    // Em falha, destaca a etapa em que o processamento parou
    const currentStepIdx = job.status === "failed" ? this.lastStepIdx || 1 : statusMap[job.status] || 1;
    this.lastStepIdx = currentStepIdx;
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step-${i}`);
      if (!step) continue;
      if (i < currentStepIdx) {
        step.className = "step-item completed";
      } else if (i === currentStepIdx) {
        step.className = job.status === "failed" ? "step-item failed" : "step-item active";
      } else {
        step.className = "step-item";
      }
    }
  }

  _appendLog(text, level = "info") {
    const time = new Date().toLocaleTimeString("pt-BR");
    this._appendLogLine(`[${time}] ${text}`, level);
  }

  /** Linha do log com cor pelo nível informado pelo backend (info | success | warning | error). */
  _appendLogLine(text, level = "info") {
    if (!this.logBox) return;
    const line = document.createElement("div");
    const safeLevel = ["info", "success", "warning", "error"].includes(level) ? level : "info";
    line.className = `log-line log-${safeLevel}`;
    line.textContent = String(text);
    this.logBox.appendChild(line);
    this.logBox.scrollTop = this.logBox.scrollHeight;
  }

  _renderMeetingData(meeting) {
    this.currentMeeting = meeting;
    this.segmentStartById = new Map(meeting.segments.map((s) => [s.id, s.start]));
    // Cores e ordem dos locutores seguem o ID estável (não mudam ao renomear)
    this.speakerOrder = Array.from(new Set(meeting.segments.map((s) => s.speaker_id || s.speaker)));

    const s = meeting.summary;
    if (s) {
      document.getElementById("minutes-title").textContent = s.title;
      document.getElementById("minutes-date").textContent = `📅 Data: ${formatDateTime(s.date || meeting.created_at)}`;
      document.getElementById("minutes-duration").textContent = `⏱️ Duração: ${(meeting.audio_duration / 60).toFixed(1)} min`;
      document.getElementById("minutes-participants").textContent = `👥 Participantes: ${(s.participants || []).join(", ")}`;
      document.getElementById("minutes-exec-summary").textContent = s.executive_summary;

      this._renderWarnings(s);
      this._renderObjectives(s.objectives || []);

      // Decisões
      const decList = document.getElementById("minutes-decisions-list");
      decList.innerHTML = "";
      const decisions = (s.decisions || []).map((d) => (typeof d === "string" ? { description: d, evidence: [] } : d));
      if (decisions.length === 0) {
        decList.innerHTML = `<li class="decision-item"><span class="decision-icon">—</span><span>Nenhuma decisão formal registrada.</span></li>`;
      }
      decisions.forEach((d) => {
        const li = document.createElement("li");
        li.className = "decision-item";
        li.innerHTML = `<span class="decision-icon">✅</span><span>${escapeHtml(d.description)}${this._groundingHtml(d)}</span>`;
        decList.appendChild(li);
      });

      // Ações
      const actionsTbody = document.getElementById("minutes-actions-tbody");
      actionsTbody.innerHTML = "";
      if (s.action_items && s.action_items.length > 0) {
        s.action_items.forEach((act) => {
          const deadline = act.due_date
            ? `${escapeHtml(act.deadline)}<br><small style="color: var(--text-muted);">${escapeHtml(formatDateOnly(act.due_date))}</small>`
            : escapeHtml(act.deadline);
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${escapeHtml(act.task)}</strong>${this._groundingHtml(act)}</td>
            <td><span class="meta-pill">${escapeHtml(act.owner)}</span></td>
            <td>${deadline}</td>
            <td><span class="tag-badge ${act.status === "Concluído" ? "tag-done" : "tag-pending"}">${escapeHtml(act.status)}</span></td>
          `;
          actionsTbody.appendChild(tr);
        });
      } else {
        actionsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Nenhuma tarefa pendente registrada.</td></tr>`;
      }

      // Tópicos
      const topicsContainer = document.getElementById("minutes-topics-container");
      topicsContainer.innerHTML = "";
      (s.main_topics || []).forEach((t, idx) => {
        const div = document.createElement("div");
        div.style.background = "rgba(255, 255, 255, 0.02)";
        div.style.padding = "1rem";
        div.style.borderRadius = "var(--radius-md)";
        div.style.border = "1px solid var(--border-color)";
        div.innerHTML = `
          <h4 style="font-size: 1rem; color: #60a5fa; margin-bottom: 0.35rem;">${idx + 1}. ${escapeHtml(t.title)}${this._evidenceHtml(t.evidence)}</h4>
          <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;">${escapeHtml(t.discussion)}</p>
          ${t.conclusions ? `<p style="font-size: 0.85rem; color: #94a3b8;"><strong>Conclusão:</strong> ${escapeHtml(t.conclusions)}</p>` : ""}
        `;
        topicsContainer.appendChild(div);
      });

      this._renderItemList("card-open-points", "minutes-open-points-list", s.open_points);
      this._renderItemList("card-risks", "minutes-risks-list", s.risks);
    }

    // Metadados da transcrição
    const speakerNames = this._speakerNames(meeting);
    const transcriptTitleEl = document.getElementById("transcript-title");
    if (transcriptTitleEl) {
      transcriptTitleEl.textContent = `Transcrição: ${s ? s.title : meeting.title}`;
    }
    const transcriptDateEl = document.getElementById("transcript-date");
    if (transcriptDateEl) {
      transcriptDateEl.textContent = `📅 Data: ${formatDateTime(s && s.date ? s.date : meeting.created_at)}`;
    }
    const transcriptDurEl = document.getElementById("transcript-duration");
    if (transcriptDurEl) {
      transcriptDurEl.textContent = `⏱️ Duração: ${(meeting.audio_duration / 60).toFixed(1)} min`;
    }
    const transcriptPartEl = document.getElementById("transcript-participants");
    if (transcriptPartEl) {
      transcriptPartEl.textContent = `👥 Locutores (${speakerNames.length}): ${speakerNames.join(", ")}`;
    }

    audioPlayer.loadAudio(meeting.audio_url, meeting.segments);
    this._renderSpeakerInputs(meeting);
    this._renderSpeakerTools(meeting);
    this._renderSpeakerSuggestions(meeting);
    this._renderTranscriptFeed(meeting.segments);
  }

  _speakerNames(meeting) {
    const names = new Map();
    meeting.segments.forEach((seg) => names.set(seg.speaker_id || seg.speaker, seg.speaker));
    return Array.from(names.values());
  }

  _speakerColor(speakerId) {
    const idx = Math.max(0, (this.speakerOrder || []).indexOf(speakerId));
    return this.speakerColors[idx % this.speakerColors.length];
  }

  _renderWarnings(summary) {
    const container = document.getElementById("minutes-warnings");
    container.innerHTML = "";
    const warnings = [...(summary.warnings || [])];
    if (summary.source === "heuristic") {
      warnings.unshift("Ata gerada em modo de contingência (sem LLM). Verifique se o Ollama está ativo e regere a ata.");
    }
    warnings.forEach((w) => {
      const div = document.createElement("div");
      div.className = "warning-banner";
      div.textContent = `⚠️ ${w}`;
      container.appendChild(div);
    });
  }

  _renderObjectives(objectives) {
    const card = document.getElementById("card-objectives");
    const list = document.getElementById("minutes-objectives-list");
    list.innerHTML = "";
    card.style.display = objectives.length ? "block" : "none";
    objectives.forEach((o) => {
      const li = document.createElement("li");
      li.className = "objective-item";
      li.innerHTML = `
        <span class="status-badge status-${escapeHtml(o.status)}">${escapeHtml(OBJECTIVE_STATUS[o.status] || o.status)}</span>
        <div class="objective-body">
          <div class="objective-desc">${escapeHtml(o.description)}${this._evidenceHtml(o.evidence)}</div>
          ${o.notes ? `<div class="objective-notes">${escapeHtml(o.notes)}</div>` : ""}
          <div class="objective-origin">${escapeHtml(OBJECTIVE_ORIGIN[o.origin] || o.origin || "")}</div>
        </div>
      `;
      list.appendChild(li);
    });
  }

  _renderItemList(cardId, listId, items) {
    const card = document.getElementById(cardId);
    const list = document.getElementById(listId);
    if (!card || !list) return;
    const normalized = (items || []).map((i) => (typeof i === "string" ? { description: i, evidence: [] } : i));
    card.style.display = normalized.length ? "block" : "none";
    list.innerHTML = "";
    normalized.forEach((item) => {
      const li = document.createElement("li");
      li.innerHTML = `${escapeHtml(item.description)}${this._evidenceHtml(item.evidence)}`;
      list.appendChild(li);
    });
  }

  /** Chips clicáveis com o instante de cada trecho que sustenta o item (vai para a transcrição/áudio). */
  _evidenceHtml(evidence) {
    if (!evidence || !evidence.length || !this.segmentStartById) return "";
    const chips = evidence
      .filter((id) => this.segmentStartById.has(id))
      .slice(0, 4)
      .map((id) => {
        const t = this.segmentStartById.get(id);
        return `<button type="button" class="evidence-chip" title="Ouvir o trecho" onclick="app.goToSegment(${Number(id)})">${audioPlayer.formatTime(t)}</button>`;
      })
      .join("");
    return chips ? `<span class="evidence-chips">${chips}</span>` : "";
  }

  _groundingHtml(item) {
    const flag = item.grounded === false ? `<span class="ungrounded-flag" title="O modelo não apontou trecho da transcrição que sustente este item">⚠️ sem evidência</span>` : "";
    return this._evidenceHtml(item.evidence) + flag;
  }

  goToSegment(segmentId) {
    const start = this.segmentStartById && this.segmentStartById.get(segmentId);
    if (start === undefined) return;
    this.switchTab("transcript");
    audioPlayer.jumpTo(start);
    const bubble = document.querySelector(`.utterance-bubble[data-id="${Number(segmentId)}"]`);
    if (bubble) {
      bubble.scrollIntoView({ behavior: "smooth", block: "center" });
      bubble.classList.add("highlight");
      setTimeout(() => bubble.classList.remove("highlight"), 2500);
    }
  }

  /** Locutores presentes na transcrição: Map(speaker_id -> nome de exibição), na ordem em que aparecem. */
  _speakersOf(meeting) {
    const seen = new Map();
    meeting.segments.forEach((seg) => {
      const id = seg.speaker_id || seg.speaker;
      if (!seen.has(id)) seen.set(id, seg.speaker);
    });
    return seen;
  }

  _speakerOptionsHtml(speakers, selectedId = null) {
    return Array.from(speakers.entries())
      .map(([id, name]) => {
        const label = name !== id ? `${name} (${id})` : id;
        return `<option value="${escapeHtml(id)}" ${id === selectedId ? "selected" : ""}>${escapeHtml(label)}</option>`;
      })
      .join("");
  }

  _renderSpeakerTools(meeting) {
    const speakers = this._speakersOf(meeting);
    const src = document.getElementById("merge-source");
    const dst = document.getElementById("merge-target");
    if (src && dst) {
      const ids = Array.from(speakers.keys());
      src.innerHTML = this._speakerOptionsHtml(speakers, ids[1] || ids[0]);
      dst.innerHTML = this._speakerOptionsHtml(speakers, ids[0]);
    }
    const tools = document.getElementById("speaker-tools");
    if (tools) tools.style.display = meeting.segments.length ? "flex" : "none";
    const rediarizeBox = document.getElementById("rediarize-tool");
    if (rediarizeBox) {
      rediarizeBox.title = meeting.source ? "" : "Disponível para reuniões processadas a partir desta versão.";
      rediarizeBox.querySelectorAll("input, button").forEach((el) => (el.disabled = !meeting.source));
    }
  }

  _renderSpeakerInputs(meeting) {
    const container = document.getElementById("speakers-inputs-container");
    container.innerHTML = "";

    const seen = this._speakersOf(meeting);

    seen.forEach((displayName, speakerId) => {
      const chip = document.createElement("div");
      chip.className = "speaker-input-chip";
      chip.innerHTML = `
        <span class="speaker-color-dot" style="background-color: ${this._speakerColor(speakerId)};"></span>
        <input type="text" data-speaker-id="${escapeHtml(speakerId)}" value="${escapeHtml(displayName)}" title="${escapeHtml(speakerId)} — edite para renomear">
      `;
      container.appendChild(chip);
    });
  }

  /** Sugestões do LLM que não foram aplicadas automaticamente (evidência fraca): o usuário decide. */
  _renderSpeakerSuggestions(meeting) {
    const container = document.getElementById("speaker-suggestions-container");
    if (!container) return;
    container.innerHTML = "";
    const pending = ((meeting.summary && meeting.summary.speaker_suggestions) || []).filter(
      (sug) => !sug.applied && sug.confidence >= 0.5 && sug.reason !== "nome já definido pelo usuário" && sug.name
    );
    pending.forEach((sug) => {
      const div = document.createElement("div");
      div.className = "speaker-suggestion";
      div.innerHTML = `
        <span>💡 ${escapeHtml(sug.speaker_id)} pode ser <strong>${escapeHtml(sug.name)}</strong>
          <small>(${Math.round(sug.confidence * 100)}% · ${escapeHtml(sug.reason || sug.kind)})</small></span>
      `;
      const btn = document.createElement("button");
      btn.className = "btn btn-secondary btn-sm";
      btn.textContent = "Usar";
      btn.onclick = () => {
        const input = document.querySelector(`.speaker-input-chip input[data-speaker-id="${CSS.escape(sug.speaker_id)}"]`);
        if (input) input.value = sug.name;
        div.remove();
      };
      div.appendChild(btn);
      container.appendChild(div);
    });
  }

  _renderTranscriptFeed(segments) {
    const feed = document.getElementById("transcript-feed-container");
    feed.innerHTML = "";
    const speakers = this.currentMeeting ? this._speakersOf(this.currentMeeting) : new Map();

    segments.forEach((seg) => {
      const color = this._speakerColor(seg.speaker_id || seg.speaker);
      const initials = (seg.speaker || "?").split(" ").map((w) => w[0]).join("").substring(0, 2).toUpperCase();

      const bubble = document.createElement("div");
      bubble.className = "utterance-bubble";
      bubble.setAttribute("data-id", seg.id);
      bubble.setAttribute("data-start", seg.start);
      bubble.setAttribute("data-end", seg.end);
      bubble.onclick = () => audioPlayer.jumpTo(seg.start);

      bubble.innerHTML = `
        <div class="speaker-avatar" style="background-color: ${color};">${escapeHtml(initials)}</div>
        <div class="utterance-body">
          <div class="utterance-header">
            <span class="speaker-name-wrap">
              <span class="speaker-name">${escapeHtml(seg.speaker)}</span>
              <select class="speaker-reassign" title="Corrigir o locutor deste trecho" data-seg="${Number(seg.id)}">
                ${this._speakerOptionsHtml(speakers, seg.speaker_id || seg.speaker)}
                <option value="__new__">➕ Novo locutor</option>
              </select>
            </span>
            <span class="utterance-timestamp">▶ ${audioPlayer.formatTime(seg.start)} - ${audioPlayer.formatTime(seg.end)}</span>
          </div>
          <div class="utterance-text">${escapeHtml(seg.text)}</div>
        </div>
      `;
      const select = bubble.querySelector(".speaker-reassign");
      select.onclick = (e) => e.stopPropagation();
      select.onchange = (e) => {
        e.stopPropagation();
        const value = select.value;
        this.reassignSegments([seg.id], value === "__new__" ? null : value);
      };
      feed.appendChild(bubble);
    });
  }

  async _meetingAction(url, body, okMessage) {
    if (!this.currentMeeting) return null;
    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Operação não concluída.");
      if (okMessage) this.showToast(okMessage);
      return data;
    } catch (e) {
      alert(`Erro: ${e.message}`);
      return null;
    }
  }

  /** Move trechos para outro locutor (ou para um locutor novo, quando speakerId é null). */
  async reassignSegments(segmentIds, speakerId) {
    const id = encodeURIComponent(this.currentMeeting.id);
    const updated = await this._meetingAction(`/api/meetings/${id}/segments/reassign`,
      { segment_ids: segmentIds, speaker_id: speakerId }, "Locutor do trecho atualizado.");
    if (updated) this._renderMeetingData(updated);
    else this._renderTranscriptFeed(this.currentMeeting.segments);
  }

  /** Une dois locutores que a diarização separou indevidamente. */
  async mergeSpeakers() {
    const source = document.getElementById("merge-source").value;
    const target = document.getElementById("merge-target").value;
    if (!source || !target || source === target) {
      this.showToast("Escolha dois locutores diferentes para mesclar.");
      return;
    }
    if (!confirm(`Mesclar ${source} em ${target}? Todas as falas de ${source} passarão para ${target}.`)) return;
    const id = encodeURIComponent(this.currentMeeting.id);
    const updated = await this._meetingAction(`/api/meetings/${id}/speakers/merge`,
      { source_ids: [source], target_id: target }, "Locutores mesclados (transcrição e ata atualizadas).");
    if (updated) this._renderMeetingData(updated);
  }

  /** Refaz a separação de locutores reaproveitando a transcrição (só a diarização e a ata rodam de novo). */
  async rediarize() {
    const raw = document.getElementById("rediarize-count").value;
    const count = raw ? parseInt(raw, 10) : null;
    const msg = count ? `Refazer a separação considerando ${count} pessoa(s)?` : "Refazer a separação com detecção automática?";
    if (!confirm(`${msg}\nA ata será regerada; nomes já confirmados são mantidos quando a voz corresponder.`)) return;
    const id = encodeURIComponent(this.currentMeeting.id);
    const data = await this._meetingAction(`/api/meetings/${id}/rediarize`, {
      num_speakers: count,
      ollama_model: document.getElementById("select-ollama-model").value || null,
    });
    if (!data) return;
    this.switchTab("processing");
    this._resetProgressUI();
    this._appendLog("Refazendo a separação de locutores (a transcrição será reaproveitada)...");
    this._listenToJobEvents(data.job_id);
  }

  filterTranscript(query) {
    const q = query.toLowerCase().trim();
    const bubbles = document.querySelectorAll(".utterance-bubble");
    bubbles.forEach((b) => {
      const text = b.querySelector(".utterance-text").textContent.toLowerCase();
      const speaker = b.querySelector(".speaker-name").textContent.toLowerCase();
      if (!q || text.includes(q) || speaker.includes(q)) {
        b.style.display = "flex";
      } else {
        b.style.display = "none";
      }
    });
  }

  async saveSpeakerNames(regenerateSummary = false) {
    if (!this.currentMeeting) return;

    // Envia speaker_id -> nome. Nome vazio volta para o rótulo padrão.
    const speakerMap = {};
    document.querySelectorAll(".speaker-input-chip input").forEach((input) => {
      speakerMap[input.getAttribute("data-speaker-id")] = input.value.trim();
    });

    try {
      this.showToast(regenerateSummary ? "Regerando ata com novos oradores..." : "Salvando nomes...");
      const resp = await fetch(`/api/meetings/${encodeURIComponent(this.currentMeeting.id)}/speakers`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speaker_map: speakerMap,
          regenerate_summary: regenerateSummary,
          ollama_model: document.getElementById("select-ollama-model").value || null,
        }),
      });

      if (!resp.ok) throw new Error("Erro ao atualizar oradores.");
      const updated = await resp.json();
      this._renderMeetingData(updated);
      this.showToast("Nomes atualizados na transcrição e na ata!");
    } catch (e) {
      alert(`Erro: ${e.message}`);
    }
  }

  copyMinutesText() {
    if (!this.currentMeeting || !this.currentMeeting.summary) return;
    navigator.clipboard.writeText(this.currentMeeting.summary.raw_markdown);
    this.showToast("Ata copiada para a área de transferência!");
  }

  exportFile(format) {
    if (!this.currentMeeting) return;
    window.location.href = `/api/meetings/${this.currentMeeting.id}/export/${format}`;
  }

  copyTranscriptText() {
    if (!this.currentMeeting || !this.currentMeeting.segments || this.currentMeeting.segments.length === 0) {
      this.showToast("Nenhuma transcrição disponível para copiar.");
      return;
    }
    const lines = this.currentMeeting.segments.map((seg) => {
      const m = Math.floor(seg.start / 60);
      const s = Math.floor(seg.start % 60);
      const timeStr = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
      return `[${timeStr}] ${seg.speaker}: ${seg.text}`;
    });
    navigator.clipboard.writeText(lines.join("\n"));
    this.showToast("Transcrição copiada para a área de transferência!");
  }

  exportTranscript(format) {
    if (!this.currentMeeting || !this.currentMeeting.segments || this.currentMeeting.segments.length === 0) {
      this.showToast("Nenhuma transcrição disponível para exportar.");
      return;
    }
    window.location.href = `/api/meetings/${this.currentMeeting.id}/export-transcript/${format}`;
  }

  printMinutes() {
    if (!this.currentMeeting || !this.currentMeeting.summary) {
      this.showToast("Ata não disponível para impressão.");
      return;
    }
    document.body.classList.remove("print-transcript-mode");
    window.print();
  }

  printTranscript() {
    if (!this.currentMeeting || !this.currentMeeting.segments || this.currentMeeting.segments.length === 0) {
      this.showToast("Nenhuma transcrição disponível para impressão.");
      return;
    }
    document.body.classList.add("print-transcript-mode");
    window.print();
    window.addEventListener(
      "afterprint",
      () => {
        document.body.classList.remove("print-transcript-mode");
      },
      { once: true }
    );
    setTimeout(() => {
      document.body.classList.remove("print-transcript-mode");
    }, 2000);
  }


  async loadMeetingsHistory() {
    try {
      const resp = await fetch("/api/meetings");
      const list = await resp.json();
      const grid = document.getElementById("history-grid-container");
      grid.innerHTML = "";

      if (list.length === 0) {
        grid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1;">
            <div class="empty-state-icon">📂</div>
            <p>Nenhuma reunião anterior encontrada no banco local.</p>
          </div>
        `;
        return;
      }

      list.forEach((m) => {
        const card = document.createElement("div");
        card.className = "meeting-card";
        card.innerHTML = `
          <div>
            <h3 class="meeting-card-title">${escapeHtml(m.title)}</h3>
            <div class="meeting-card-meta">
              <span>📅 ${escapeHtml(formatDateTime(m.created_at))}</span>
              <span>⏱️ ${(m.audio_duration / 60).toFixed(1)} minutos</span>
              <span>👥 ${m.speaker_count} interlocutores</span>
            </div>
          </div>
          <div class="meeting-card-actions">
            <button class="btn btn-primary btn-sm" onclick="app.openMeeting('${escapeHtml(m.id)}')">Abrir Ata</button>
            <button class="btn btn-secondary btn-sm" style="color: var(--accent-rose);" onclick="app.deleteMeeting('${escapeHtml(m.id)}')">Excluir</button>
          </div>
        `;
        grid.appendChild(card);
      });
    } catch (e) {
      console.warn("Erro ao carregar histórico:", e);
    }
  }

  filterHistory(query) {
    const q = query.toLowerCase().trim();
    const cards = document.querySelectorAll(".meeting-card");
    cards.forEach((c) => {
      const title = c.querySelector(".meeting-card-title").textContent.toLowerCase();
      const meta = c.querySelector(".meeting-card-meta").textContent.toLowerCase();
      if (!q || title.includes(q) || meta.includes(q)) {
        c.style.display = "flex";
      } else {
        c.style.display = "none";
      }
    });
  }

  async openMeeting(meetingId) {
    try {
      const resp = await fetch(`/api/meetings/${encodeURIComponent(meetingId)}`);
      if (!resp.ok) throw new Error("not found");
      const meeting = await resp.json();
      this._renderMeetingData(meeting);
      this.switchTab("minutes");
      this.showToast(`Reunião '${meeting.title}' carregada!`);
    } catch (e) {
      alert("Erro ao abrir reunião.");
    }
  }

  async deleteMeeting(meetingId) {
    if (!confirm("Excluir esta reunião? O áudio e a transcrição também serão apagados do disco.")) return;
    try {
      await fetch(`/api/meetings/${encodeURIComponent(meetingId)}`, { method: "DELETE" });
      this.loadMeetingsHistory();
      this.showToast("Reunião e arquivos de áudio removidos.");
    } catch (e) {
      alert("Erro ao excluir reunião.");
    }
  }

  showToast(message, duration = 3000) {
    const existing = document.querySelector(".notification-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "notification-toast";
    toast.innerHTML = `<span>✨</span><span>${escapeHtml(message)}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
}

const app = new MeetingApp();
