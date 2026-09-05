/**
 * APLICAÇÃO PRINCIPAL - CONTROLE DE ESTADO, ABAS E FLUXO DE PROCESSAMENTO
 */
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

      if (!uploadResp.ok) throw new Error("Falha no upload do arquivo.");
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
      if (minSpeakers) processForm.append("min_speakers", minSpeakers);
      if (customPrompt) processForm.append("custom_prompt", customPrompt);
      if (hfToken) processForm.append("hf_token", hfToken);

      const startResp = await fetch("/api/process", {
        method: "POST",
        body: processForm,
      });

      if (!startResp.ok) throw new Error("Erro ao disparar pipeline.");
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

    this.eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

    this.eventSource.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data);
        this._updateProgressUI(job);

        if (job.status === "completed") {
          this.eventSource.close();
          this.showToast("🎉 Reunião processada e Ata gerada com sucesso!");
          this.currentMeeting = job.result;
          this._renderMeetingData(job.result);
          setTimeout(() => this.switchTab("minutes"), 600);
        } else if (job.status === "failed") {
          this.eventSource.close();
          alert(`Falha no processamento: ${job.error}`);
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
        const res = await fetch(`/api/jobs/${jobId}`);
        const job = await res.json();
        this._updateProgressUI(job);

        if (job.status === "completed") {
          clearInterval(interval);
          this.showToast("🎉 Reunião concluída!");
          this.currentMeeting = job.result;
          this._renderMeetingData(job.result);
          this.switchTab("minutes");
        } else if (job.status === "failed") {
          clearInterval(interval);
          alert(`Falha no processamento: ${job.error}`);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 2000);
  }

  _resetProgressUI() {
    this.progressFill.style.width = "0%";
    this.currentStepLabel.textContent = "Preparando modelos locais...";
    this.logBox.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step-${i}`);
      step.className = "step-item";
    }
  }

  _updateProgressUI(job) {
    this.progressFill.style.width = `${job.progress}%`;
    this.currentStepLabel.textContent = job.current_step;
    this._appendLog(`[Progresso ${job.progress}%] ${job.current_step}`);

    const statusMap = {
      preprocessing: 1,
      diarizing: 2,
      transcribing: 3,
      aligning: 4,
      summarizing: 5,
      completed: 6,
    };

    const currentStepIdx = statusMap[job.status] || 1;
    for (let i = 1; i <= 5; i++) {
      const step = document.getElementById(`step-${i}`);
      if (i < currentStepIdx) {
        step.className = "step-item completed";
      } else if (i === currentStepIdx) {
        step.className = "step-item active";
      } else {
        step.className = "step-item";
      }
    }
  }

  _appendLog(text) {
    const line = document.createElement("div");
    const time = new Date().toLocaleTimeString();
    line.textContent = `[${time}] ${text}`;
    this.logBox.appendChild(line);
    this.logBox.scrollTop = this.logBox.scrollHeight;
  }

  _renderMeetingData(meeting) {
    this.currentMeeting = meeting;

    // 1. Preencher Ata
    if (meeting.summary) {
      document.getElementById("minutes-title").textContent = meeting.summary.title;
      document.getElementById("minutes-date").textContent = `📅 Data: ${meeting.summary.date}`;
      document.getElementById("minutes-duration").textContent = `⏱️ Duração: ${(meeting.audio_duration / 60).toFixed(1)} min`;
      document.getElementById("minutes-participants").textContent = `👥 Participantes: ${meeting.summary.participants.join(", ")}`;
      document.getElementById("minutes-exec-summary").textContent = meeting.summary.executive_summary;

      // Decisões
      const decList = document.getElementById("minutes-decisions-list");
      decList.innerHTML = "";
      (meeting.summary.decisions || []).forEach((d) => {
        const li = document.createElement("li");
        li.className = "decision-item";
        li.innerHTML = `<span class="decision-icon">✅</span><span>${d}</span>`;
        decList.appendChild(li);
      });

      // Ações
      const actionsTbody = document.getElementById("minutes-actions-tbody");
      actionsTbody.innerHTML = "";
      if (meeting.summary.action_items && meeting.summary.action_items.length > 0) {
        meeting.summary.action_items.forEach((act) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${act.task}</strong></td>
            <td><span class="meta-pill">${act.owner}</span></td>
            <td>${act.deadline}</td>
            <td><span class="tag-badge ${act.status === 'Concluído' ? 'tag-done' : 'tag-pending'}">${act.status}</span></td>
          `;
          actionsTbody.appendChild(tr);
        });
      } else {
        actionsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Nenhuma tarefa pendente registrada.</td></tr>`;
      }

      // Tópicos
      const topicsContainer = document.getElementById("minutes-topics-container");
      topicsContainer.innerHTML = "";
      (meeting.summary.main_topics || []).forEach((t, idx) => {
        const div = document.createElement("div");
        div.style.background = "rgba(255, 255, 255, 0.02)";
        div.style.padding = "1rem";
        div.style.borderRadius = "var(--radius-md)";
        div.style.border = "1px solid var(--border-color)";
        div.innerHTML = `
          <h4 style="font-size: 1rem; color: #60a5fa; margin-bottom: 0.35rem;">${idx + 1}. ${t.title}</h4>
          <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;">${t.discussion}</p>
          ${t.conclusions ? `<p style="font-size: 0.85rem; color: #94a3b8;"><strong>Conclusão:</strong> ${t.conclusions}</p>` : ''}
        `;
        topicsContainer.appendChild(div);
      });

      // Pontos em aberto
      const openPointsCard = document.getElementById("card-open-points");
      const openPointsList = document.getElementById("minutes-open-points-list");
      if (meeting.summary.open_points && meeting.summary.open_points.length > 0) {
        openPointsCard.style.display = "block";
        openPointsList.innerHTML = "";
        meeting.summary.open_points.forEach((op) => {
          const li = document.createElement("li");
          li.textContent = op;
          openPointsList.appendChild(li);
        });
      } else {
        openPointsCard.style.display = "none";
      }
    }

    // 2. Carregar Player de Áudio
    audioPlayer.loadAudio(meeting.audio_url, meeting.segments);

    // 3. Renderizar Locutores e Transcrição
    this._renderSpeakerInputs(meeting);
    this._renderTranscriptFeed(meeting.segments);
  }

  _renderSpeakerInputs(meeting) {
    const container = document.getElementById("speakers-inputs-container");
    container.innerHTML = "";

    const uniqueSpeakers = Array.from(new Set(meeting.segments.map((s) => s.speaker)));

    uniqueSpeakers.forEach((speaker, index) => {
      const color = this.speakerColors[index % this.speakerColors.length];
      const chip = document.createElement("div");
      chip.className = "speaker-input-chip";
      chip.innerHTML = `
        <span class="speaker-color-dot" style="background-color: ${color};"></span>
        <input type="text" data-orig="${speaker}" value="${speaker}" title="Edite para renomear este orador">
      `;
      container.appendChild(chip);
    });
  }

  _renderTranscriptFeed(segments) {
    const feed = document.getElementById("transcript-feed-container");
    feed.innerHTML = "";

    const uniqueSpeakers = Array.from(new Set(segments.map((s) => s.speaker)));

    segments.forEach((seg) => {
      const speakerIdx = uniqueSpeakers.indexOf(seg.speaker);
      const color = this.speakerColors[speakerIdx % this.speakerColors.length];
      const initials = seg.speaker.split(" ").map((w) => w[0]).join("").substring(0, 2).toUpperCase();

      const bubble = document.createElement("div");
      bubble.className = "utterance-bubble";
      bubble.setAttribute("data-start", seg.start);
      bubble.setAttribute("data-end", seg.end);
      bubble.onclick = () => audioPlayer.jumpTo(seg.start);

      bubble.innerHTML = `
        <div class="speaker-avatar" style="background-color: ${color};">${initials}</div>
        <div class="utterance-body">
          <div class="utterance-header">
            <span class="speaker-name">${seg.speaker}</span>
            <span class="utterance-timestamp">▶ ${audioPlayer.formatTime(seg.start)} - ${audioPlayer.formatTime(seg.end)}</span>
          </div>
          <div class="utterance-text">${seg.text}</div>
        </div>
      `;
      feed.appendChild(bubble);
    });
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

    const inputs = document.querySelectorAll(".speaker-input-chip input");
    const speakerMap = {};
    inputs.forEach((input) => {
      const orig = input.getAttribute("data-orig");
      const newVal = input.value.trim();
      if (newVal) speakerMap[orig] = newVal;
    });

    try {
      this.showToast(regenerateSummary ? "Regerando ata com novos oradores..." : "Salvando nomes...");
      const resp = await fetch(`/api/meetings/${this.currentMeeting.id}/speakers`, {
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
      this.showToast("Nomes atualizados com sucesso!");
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
            <h3 class="meeting-card-title">${m.title}</h3>
            <div class="meeting-card-meta">
              <span>📅 ${m.created_at}</span>
              <span>⏱️ ${(m.audio_duration / 60).toFixed(1)} minutos</span>
              <span>👥 ${m.speaker_count} interlocutores</span>
            </div>
          </div>
          <div class="meeting-card-actions">
            <button class="btn btn-primary btn-sm" onclick="app.openMeeting('${m.id}')">Abrir Ata</button>
            <button class="btn btn-secondary btn-sm" style="color: var(--accent-rose);" onclick="app.deleteMeeting('${m.id}')">Excluir</button>
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
      const resp = await fetch(`/api/meetings/${meetingId}`);
      const meeting = await resp.json();
      this._renderMeetingData(meeting);
      this.switchTab("minutes");
      this.showToast(`Reunião '${meeting.title}' carregada!`);
    } catch (e) {
      alert("Erro ao abrir reunião.");
    }
  }

  async deleteMeeting(meetingId) {
    if (!confirm("Tem certeza que deseja excluir esta reunião?")) return;
    try {
      await fetch(`/api/meetings/${meetingId}`, { method: "DELETE" });
      this.loadMeetingsHistory();
      this.showToast("Reunião removida com sucesso.");
    } catch (e) {
      alert("Erro ao excluir reunião.");
    }
  }

  showToast(message, duration = 3000) {
    const existing = document.querySelector(".notification-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "notification-toast";
    toast.innerHTML = `<span>✨</span><span>${message}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
}

const app = new MeetingApp();
