/**
 * GERENCIADOR DE CONFIGURAÇÕES E ESTADO DO OLLAMA / WHISPER
 */
class SettingsManager {
  constructor() {
    this.ollamaUrlInput = document.getElementById("setting-ollama-url");
    this.hfTokenInput = document.getElementById("setting-hf-token");
    this.whisperDeviceSelect = document.getElementById("setting-whisper-device");
    this.ollamaModelSelect = document.getElementById("select-ollama-model");
    this.ollamaBadge = document.getElementById("badge-ollama");
    this.ollamaBadgeText = document.getElementById("badge-ollama-text");

    this._loadStoredPreferences();
  }

  _loadStoredPreferences() {
    const storedToken = localStorage.getItem("meeting_ai_hf_token") || "";
    if (this.hfTokenInput) this.hfTokenInput.value = storedToken;

    const storedDevice = localStorage.getItem("meeting_ai_whisper_device") || "cpu";
    if (this.whisperDeviceSelect) this.whisperDeviceSelect.value = storedDevice;
  }

  async testOllamaConnection() {
    try {
      const resp = await fetch("/api/models/ollama");
      const data = await resp.json();

      if (data.connected && data.available_models.length > 0) {
        this.ollamaBadge.className = "status-badge";
        this.ollamaBadgeText.textContent = `Ollama: Online (${data.available_models.length} modelos)`;
        this._populateOllamaSelect(data.available_models);
        app.showToast(`Conectado ao Ollama! ${data.available_models.length} modelos encontrados.`);
      } else {
        this.ollamaBadge.className = "status-badge warning";
        this.ollamaBadgeText.textContent = "Ollama: Sem modelos baixados";
        this._populateOllamaSelect([]);
        app.showToast("Ollama respondeu, mas nenhum modelo foi baixado ainda. Use 'ollama pull llama3.2' no terminal.", 5000);
      }
    } catch (e) {
      console.warn("Erro ao testar Ollama:", e);
      this.ollamaBadge.className = "status-badge warning";
      this.ollamaBadgeText.textContent = "Ollama: Desconectado";
      this._populateOllamaSelect([]);
      app.showToast("Não foi possível conectar ao Ollama. Verifique se o container está rodando.", 4000);
    }
  }

  _populateOllamaSelect(models) {
    if (!this.ollamaModelSelect) return;
    this.ollamaModelSelect.innerHTML = "";

    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Nenhum modelo detectado (usando gerador de contingência)";
      this.ollamaModelSelect.appendChild(opt);
      return;
    }

    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m.includes("llama3") || m.includes("qwen") || m.includes("mistral")) {
        opt.selected = true;
      }
      this.ollamaModelSelect.appendChild(opt);
    });
  }

  async pullModel(modelName, btnElement) {
    const originalText = btnElement ? btnElement.innerHTML : "";
    if (btnElement) {
      btnElement.disabled = true;
      btnElement.innerHTML = `⏳ Baixando ${modelName}...`;
    }
    app.showToast(`Iniciando download de '${modelName}' no Ollama... Isso pode levar alguns minutos.`, 6000);

    try {
      const resp = await fetch("/api/models/ollama/pull", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Erro ao baixar modelo.");

      app.showToast(`🎉 Modelo '${modelName}' baixado e pronto para uso!`, 6000);
      if (btnElement) {
        btnElement.innerHTML = `✅ Instalado`;
        btnElement.style.color = "var(--accent-emerald)";
      }
      await this.testOllamaConnection();
    } catch (err) {
      console.error(err);
      alert(`Falha ao baixar ${modelName}: ${err.message}`);
      if (btnElement) {
        btnElement.disabled = false;
        btnElement.innerHTML = originalText;
      }
    }
  }

  saveSettings() {
    const hfToken = this.hfTokenInput.value.trim();
    const whisperDevice = this.whisperDeviceSelect.value;

    localStorage.setItem("meeting_ai_hf_token", hfToken);
    localStorage.setItem("meeting_ai_whisper_device", whisperDevice);

    app.showToast("Configurações salvas com sucesso!");
  }

  getHfToken() {
    return localStorage.getItem("meeting_ai_hf_token") || "";
  }
}

const settingsManager = new SettingsManager();
