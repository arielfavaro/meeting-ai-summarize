/**
 * GRAVADOR DE ÁUDIO COM VISUALIZADOR DE ONDAS EM TEMPO REAL
 */
class AudioRecorder {
  constructor() {
    this.recordBtn = document.getElementById("btn-record-mic");
    this.timerLabel = document.getElementById("recording-timer");
    this.statusLabel = document.getElementById("mic-status-text");
    this.canvas = document.getElementById("mic-visualizer");
    this.canvasCtx = this.canvas ? this.canvas.getContext("2d") : null;

    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.startTime = null;
    this.timerInterval = null;

    this.audioCtx = null;
    this.analyser = null;
    this.animationId = null;

    this._setupEvents();
  }

  _setupEvents() {
    if (this.recordBtn) {
      this.recordBtn.addEventListener("click", () => this.toggleRecord());
    }
  }

  async toggleRecord() {
    if (!this.isRecording) {
      await this.startRecord();
    } else {
      this.stopRecord();
    }
  }

  async startRecord() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder.mimeType || "audio/webm";
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
        app.setRecordedAudio(audioBlob, "gravacao_microfone.webm");
        stream.getTracks().forEach((track) => track.stop());
      };

      this.mediaRecorder.start(250);
      this.isRecording = true;

      // UI
      this.recordBtn.classList.add("recording");
      this.statusLabel.textContent = "Gravando ao vivo... Clique novamente para finalizar.";
      this.statusLabel.style.color = "var(--accent-rose)";

      // Timer
      this.startTime = Date.now();
      this.timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, "0");
        const secs = (elapsed % 60).toString().padStart(2, "0");
        this.timerLabel.textContent = `${mins}:${secs}`;
      }, 1000);

      // Visualizador Canvas
      this._startVisualizer(stream);
    } catch (err) {
      console.error("Erro ao acessar microfone:", err);
      alert("Permissão para usar o microfone foi negada ou não está disponível.");
    }
  }

  stopRecord() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;

      // UI
      this.recordBtn.classList.remove("recording");
      this.statusLabel.textContent = "Gravação finalizada! Pronto para processar.";
      this.statusLabel.style.color = "var(--accent-emerald)";

      clearInterval(this.timerInterval);
      if (this.animationId) {
        cancelAnimationFrame(this.animationId);
      }
      if (this.audioCtx) {
        this.audioCtx.close();
      }
    }
  }

  _startVisualizer(stream) {
    if (!this.canvasCtx) return;

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = this.audioCtx.createMediaStreamSource(stream);
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 64;
    source.connect(this.analyser);

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      this.animationId = requestAnimationFrame(draw);
      this.analyser.getByteFrequencyData(dataArray);

      const width = this.canvas.width;
      const height = this.canvas.height;
      this.canvasCtx.clearRect(0, 0, width, height);

      const barWidth = (width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * height;
        this.canvasCtx.fillStyle = `rgba(59, 130, 246, ${Math.max(0.2, dataArray[i] / 255)})`;
        this.canvasCtx.fillRect(x, height - barHeight, barWidth - 1, barHeight);
        x += barWidth;
      }
    };

    draw();
  }
}

const audioRecorder = new AudioRecorder();
