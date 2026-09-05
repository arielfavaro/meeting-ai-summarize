/**
 * PLAYER DE ÁUDIO SINCRONIZADO COM TRANSCRIÇÃO
 */
class AudioPlayer {
  constructor() {
    this.audio = document.getElementById("global-audio-element");
    this.playBtn = document.getElementById("audio-play-btn");
    this.playIcon = document.getElementById("play-icon");
    this.pauseIcon = document.getElementById("pause-icon");
    this.progressBar = document.getElementById("audio-progress-bar");
    this.currentTimeLabel = document.getElementById("time-current");
    this.totalTimeLabel = document.getElementById("time-total");
    this.isPlaying = false;
    this.currentSegments = [];

    this._setupListeners();
  }

  _setupListeners() {
    this.audio.addEventListener("timeupdate", () => this._onTimeUpdate());
    this.audio.addEventListener("loadedmetadata", () => this._onLoadedMetadata());
    this.audio.addEventListener("ended", () => {
      this.isPlaying = false;
      this._updatePlayButton();
    });

    this.progressBar.addEventListener("input", (e) => {
      if (this.audio.duration) {
        const targetTime = (e.target.value / 100) * this.audio.duration;
        this.audio.currentTime = targetTime;
      }
    });
  }

  loadAudio(url, segments = []) {
    this.audio.src = url;
    this.currentSegments = segments;
    this.audio.load();
    this.isPlaying = false;
    this._updatePlayButton();
  }

  togglePlay() {
    if (!this.audio.src) return;
    if (this.audio.paused) {
      this.audio.play();
      this.isPlaying = true;
    } else {
      this.audio.pause();
      this.isPlaying = false;
    }
    this._updatePlayButton();
  }

  jumpTo(seconds) {
    if (!this.audio.src) return;
    this.audio.currentTime = seconds;
    if (this.audio.paused) {
      this.audio.play();
      this.isPlaying = true;
      this._updatePlayButton();
    }
  }

  setSpeed(speed) {
    this.audio.playbackRate = speed;
  }

  _updatePlayButton() {
    if (this.isPlaying) {
      this.playIcon.style.display = "none";
      this.pauseIcon.style.display = "block";
    } else {
      this.playIcon.style.display = "block";
      this.pauseIcon.style.display = "none";
    }
  }

  _onLoadedMetadata() {
    this.totalTimeLabel.textContent = this.formatTime(this.audio.duration);
    this.progressBar.value = 0;
  }

  _onTimeUpdate() {
    const cur = this.audio.currentTime;
    const dur = this.audio.duration;
    this.currentTimeLabel.textContent = this.formatTime(cur);

    if (dur > 0) {
      this.progressBar.value = (cur / dur) * 100;
    }

    // Destacar o balão de fala correspondente
    this._highlightActiveSegment(cur);
  }

  _highlightActiveSegment(currentTime) {
    const bubbles = document.querySelectorAll(".utterance-bubble");
    bubbles.forEach((bubble) => {
      const start = parseFloat(bubble.getAttribute("data-start"));
      const end = parseFloat(bubble.getAttribute("data-end"));
      if (currentTime >= start && currentTime <= end) {
        bubble.classList.add("active-speech");
      } else {
        bubble.classList.remove("active-speech");
      }
    });
  }

  formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
}

const audioPlayer = new AudioPlayer();
