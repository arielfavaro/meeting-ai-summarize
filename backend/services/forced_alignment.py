"""
Alinhamento forçado palavra a palavra (estilo WhisperX), 100% local.

Os timestamps por palavra do Whisper vêm da atenção cruzada e erram por centenas de ms —
o suficiente para uma palavra "cair" no turno do locutor errado. Aqui um modelo CTC de
fonemas/letras em português (wav2vec2 XLSR-53 PT) recebe o áudio de cada segmento e o
texto que o Whisper transcreveu; o caminho de Viterbi do CTC dá o início e o fim reais de
cada palavra.

- O texto NÃO muda: só os tempos das palavras são refinados.
- Palavras sem letras alinháveis (números, siglas com símbolos) mantêm/interpolam o tempo.
- Se o alinhamento de um segmento falhar (áudio curto demais para o texto), o segmento
  fica com os tempos do Whisper.
"""
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
NEG_INF = -1e30


# --------------------------------------------------------------------------- CTC (numpy puro)
def ctc_align(log_probs: np.ndarray, tokens: Sequence[int], blank: int) -> Optional[List[Tuple[int, int]]]:
    """
    Viterbi do CTC: para cada token, o (primeiro, último) quadro em que ele é emitido.
    `log_probs`: (quadros, vocabulário) em log. Retorna None se não houver caminho viável.
    """
    T, N = log_probs.shape[0], len(tokens)
    if N == 0 or T < N:
        return None
    tokens = np.asarray(tokens)
    emit = log_probs[:, tokens]                     # (T, N)
    blank_lp = log_probs[:, blank]                  # (T,)
    # Estado j = "já emiti os j primeiros tokens" (0..N). Ficar no estado j>0 = blank ou repetir o token.
    score = np.full(N + 1, NEG_INF)
    score[0] = 0.0
    # 0 = ficou com blank, 1 = avançou (emitiu token novo), 2 = ficou repetindo o token
    moves = np.zeros((T, N + 1), dtype=np.int8)
    for t in range(T):
        stay_blank = score + blank_lp[t]
        stay_rep = np.full(N + 1, NEG_INF)
        stay_rep[1:] = score[1:] + emit[t]
        adv = np.full(N + 1, NEG_INF)
        adv[1:] = score[:-1] + emit[t]
        best_stay = np.maximum(stay_blank, stay_rep)
        new = np.maximum(best_stay, adv)
        moves[t] = np.where(adv >= best_stay, 1, np.where(stay_rep > stay_blank, 2, 0))
        score = new
    if score[N] <= NEG_INF / 2:
        return None

    spans: List[List[int]] = [[-1, -1] for _ in range(N)]
    j = N
    for t in range(T - 1, -1, -1):
        move = moves[t, j]
        if j > 0 and move in (1, 2):
            first, last = spans[j - 1]
            spans[j - 1] = [t, last if last >= 0 else t]
        if move == 1:
            j -= 1
    if j != 0 or any(s[0] < 0 for s in spans):
        return None
    return [(s[0], s[1]) for s in spans]


# --------------------------------------------------------------------------- texto -> tokens
def _chars(word: str, vocab: Dict[str, int]) -> List[str]:
    out = []
    for c in unicodedata.normalize("NFC", word.lower()):
        if c in vocab:
            out.append(c)
            continue
        base = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii")
        if base and base in vocab:
            out.append(base)
    return out


@dataclass
class _WordTokens:
    index: int            # posição da palavra no segmento
    first: int            # índice do 1º token da palavra na sequência
    last: int             # índice do último token


def build_tokens(words: Sequence[str], vocab: Dict[str, int], delimiter: Optional[str]) -> Tuple[List[int], List[_WordTokens]]:
    tokens: List[int] = []
    spans: List[_WordTokens] = []
    for i, word in enumerate(words):
        chars = _chars(word, vocab)
        if not chars:
            continue
        if tokens and delimiter and delimiter in vocab:
            tokens.append(vocab[delimiter])
        first = len(tokens)
        tokens.extend(vocab[c] for c in chars)
        spans.append(_WordTokens(i, first, len(tokens) - 1))
    return tokens, spans


def refine_segment_words(words: List[Dict[str, Any]], log_probs: np.ndarray, frame_sec: float, offset: float,
                         vocab: Dict[str, int], blank: int, delimiter: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Novos tempos para as palavras de UM segmento (ou None se não deu para alinhar)."""
    tokens, spans = build_tokens([w["word"] for w in words], vocab, delimiter)
    frames = ctc_align(log_probs, tokens, blank)
    if frames is None:
        return None
    out = [dict(w) for w in words]
    for span in spans:
        start_f, end_f = frames[span.first][0], frames[span.last][1]
        out[span.index]["start"] = round(offset + start_f * frame_sec, 3)
        out[span.index]["end"] = round(offset + (end_f + 1) * frame_sec, 3)
        out[span.index]["aligned"] = True
    _interpolate_unaligned(out)
    return out


def _interpolate_unaligned(words: List[Dict[str, Any]]) -> None:
    """Palavras sem letras alinháveis ficam entre a anterior e a seguinte já alinhadas."""
    for i, w in enumerate(words):
        if w.get("aligned"):
            continue
        prev_end = next((words[k]["end"] for k in range(i - 1, -1, -1) if words[k].get("aligned")), None)
        next_start = next((words[k]["start"] for k in range(i + 1, len(words)) if words[k].get("aligned")), None)
        if prev_end is not None and next_start is not None and next_start >= prev_end:
            w["start"], w["end"] = prev_end, next_start
        elif prev_end is not None:
            w["start"] = max(w["start"], prev_end)
            w["end"] = max(w["end"], w["start"])
        elif next_start is not None:
            w["end"] = min(w["end"], next_start)
            w["start"] = min(w["start"], w["end"])


# --------------------------------------------------------------------------- serviço
EmissionFn = Callable[[np.ndarray], np.ndarray]  # áudio 16 kHz -> log-probs (quadros, vocab)


class ForcedAligner:
    """wav2vec2 CTC lido de diretório local (baixado uma vez pelo script de download)."""

    def __init__(self, model_dir: Path, device: str = "cpu", emissions: Optional[EmissionFn] = None,
                 vocab: Optional[Dict[str, int]] = None, blank: int = 0, delimiter: Optional[str] = "|"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._emissions = emissions
        self._vocab = vocab
        self._blank = blank
        self._delimiter = delimiter
        self._model = None
        self._processor = None

    def unavailable_reason(self) -> Optional[str]:
        if self._emissions is not None:
            return None
        from backend.services.diarization.base import has_module
        for module in ("torch", "transformers"):
            if not has_module(module):
                return f"biblioteca '{module}' não instalada"
        if not (self.model_dir / "config.json").exists():
            return (f"modelo de alinhamento não baixado em {self.model_dir} — baixe uma vez com: docker compose run "
                    "--rm -e HF_HUB_OFFLINE=0 app python -m backend.scripts.download_models (baixa só o que falta)")
        return None

    def _load(self) -> None:
        if self._emissions is not None or self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info("Carregando modelo de alinhamento de %s (%s)...", self.model_dir, self.device)
        self._processor = Wav2Vec2Processor.from_pretrained(str(self.model_dir), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(self.model_dir), local_files_only=True)
        device = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._model = model.to(device).eval()
        tok = self._processor.tokenizer
        self._vocab = {k.lower(): v for k, v in tok.get_vocab().items()}
        self._blank = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._delimiter = getattr(tok, "word_delimiter_token", "|")

    def _log_probs(self, audio: np.ndarray) -> np.ndarray:
        if self._emissions is not None:
            return self._emissions(audio)
        import torch
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(inputs.input_values.to(self._model.device)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def refine(self, audio_path: Path, segments: List[Dict[str, Any]],
               progress: Optional[Callable[[float, float], None]] = None) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Refina os tempos das palavras de todos os segmentos. Retorna (segmentos, alinhados, total).
        Segmentos sem palavras ou que não alinham ficam como vieram do Whisper.
        """
        from backend.services.voice.embedder import load_audio

        self._load()
        audio = load_audio(audio_path)
        total = len(audio) / SAMPLE_RATE
        out, ok, tried = [], 0, 0
        for seg in segments:
            words = seg.get("words") or []
            if not words:
                out.append(seg)
                continue
            tried += 1
            start = max(0.0, float(seg["start"]) - 0.1)
            end = min(total, float(seg["end"]) + 0.1)
            chunk = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            refined = None
            if len(chunk) >= int(0.2 * SAMPLE_RATE):
                try:
                    lp = self._log_probs(chunk)
                    frame_sec = (len(chunk) / SAMPLE_RATE) / max(1, lp.shape[0])
                    refined = refine_segment_words(words, lp, frame_sec, start, self._vocab, self._blank,
                                                   self._delimiter)
                except Exception:  # um segmento problemático não derruba o resto
                    logger.debug("Falha ao alinhar segmento %.1f-%.1f", start, end, exc_info=True)
            if refined:
                ok += 1
                out.append({**seg, "words": refined, "start": refined[0]["start"], "end": refined[-1]["end"]})
            else:
                out.append(seg)
            if progress:
                progress(float(seg["end"]), total)
        return out, ok, tried

    def release(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        from backend.services.diarization.local_engines import _empty_cuda_cache
        _empty_cuda_cache()
