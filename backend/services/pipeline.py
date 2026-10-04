"""
Orquestração do processamento de uma reunião (casos de uso "processar" e "re-diarizar").

- Etapas pesadas (FFmpeg, diarização, Whisper) rodam em threads (`asyncio.to_thread`): o
  event loop continua respondendo. Um semáforo limita os pipelines simultâneos.
- Cada etapa pesada grava um checkpoint (ArtifactStore): "tentar novamente" retoma de onde
  parou e re-diarizar reaproveita a transcrição.
- Ordem pensada para GPU: diariza todas as faixas → libera o modelo → transcreve todas as
  faixas → libera o Whisper → ata (Ollama). Os modelos não disputam VRAM.
- Tudo local: nenhuma etapa envia áudio ou texto para fora da máquina.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from backend.models.schemas import MeetingDetail, MeetingSource, ProcessOptions, RediarizeRequest, SpeakerSegment
from backend.services.artifacts import ArtifactStore
from backend.services.speaker_editing import carry_speaker_names
from backend.services.alignment import align_transcription_with_diarization, merge_multitrack_segments
from backend.services.audio_service import AudioService
from backend.services.diarization import DiarizationService
from backend.services.file_store import FileStore
from backend.services.jobs import JobRegistry, JobReporter
from backend.services.minutes.generator import MinutesGenerator, MinutesRequest
from backend.services.speakers import present_meeting, sync_display_names
from backend.services.transcription import TranscriptionService

logger = logging.getLogger("meeting_ai.pipeline")


def apply_speaker_suggestions(meeting: MeetingDetail, log: Optional[Callable[..., None]] = None) -> None:
    """Aplica nomes validados pelo `speaker_naming` sem sobrescrever nomes definidos pelo usuário."""
    if not meeting.summary:
        return
    for sug in meeting.summary.speaker_suggestions:
        if sug.applied and meeting.speaker_name_sources.get(sug.speaker_id) != "user":
            logger.info("Nome inferido: '%s' -> '%s' (%s, conf=%.2f)", sug.speaker_id, sug.name, sug.reason, sug.confidence)
            if log:
                log(f"Nome identificado: {sug.speaker_id} ➔ {sug.name} ({sug.reason}, {sug.confidence:.0%}).", "success")
            meeting.speaker_map[sug.speaker_id] = sug.name
            meeting.speaker_name_sources[sug.speaker_id] = "llm"
    sync_display_names(meeting)


def _transcription_progress(reporter: JobReporter, base: int, span: float, label: str):
    """Callback do Whisper: progresso contínuo na barra e uma linha de log a cada 20%."""
    last_bucket = {"value": -1}

    def on_progress(current_sec: float, total_sec: float) -> None:
        if total_sec <= 0:
            return
        pct = min(99, int(current_sec / total_sec * 100))
        text = f"{label}: {pct}% ({current_sec / 60:.1f}/{total_sec / 60:.1f} min)"
        bucket = pct // 20
        if bucket > last_bucket["value"]:
            last_bucket["value"] = bucket
            reporter.log(text, progress=base + int(pct / 100 * span))
        else:
            reporter.progress(base + int(pct / 100 * span), f"{text}...")
    return on_progress


class MeetingPipeline:
    def __init__(
        self,
        jobs: JobRegistry,
        files: FileStore,
        repository,
        minutes: MinutesGenerator,
        processed_dir: Path,
        artifacts: Optional[ArtifactStore] = None,
        diarizer_factory: Callable[..., DiarizationService] = DiarizationService,
        transcriber=TranscriptionService,
        audio=AudioService,
        release_models: bool = False,
    ):
        self.jobs = jobs
        self.files = files
        self.repository = repository
        self.minutes = minutes
        self.processed_dir = processed_dir
        self.artifacts = artifacts or ArtifactStore(processed_dir / "artifacts")
        self.diarizer_factory = diarizer_factory
        self.transcriber = transcriber
        self.audio = audio
        self.release_models = release_models

    # ------------------------------------------------------------ casos de uso
    async def run(self, job_id: str, file_id: str, title: str, options: ProcessOptions) -> None:
        await self._guard(job_id, lambda reporter: self._process(file_id, title, options, reporter))

    async def run_rediarize(self, job_id: str, meeting_id: str, request: RediarizeRequest) -> None:
        await self._guard(job_id, lambda reporter: self._rediarize(meeting_id, request, reporter))

    async def _guard(self, job_id: str, work: Callable[[JobReporter], Awaitable[None]]) -> None:
        reporter = JobReporter(self.jobs, job_id, logger)
        try:
            if self.jobs.slots.locked():
                reporter.log("Aguardando outro processamento terminar...")
            async with self.jobs.slots:
                await work(reporter)
        except Exception as e:
            logger.exception("Erro durante processamento do job %s", job_id)
            reporter.log(f"Falha no processamento: {e}", "error", status="failed", progress=100, error=str(e))

    async def _process(self, file_id: str, title: str, options: ProcessOptions, reporter: JobReporter) -> None:
        raw_audio_path = self.files.upload_path(file_id)
        terms = [*options.context.participants, *options.context.glossary]

        tracks = await self._tracks(file_id, raw_audio_path, reporter)
        wav_path, wav_filename, duration = await self._master_audio(file_id, raw_audio_path, tracks, reporter)

        diarizer = self.diarizer_factory(options.hf_token)
        aligned, engine_used = await self._speech_to_segments(
            file_id, raw_audio_path, tracks, wav_path, diarizer, options.min_speakers, options.max_speakers,
            options.whisper_model, options.language, terms, reporter)

        created_at = datetime.now().astimezone()
        speaker_ids = list(dict.fromkeys(s.speaker_id for s in aligned))
        meeting = MeetingDetail(
            id=str(uuid.uuid4()),
            title=title or f"Reunião {created_at.strftime('%d/%m/%Y')}",
            created_at=created_at.isoformat(timespec="seconds"),
            audio_filename=wav_filename,
            audio_duration=duration,
            audio_url=f"/api/audio/{wav_filename}",
            segments=aligned,
            speaker_map={sid: sid for sid in speaker_ids},
            speaker_name_sources={sid: "default" for sid in speaker_ids},
            context=options.context,
            source=MeetingSource(file_id=file_id, tracks=tracks, whisper_model=options.whisper_model,
                                 language=options.language, min_speakers=options.min_speakers,
                                 max_speakers=options.max_speakers, diarization_engine=engine_used),
        )
        await self._summarize_and_save(meeting, options.ollama_model, reporter)

    async def _rediarize(self, meeting_id: str, request: RediarizeRequest, reporter: JobReporter) -> None:
        meeting = await asyncio.to_thread(self.repository.get, meeting_id)
        if not meeting:
            raise ValueError("Reunião não encontrada.")
        if not meeting.source:
            raise ValueError("Esta reunião foi processada antes do suporte a re-diarização; processe o áudio novamente.")
        src = meeting.source
        min_spk = request.num_speakers or request.min_speakers
        max_spk = request.num_speakers or request.max_speakers
        reporter.step(f"Refazendo a separação de locutores ({request.num_speakers or 'automático'} pessoa(s))...",
                      status="preprocessing", progress=5)

        try:
            raw_audio_path: Optional[Path] = self.files.upload_path(src.file_id)
        except FileNotFoundError:
            raw_audio_path = None  # o WAV mestre/faixas já processados bastam
        wav_path = self.processed_dir / meeting.audio_filename
        context = meeting.context
        terms = [*context.participants, *context.glossary]

        diarizer = self.diarizer_factory(None, engine=request.engine) if request.engine else self.diarizer_factory(None)
        aligned, engine_used = await self._speech_to_segments(
            src.file_id, raw_audio_path, src.tracks, wav_path, diarizer, min_spk, max_spk,
            src.whisper_model, src.language, terms, reporter)

        speaker_map, sources = carry_speaker_names(meeting, aligned)
        kept = [f"{sid} ➔ {name}" for sid, name in speaker_map.items() if name != sid]
        if kept:
            reporter.log(f"Nomes mantidos pela sobreposição de fala: {', '.join(kept)}.", "success")
        meeting.segments = aligned
        meeting.speaker_map, meeting.speaker_name_sources = speaker_map, sources
        meeting.source = src.model_copy(update={"min_speakers": min_spk, "max_speakers": max_spk,
                                                "diarization_engine": engine_used})
        sync_display_names(meeting)
        await self._summarize_and_save(meeting, request.ollama_model, reporter)

    # ------------------------------------------------------------------ etapas
    async def _tracks(self, file_id: str, raw_audio_path: Path, reporter: JobReporter) -> List[int]:
        cached = self.artifacts.load(file_id, "tracks", {})
        if cached is not None:
            reporter.step(f"Faixas de áudio já analisadas: {cached}.", status="preprocessing", progress=5)
            return cached
        reporter.step("Analisando faixas de áudio (FFmpeg)...", status="preprocessing", progress=5)
        tracks = await asyncio.to_thread(self.audio.detect_active_audio_tracks, raw_audio_path, reporter.callback())
        self.artifacts.save(file_id, "tracks", {}, tracks)
        return tracks

    async def _master_audio(self, file_id: str, raw_audio_path: Path, tracks: List[int],
                            reporter: JobReporter) -> Tuple[Path, str, float]:
        wav_filename = f"{Path(file_id).stem}_16k.wav"
        wav_path = self.processed_dir / wav_filename
        cached = self.artifacts.load(file_id, "master_audio", {"tracks": tracks})
        if cached and wav_path.exists():
            reporter.step(f"Áudio mestre reaproveitado ({cached['duration'] / 60:.1f} min).", progress=15, level="success")
            return wav_path, wav_filename, cached["duration"]
        reporter.step("Gerando áudio mestre 16kHz mono para o player...", progress=10)
        wav_path, duration = await asyncio.to_thread(self.audio.convert_to_wav_16k_mono, raw_audio_path, wav_path, tracks)
        self.artifacts.save(file_id, "master_audio", {"tracks": tracks}, {"duration": duration})
        reporter.step(f"Áudio mestre pronto: {duration / 60:.1f} min.", progress=15, level="success")
        return wav_path, wav_filename, duration

    async def _track_audio(self, file_id: str, raw_audio_path: Optional[Path], track_no: int,
                           tracks: List[int], master_wav: Path) -> Path:
        if len(tracks) == 1:
            return master_wav
        track_path = self.processed_dir / f"{Path(file_id).stem}_track_{track_no}_16k.wav"
        if not track_path.exists():
            if raw_audio_path is None:
                raise FileNotFoundError("Arquivo original não está mais disponível para extrair as faixas.")
            await asyncio.to_thread(self.audio.extract_track_to_wav_16k, raw_audio_path, track_no, track_path)
        return track_path

    async def _speech_to_segments(
        self, file_id: str, raw_audio_path: Optional[Path], tracks: List[int], master_wav: Path,
        diarizer: DiarizationService, min_speakers: Optional[int], max_speakers: Optional[int],
        whisper_model: str, language: str, terms: List[str], reporter: JobReporter,
    ) -> Tuple[List[SpeakerSegment], Optional[str]]:
        n = len(tracks)
        multi = n > 1
        if multi:
            reporter.step(f"Gravação multi-faixa: {n} faixas ativas {tracks}.", progress=15)
        labels = {t: (f"[Faixa {i + 1}/{n}] " if multi else "") for i, t in enumerate(tracks)}
        audio_paths = {t: await self._track_audio(file_id, raw_audio_path, t, tracks, master_wav) for t in tracks}

        # 1) Diarização de todas as faixas (modelo de diarização carregado uma vez só)
        diarizations: Dict[int, List[Dict[str, Any]]] = {}
        engines_used = set()
        for i, t in enumerate(tracks):
            # Em faixas individuais não forçamos min_speakers: um microfone dedicado deve virar 1 locutor.
            mn = None if multi else min_speakers
            params = {"track": t, "engine": diarizer.preferred, "min": mn, "max": max_speakers}
            reporter.step(f"{labels[t]}Separando locutores...", status="diarizing", progress=15 + int(i / n * 20))
            cached = self.artifacts.load(file_id, "diarization", params)
            if cached:
                reporter.log(f"{labels[t]}Separação de locutores reaproveitada ({cached['engine']}).", "success")
            else:
                turns = await asyncio.to_thread(diarizer.diarize, audio_paths[t], mn, max_speakers,
                                                reporter.callback(labels[t]))
                cached = {"turns": turns, "engine": diarizer.last_engine}
                self.artifacts.save(file_id, "diarization", params, cached)
            diarizations[t] = cached["turns"]
            engines_used.add(cached.get("engine"))
        if self.release_models:
            await asyncio.to_thread(diarizer.release)

        # 2) Transcrição de todas as faixas (Whisper carregado uma vez só)
        transcriptions: Dict[int, List[Dict[str, Any]]] = {}
        for i, t in enumerate(tracks):
            params = {"track": t, "model": whisper_model, "language": language, "terms": terms}
            base, span = 35 + int(i / n * 40), 40 / n
            cached = self.artifacts.load(file_id, "transcription", params)
            if cached is not None:
                reporter.step(f"{labels[t]}Transcrição reaproveitada (etapa já concluída).", status="transcribing",
                              progress=int(base + span), level="success")
                transcriptions[t] = cached
                continue
            reporter.step(f"{labels[t]}Transcrevendo (Faster-Whisper '{whisper_model}')...", status="transcribing",
                          progress=base)
            on_progress = _transcription_progress(reporter, base=base, span=span, label=f"{labels[t]}Transcrição")
            result = await asyncio.to_thread(
                self.transcriber.transcribe, audio_paths[t], whisper_model, language, 5, on_progress, terms)
            self.artifacts.save(file_id, "transcription", params, result)
            transcriptions[t] = result
        if self.release_models:
            await asyncio.to_thread(self.transcriber.release)
            reporter.log("Modelos de áudio liberados da memória antes da ata.")

        # 3) Alinhamento por frase e intercalação das faixas
        reporter.step("Sincronizando falas e locutores (atribuição por frase)...", status="aligning", progress=75)
        per_track = [align_transcription_with_diarization(transcriptions[t], diarizations[t]) for t in tracks]
        aligned = merge_multitrack_segments(per_track) if multi else per_track[0]
        speakers = len({s.speaker_id for s in aligned})
        reporter.step(f"Alinhamento concluído: {len(aligned)} falas, {speakers} locutor(es).", progress=80,
                      level="success")
        engines_used.discard(None)
        return aligned, ",".join(sorted(engines_used)) or None

    async def _summarize_and_save(self, meeting: MeetingDetail, model: Optional[str], reporter: JobReporter) -> None:
        reporter.step("Gerando ata com IA local (Ollama)...", status="summarizing", progress=82)
        meeting.summary = await self.minutes.generate(MinutesRequest(
            segments=meeting.segments,
            title=meeting.title,
            duration_minutes=meeting.audio_duration / 60.0,
            meeting_date=_meeting_date(meeting),
            context=meeting.context,
            speaker_map=meeting.speaker_map,
            speaker_name_sources=meeting.speaker_name_sources,
            model=model,
            on_progress=reporter.log,
        ))
        apply_speaker_suggestions(meeting, reporter.log)
        await asyncio.to_thread(self.repository.save, meeting)

        self.jobs.update(reporter.job_id, meeting_id=meeting.id, result=present_meeting(meeting).model_dump())
        reporter.step("Processamento concluído! Reunião salva no banco local.", status="completed", progress=100,
                      level="success")
        logger.info("[%s] Reunião %s salva.", reporter.job_id, meeting.id)


def _meeting_date(meeting: MeetingDetail) -> datetime:
    try:
        return datetime.fromisoformat(meeting.created_at)
    except ValueError:
        return datetime.now().astimezone()
