"""
Orquestração do processamento de uma reunião (caso de uso "ProcessMeeting").

As etapas pesadas (FFmpeg, diarização, Whisper) são síncronas e rodam em threads via
`asyncio.to_thread`, para não bloquear o event loop (SSE, health check e demais rotas
continuam respondendo). Um semáforo limita quantos pipelines rodam ao mesmo tempo.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from backend.models.schemas import MeetingDetail, ProcessOptions, SpeakerSegment
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
        diarizer_factory: Callable[[Optional[str]], DiarizationService] = DiarizationService,
        transcriber=TranscriptionService,
        audio=AudioService,
    ):
        self.jobs = jobs
        self.files = files
        self.repository = repository
        self.minutes = minutes
        self.processed_dir = processed_dir
        self.diarizer_factory = diarizer_factory
        self.transcriber = transcriber
        self.audio = audio

    async def run(self, job_id: str, file_id: str, title: str, options: ProcessOptions) -> None:
        reporter = JobReporter(self.jobs, job_id, logger)
        try:
            if self.jobs.slots.locked():
                reporter.log("Aguardando outro processamento terminar...")
            async with self.jobs.slots:
                await self._run(file_id, title, options, reporter)
        except Exception as e:
            logger.exception("Erro durante processamento do job %s", job_id)
            reporter.log(f"Falha no processamento: {e}", "error", status="failed", progress=100, error=str(e))

    async def _run(self, file_id: str, title: str, options: ProcessOptions, reporter: JobReporter) -> None:
        raw_audio_path = self.files.upload_path(file_id)
        terms = [*options.context.participants, *options.context.glossary]

        reporter.step("Analisando faixas de áudio (FFmpeg)...", status="preprocessing", progress=5)
        active_tracks = await asyncio.to_thread(self.audio.detect_active_audio_tracks, raw_audio_path,
                                                reporter.callback())
        wav_filename = f"{Path(file_id).stem}_16k.wav"
        reporter.step("Gerando áudio mestre 16kHz mono para o player...", progress=10)
        wav_path, duration = await asyncio.to_thread(
            self.audio.convert_to_wav_16k_mono, raw_audio_path, self.processed_dir / wav_filename, active_tracks
        )
        reporter.step(f"Áudio mestre pronto: {duration / 60:.1f} min.", progress=15, level="success")
        diarizer = self.diarizer_factory(options.hf_token)

        if len(active_tracks) > 1:
            aligned = await self._process_multitrack(raw_audio_path, file_id, active_tracks, diarizer, options, terms, reporter)
        else:
            aligned = await self._process_single(wav_path, diarizer, options, terms, reporter)

        reporter.step("Gerando ata com IA local (Ollama)...", status="summarizing", progress=82)
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
        )
        meeting.summary = await self.minutes.generate(MinutesRequest(
            segments=meeting.segments,
            title=meeting.title,
            duration_minutes=duration / 60.0,
            meeting_date=created_at,
            context=meeting.context,
            speaker_map=meeting.speaker_map,
            speaker_name_sources=meeting.speaker_name_sources,
            model=options.ollama_model,
            on_progress=reporter.log,
        ))
        apply_speaker_suggestions(meeting, reporter.log)
        await asyncio.to_thread(self.repository.save, meeting)

        self.jobs.update(reporter.job_id, meeting_id=meeting.id, result=present_meeting(meeting).model_dump())
        reporter.step("Processamento concluído! Reunião salva no banco local.", status="completed", progress=100,
                      level="success")

    async def _process_single(self, wav_path: Path, diarizer, options: ProcessOptions, terms: List[str],
                              reporter: JobReporter) -> List[SpeakerSegment]:
        reporter.step("Identificando e separando locutores (diarização local)...", status="diarizing", progress=20)
        diarization = await asyncio.to_thread(diarizer.diarize, wav_path, options.min_speakers, options.max_speakers,
                                              reporter.callback())

        reporter.step(f"Transcrevendo fala (Faster-Whisper '{options.whisper_model}')...", status="transcribing",
                      progress=35)
        on_progress = _transcription_progress(reporter, base=35, span=35, label="Transcrição")
        transcription = await asyncio.to_thread(
            self.transcriber.transcribe, wav_path, options.whisper_model, options.language, 5, on_progress, terms
        )
        reporter.step("Sincronizando falas e locutores (alinhamento por palavra)...", status="aligning", progress=75)
        aligned = align_transcription_with_diarization(transcription, diarization)
        reporter.step(f"Alinhamento concluído: {len(aligned)} falas atribuídas.", progress=80, level="success")
        return aligned

    async def _process_multitrack(self, raw_audio_path: Path, file_id: str, tracks: List[int], diarizer,
                                  options: ProcessOptions, terms: List[str], reporter: JobReporter
                                  ) -> List[SpeakerSegment]:
        n = len(tracks)
        reporter.step(f"Gravação multi-faixa: {n} faixas ativas {tracks}.", progress=15)
        results = []
        for idx, track_no in enumerate(tracks):
            label = f"[Faixa {idx + 1}/{n}]"
            base = 15 + int(idx / n * 60)
            reporter.step(f"{label} Extraindo stream #{track_no} e separando locutores...", status="diarizing",
                          progress=base)
            track_path = self.processed_dir / f"{Path(file_id).stem}_track_{track_no}_16k.wav"
            await asyncio.to_thread(self.audio.extract_track_to_wav_16k, raw_audio_path, track_no, track_path)

            # Em faixas individuais não forçamos min_speakers: um microfone dedicado deve virar 1 locutor.
            diarization = await asyncio.to_thread(diarizer.diarize, track_path, None, options.max_speakers,
                                                  reporter.callback(f"{label} "))

            span = 60 / n
            reporter.step(f"{label} Transcrevendo (Faster-Whisper '{options.whisper_model}')...", status="transcribing",
                          progress=base + int(0.2 * span))
            on_progress = _transcription_progress(reporter, base=base + int(0.2 * span), span=0.8 * span,
                                                  label=f"{label} Transcrição")
            transcription = await asyncio.to_thread(
                self.transcriber.transcribe, track_path, options.whisper_model, options.language, 5, on_progress, terms
            )
            aligned = align_transcription_with_diarization(transcription, diarization)
            results.append(aligned)
            reporter.log(f"{label} Concluída: {len(aligned)} falas sincronizadas.", "success")

        reporter.step(f"Intercalando as {n} faixas em ordem cronológica...", status="aligning", progress=75)
        merged = merge_multitrack_segments(results)
        reporter.step(f"Intercalação concluída: {len(merged)} falas.", progress=80, level="success")
        return merged
