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
from backend.services.jobs import JobRegistry
from backend.services.minutes.generator import MinutesGenerator, MinutesRequest
from backend.services.speakers import present_meeting, sync_display_names
from backend.services.transcription import TranscriptionService

logger = logging.getLogger("meeting_ai.pipeline")


def apply_speaker_suggestions(meeting: MeetingDetail) -> None:
    """Aplica nomes validados pelo `speaker_naming` sem sobrescrever nomes definidos pelo usuário."""
    if not meeting.summary:
        return
    for sug in meeting.summary.speaker_suggestions:
        if sug.applied and meeting.speaker_name_sources.get(sug.speaker_id) != "user":
            logger.info("Nome inferido: '%s' -> '%s' (%s, conf=%.2f)", sug.speaker_id, sug.name, sug.reason, sug.confidence)
            meeting.speaker_map[sug.speaker_id] = sug.name
            meeting.speaker_name_sources[sug.speaker_id] = "llm"
    sync_display_names(meeting)


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
        upd = lambda **kw: self.jobs.update(job_id, **kw)  # noqa: E731
        try:
            if self.jobs.slots.locked():
                upd(current_step="Aguardando outro processamento terminar...")
            async with self.jobs.slots:
                await self._run(job_id, file_id, title, options, upd)
        except Exception as e:
            logger.exception("Erro durante processamento do job %s", job_id)
            upd(status="failed", progress=100, error=str(e), current_step=f"Falha no processamento: {e}")

    async def _run(self, job_id: str, file_id: str, title: str, options: ProcessOptions, upd) -> None:
        raw_audio_path = self.files.upload_path(file_id)
        terms = [*options.context.participants, *options.context.glossary]

        upd(status="preprocessing", progress=5, current_step="Analisando faixas de áudio e convertendo (FFmpeg)...")
        active_tracks = await asyncio.to_thread(self.audio.detect_active_audio_tracks, raw_audio_path)
        wav_filename = f"{Path(file_id).stem}_16k.wav"
        wav_path, duration = await asyncio.to_thread(
            self.audio.convert_to_wav_16k_mono, raw_audio_path, self.processed_dir / wav_filename, active_tracks
        )
        diarizer = self.diarizer_factory(options.hf_token)

        if len(active_tracks) > 1:
            aligned = await self._process_multitrack(raw_audio_path, file_id, active_tracks, diarizer, options, terms, upd)
        else:
            aligned = await self._process_single(wav_path, diarizer, options, terms, upd)

        upd(status="summarizing", progress=85, current_step="Gerando ata com IA local (Ollama)...")
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
        ))
        apply_speaker_suggestions(meeting)
        await asyncio.to_thread(self.repository.save, meeting)

        upd(status="completed", progress=100, current_step="Processamento concluído com sucesso!",
            meeting_id=meeting.id, result=present_meeting(meeting).model_dump())
        logger.info("[%s] Reunião processada e salva com ID: %s", job_id, meeting.id)

    async def _process_single(self, wav_path: Path, diarizer, options: ProcessOptions, terms: List[str], upd
                              ) -> List[SpeakerSegment]:
        upd(status="diarizing", progress=20, current_step="Identificando e separando locutores (diarização local)...")
        diarization = await asyncio.to_thread(diarizer.diarize, wav_path, options.min_speakers, options.max_speakers)

        upd(status="transcribing", progress=35,
            current_step=f"Transcrevendo fala (Faster-Whisper '{options.whisper_model}')...")

        def on_progress(current_sec: float, total_sec: float) -> None:
            if total_sec > 0:
                pct = min(99, int(current_sec / total_sec * 100))
                upd(progress=35 + int(pct * 0.35),
                    current_step=f"Transcrevendo: {pct}% ({current_sec / 60:.1f}/{total_sec / 60:.1f} min)...")

        transcription = await asyncio.to_thread(
            self.transcriber.transcribe, wav_path, options.whisper_model, options.language, 5, on_progress, terms
        )
        upd(status="aligning", progress=75, current_step="Sincronizando falas e locutores...")
        return align_transcription_with_diarization(transcription, diarization)

    async def _process_multitrack(self, raw_audio_path: Path, file_id: str, tracks: List[int], diarizer,
                                  options: ProcessOptions, terms: List[str], upd) -> List[SpeakerSegment]:
        results = []
        n = len(tracks)
        for idx, track_no in enumerate(tracks):
            base = 15 + int(idx / n * 55)
            upd(status="diarizing", progress=base, current_step=f"Processando faixa {idx + 1}/{n} (stream #{track_no})...")
            track_path = self.processed_dir / f"{Path(file_id).stem}_track_{track_no}_16k.wav"
            await asyncio.to_thread(self.audio.extract_track_to_wav_16k, raw_audio_path, track_no, track_path)

            # Em faixas individuais não forçamos min_speakers: um microfone dedicado deve virar 1 locutor.
            diarization = await asyncio.to_thread(diarizer.diarize, track_path, None, options.max_speakers)

            def on_progress(current_sec: float, total_sec: float, t_idx=idx, t_base=base) -> None:
                if total_sec > 0:
                    pct = min(99, int(current_sec / total_sec * 100))
                    upd(status="transcribing", progress=min(70, t_base + int(pct / 100 * (55 / n))),
                        current_step=f"Transcrevendo faixa {t_idx + 1}/{n}: {pct}%...")

            transcription = await asyncio.to_thread(
                self.transcriber.transcribe, track_path, options.whisper_model, options.language, 5, on_progress, terms
            )
            results.append(align_transcription_with_diarization(transcription, diarization))

        upd(status="aligning", progress=75, current_step="Sincronizando e intercalando faixas de áudio...")
        return merge_multitrack_segments(results)
