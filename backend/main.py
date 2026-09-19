import os
import sys
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configurar caminhos para bibliotecas CUDA (CTranslate2 / Faster-Whisper)
for _dir in [Path("/usr/local/lib/python3.10/site-packages"), Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"]:
    _cublas_dir = _dir / "nvidia" / "cublas" / "lib"
    _cudnn_dir = _dir / "nvidia" / "cudnn" / "lib"
    if _cublas_dir.exists() or _cudnn_dir.exists():
        _cur = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_cublas_dir}:{_cudnn_dir}:{_cur}"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.schemas import (
    JobStatus, MeetingDetail, MeetingListItem, ProcessOptions,
    UpdateSpeakersRequest, RegenerateSummaryRequest, SpeakerSegment
)
from backend import database
from backend.services.audio_service import AudioService
from backend.services.diarization import DiarizationService
from backend.services.transcription import TranscriptionService
from backend.services.alignment import align_transcription_with_diarization, merge_multitrack_segments
from backend.services.summarizer import SummarizerService
from backend.services.exporter import ExporterService

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("meeting_ai")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Sistema de IA Local para Transcrição, Diarização e Atas de Reunião em PT-BR"
)

# CORS liberado para acesso local ou de rede
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memória temporária para status dos jobs em execução
JOBS: Dict[str, JobStatus] = {}


async def run_pipeline_task(job_id: str, file_id: str, title: str, options: ProcessOptions):
    """Pipeline assíncrono executado em segundo plano com suporte a faixas ativas múltiplas."""
    job = JOBS[job_id]
    try:
        raw_audio_path = settings.UPLOAD_DIR / file_id
        if not raw_audio_path.exists():
            raise FileNotFoundError(f"Arquivo {file_id} não encontrado.")

        # Passo 1: Pré-processamento e análise de faixas de áudio
        job.status = "preprocessing"
        job.progress = 5
        job.current_step = "Analisando faixas de áudio e convertendo (FFmpeg)..."
        logger.info(f"[{job_id}] {job.current_step}")

        active_tracks = AudioService.detect_active_audio_tracks(raw_audio_path)
        logger.info(f"[{job_id}] Faixas ativas detectadas: {active_tracks}")

        wav_filename = f"{Path(file_id).stem}_16k.wav"
        processed_wav_path = settings.PROCESSED_DIR / wav_filename
        # Gera o áudio mestre para reprodução no player web (com todas as faixas ativas combinadas)
        wav_path, duration = AudioService.convert_to_wav_16k_mono(raw_audio_path, processed_wav_path, active_tracks=active_tracks)
        duration_minutes = duration / 60.0

        if len(active_tracks) > 1:
            job.current_step = f"Processando gravação multi-faixa ({len(active_tracks)} faixas de áudio ativas)..."
            logger.info(f"[{job_id}] {job.current_step}")

            track_aligned_results = []
            diarizer = DiarizationService(hf_token=options.hf_token)

            for idx, track_no in enumerate(active_tracks):
                track_step_base = 15 + int((idx / len(active_tracks)) * 55)
                job.progress = track_step_base
                job.current_step = f"Processando faixa {idx + 1}/{len(active_tracks)} (stream #{track_no})..."
                logger.info(f"[{job_id}] {job.current_step}")

                # Extrai a faixa individual para arquivo temporário de 16kHz
                track_wav_filename = f"{Path(file_id).stem}_track_{track_no}_16k.wav"
                track_wav_path = settings.PROCESSED_DIR / track_wav_filename
                AudioService.extract_track_to_wav_16k(raw_audio_path, track_no, track_wav_path)

                # Diarização da faixa individual
                # Em arquivos multi-faixa, não forçamos min_speakers alto em canais individuais
                # permitindo que um microfone permaneça como 1 orador único (k=1)
                track_diar_segments = diarizer.diarize(
                    audio_path=track_wav_path,
                    min_speakers=None,
                    max_speakers=options.max_speakers
                )

                # Transcrição da faixa individual
                def on_track_progress(current_sec: float, total_sec: float, t_idx=idx, base=track_step_base):
                    if total_sec > 0:
                        pct = min(99, int((current_sec / total_sec) * 100))
                        progress_increment = int((pct / 100) * (55 / len(active_tracks)))
                        job.progress = min(70, base + progress_increment)
                        job.current_step = f"Transcrevendo faixa {t_idx + 1}/{len(active_tracks)}: {pct}% ({current_sec/60:.1f}/{total_sec/60:.1f} min)..."

                track_transcription = TranscriptionService.transcribe(
                    audio_path=track_wav_path,
                    model_size=options.whisper_model,
                    language=options.language,
                    progress_callback=on_track_progress
                )

                # Alinha a transcrição da faixa com sua diarização
                track_aligned = align_transcription_with_diarization(
                    transcription_segments=track_transcription,
                    diarization_segments=track_diar_segments
                )
                track_aligned_results.append(track_aligned)

            # Combina e intercala cronologicamente as faixas
            job.status = "aligning"
            job.progress = 75
            job.current_step = "Sincronizando e intercalando faixas de áudio..."
            logger.info(f"[{job_id}] {job.current_step}")

            aligned_segments = merge_multitrack_segments(track_aligned_results)

        else:
            # Fluxo padrão de faixa única
            job.status = "diarizing"
            job.progress = 30
            job.current_step = "Identificando e separando locutores (Diarização Local)..."
            logger.info(f"[{job_id}] {job.current_step}")

            diarizer = DiarizationService(hf_token=options.hf_token)
            diarization_segments = diarizer.diarize(
                audio_path=wav_path,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers
            )

            job.status = "transcribing"
            job.progress = 35
            job.current_step = f"Transcrevendo fala em Português (Faster-Whisper '{options.whisper_model}')..."
            logger.info(f"[{job_id}] {job.current_step}")

            def on_transcribe_progress(current_sec: float, total_sec: float):
                if total_sec > 0:
                    pct = min(99, int((current_sec / total_sec) * 100))
                    job.progress = 35 + int((pct / 100) * 35)
                    job.current_step = f"Transcrevendo: {pct}% ({current_sec/60:.1f}/{total_sec/60:.1f} min)..."

            transcription_raw = TranscriptionService.transcribe(
                audio_path=wav_path,
                model_size=options.whisper_model,
                language=options.language,
                progress_callback=on_transcribe_progress
            )

            job.status = "aligning"
            job.progress = 75
            job.current_step = "Sincronizando falas e oradores..."
            logger.info(f"[{job_id}] {job.current_step}")

            aligned_segments = align_transcription_with_diarization(
                transcription_segments=transcription_raw,
                diarization_segments=diarization_segments
            )

        # Passo 5: Geração da Ata de Reunião com Ollama Local
        job.status = "summarizing"
        job.progress = 85
        job.current_step = "Gerando Ata Executiva com IA Local (Ollama)..."
        logger.info(f"[{job_id}] {job.current_step}")

        meeting_id = str(uuid.uuid4())
        summary = await SummarizerService.generate_minutes(
            segments=aligned_segments,
            title=title or f"Reunião {datetime.now().strftime('%d/%m/%Y')}",
            model_name=options.ollama_model,
            custom_prompt=options.custom_prompt,
            duration_minutes=duration_minutes
        )

        # Mapeamento inicial de locutores
        unique_speakers = sorted(list(set(s.speaker for s in aligned_segments)))
        speaker_map = {s: s for s in unique_speakers}

        # Aplicar sugestões de nomes reais inferidos pelo Ollama se houver alta confiança
        if summary and hasattr(summary, "suggested_speakers") and summary.suggested_speakers:
            for spk_id, real_name in summary.suggested_speakers.items():
                real_name_clean = str(real_name).strip()
                if spk_id in speaker_map and real_name_clean and real_name_clean != spk_id:
                    logger.info(f"[{job_id}] Nome real inferido: '{spk_id}' -> '{real_name_clean}'")
                    speaker_map[spk_id] = real_name_clean
            # Sincronizar os identificadores dos segmentos
            for seg in aligned_segments:
                if seg.speaker in speaker_map:
                    seg.speaker = speaker_map[seg.speaker]

        meeting_detail = MeetingDetail(
            id=meeting_id,
            title=title or summary.title,
            created_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
            audio_filename=wav_filename,
            audio_duration=duration,
            audio_url=f"/api/audio/{wav_filename}",
            segments=aligned_segments,
            summary=summary,
            speaker_map=speaker_map
        )

        # Salvar no banco SQLite
        database.save_meeting(meeting_detail)

        job.status = "completed"
        job.progress = 100
        job.current_step = "Processamento concluído com sucesso!"
        job.meeting_id = meeting_id
        job.result = meeting_detail
        logger.info(f"[{job_id}] Reunião processada e salva com ID: {meeting_id}")

    except Exception as e:
        logger.exception(f"Erro durante processamento do job {job_id}: {e}")
        job.status = "failed"
        job.progress = 100
        job.error = str(e)
        job.current_step = f"Falha no processamento: {str(e)}"


# Rotas de Upload e Processamento
@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Recebe o arquivo de áudio gravado ou carregado e armazena localmente."""
    try:
        content = await file.read()
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo excede o limite máximo permitido de {settings.MAX_FILE_SIZE_MB}MB."
            )

        file_ext = Path(file.filename).suffix or ".wav"
        file_id = f"{uuid.uuid4()}{file_ext}"
        destination = settings.UPLOAD_DIR / file_id

        with open(destination, "wb") as buffer:
            buffer.write(content)

        info = AudioService.get_audio_info(destination)

        return {
            "file_id": file_id,
            "original_name": file.filename,
            "duration": info.get("duration", 0.0),
            "size_bytes": len(content)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload de arquivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    title: str = Form("Reunião"),
    whisper_model: str = Form("small"),
    language: str = Form("pt"),
    ollama_model: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None)
):
    """Inicia a esteira completa de diarização, transcrição e geração de ata."""
    job_id = str(uuid.uuid4())
    options = ProcessOptions(
        whisper_model=whisper_model,
        language=language,
        ollama_model=ollama_model,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        custom_prompt=custom_prompt,
        hf_token=hf_token
    )

    job = JobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        current_step="Na fila para processamento..."
    )
    JOBS[job_id] = job

    background_tasks.add_task(run_pipeline_task, job_id, file_id, title, options)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Consulta o status atual de uma tarefa em execução."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Server-Sent Events (SSE) para atualização em tempo real do status na interface."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")

    async def event_generator():
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
            data = job.model_dump_json()
            yield f"data: {data}\n\n"
            if job.status in ("completed", "failed"):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Rotas de Reuniões e Histórico
@app.get("/api/meetings")
async def list_meetings():
    """Retorna o histórico de todas as reuniões salvas."""
    return database.list_meetings()


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Obtém detalhes completos de uma reunião (diálogo com oradores e ata)."""
    meeting = database.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")
    return meeting


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """Remove uma reunião do histórico."""
    success = database.delete_meeting(meeting_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")
    return {"deleted": True, "id": meeting_id}


@app.put("/api/meetings/{meeting_id}/speakers")
async def update_meeting_speakers(meeting_id: str, req: UpdateSpeakersRequest):
    """
    Renomeia os interlocutores da reunião (ex: 'Locutor 1' -> 'Mariana').
    Atualiza todos os segmentos e opcionalmente regera a ata com os nomes novos.
    """
    meeting = database.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")

    # Atualiza o mapa e os segmentos
    meeting.speaker_map.update(req.speaker_map)
    for seg in meeting.segments:
        if seg.speaker in req.speaker_map:
            seg.speaker = req.speaker_map[seg.speaker]

    # Se solicitado, regera a ata com os novos nomes
    if req.regenerate_summary:
        summary = await SummarizerService.generate_minutes(
            segments=meeting.segments,
            title=meeting.title,
            model_name=req.ollama_model,
            duration_minutes=meeting.audio_duration / 60.0
        )
        meeting.summary = summary
    elif meeting.summary:
        # Atualiza apenas a lista de participantes na ata existente
        meeting.summary.participants = sorted(list(set(s.speaker for s in meeting.segments)))

    database.save_meeting(meeting)
    return meeting


@app.post("/api/meetings/{meeting_id}/regenerate-summary")
async def regenerate_summary(meeting_id: str, req: RegenerateSummaryRequest):
    """Regera a Ata de Reunião com outro modelo do Ollama ou prompt customizado."""
    meeting = database.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")

    summary = await SummarizerService.generate_minutes(
        segments=meeting.segments,
        title=meeting.title,
        model_name=req.ollama_model,
        custom_prompt=req.custom_prompt,
        duration_minutes=meeting.audio_duration / 60.0
    )
    meeting.summary = summary
    database.save_meeting(meeting)
    return meeting


# Rotas de Exportação
@app.get("/api/meetings/{meeting_id}/export/{format}")
async def export_meeting(meeting_id: str, format: str):
    """Exporta a ata para Markdown (.md), Word (.docx) ou Texto (.txt)."""
    meeting = database.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")

    safe_title = "".join(c for c in meeting.title if c.isalnum() or c in (" ", "_", "-")).strip()
    filename_base = f"Ata_{safe_title}_{datetime.now().strftime('%Y%m%d')}"

    if format == "md":
        content = ExporterService.to_markdown(meeting)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'}
        )
    elif format == "txt":
        content = ExporterService.to_plain_text(meeting)
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.txt"'}
        )
    elif format == "docx":
        stream = ExporterService.to_docx_bytes(meeting)
        return StreamingResponse(
            stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Formato '{format}' não suportado (use 'md', 'docx' ou 'txt').")


# Áudio Estático para Player Interativo
@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Transmite o arquivo de áudio para o player da página web."""
    path = settings.PROCESSED_DIR / filename
    if not path.exists():
        path = settings.UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo de áudio não encontrado.")
    return FileResponse(path, media_type="audio/wav")


# Rotas de Modelos e Diagnóstico
@app.get("/api/models/whisper")
async def list_whisper_models():
    """Modelos de transcrição Faster-Whisper recomendados."""
    return [
        {"id": "tiny", "name": "Tiny (Mais rápido, 39M params)", "recommended_for": "Testes rápidos"},
        {"id": "base", "name": "Base (Equilibrado e leve, 74M params)", "recommended_for": "CPUs comuns"},
        {"id": "small", "name": "Small (Recomendado para PT-BR, 244M params)", "recommended_for": "Excelente precisão / velocidade"},
        {"id": "medium", "name": "Medium (Alta precisão, 769M params)", "recommended_for": "Reuniões complexas / Termos técnicos"},
        {"id": "large-v3", "name": "Large v3 (Precisão máxima, 1550M params)", "recommended_for": "Uso com GPU ou servidores dedicados"}
    ]


@app.get("/api/models/ollama")
async def list_ollama_models():
    """Busca dinâmica dos modelos instalados no Ollama."""
    models = await SummarizerService.get_available_ollama_models()
    return {
        "ollama_url": settings.OLLAMA_BASE_URL,
        "available_models": models,
        "connected": len(models) > 0
    }


@app.post("/api/models/ollama/pull")
async def pull_ollama_model(payload: dict):
    """Baixa um modelo diretamente no Ollama."""
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Nome do modelo não informado.")
    try:
        res = await SummarizerService.pull_model(model_name)
        return {"status": "success", "model": model_name, "detail": res}
    except Exception as e:
        logger.error(f"Erro ao baixar modelo {model_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Falha ao baixar modelo '{model_name}': {str(e)}")


@app.get("/api/health")
async def health_check():
    """Checagem geral de saúde dos serviços."""
    ollama_models = await SummarizerService.get_available_ollama_models()
    return {
        "status": "online",
        "version": settings.APP_VERSION,
        "whisper_device": settings.WHISPER_DEVICE,
        "ollama_connected": len(ollama_models) > 0,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_models_count": len(ollama_models)
    }


# Montar frontend estático
FRONTEND_DIR = getattr(settings, "FRONTEND_DIR", Path(__file__).resolve().parent.parent / "frontend")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
