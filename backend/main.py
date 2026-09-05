import os
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

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
from backend.services.alignment import align_transcription_with_diarization
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
    """Pipeline assíncrono executado em segundo plano."""
    job = JOBS[job_id]
    try:
        raw_audio_path = settings.UPLOAD_DIR / file_id
        if not raw_audio_path.exists():
            raise FileNotFoundError(f"Arquivo {file_id} não encontrado.")

        # Passo 1: Pré-processamento e normalização com FFmpeg
        job.status = "preprocessing"
        job.progress = 10
        job.current_step = "Convertendo áudio para 16kHz mono (FFmpeg)..."
        logger.info(f"[{job_id}] {job.current_step}")

        wav_filename = f"{Path(file_id).stem}_16k.wav"
        processed_wav_path = settings.PROCESSED_DIR / wav_filename
        wav_path, duration = AudioService.convert_to_wav_16k_mono(raw_audio_path, processed_wav_path)
        duration_minutes = duration / 60.0

        # Passo 2: Diarização (Separação de locutores)
        job.status = "diarizing"
        job.progress = 30
        job.current_step = "Identificando e separando locutores no áudio..."
        logger.info(f"[{job_id}] {job.current_step}")

        diarizer = DiarizationService(hf_token=options.hf_token)
        diarization_segments = diarizer.diarize(
            audio_path=wav_path,
            min_speakers=options.min_speakers,
            max_speakers=options.max_speakers
        )

        # Passo 3: Transcrição com Faster-Whisper com atualização de progresso
        job.status = "transcribing"
        job.progress = 35
        job.current_step = f"Transcrevendo fala em Português (Faster-Whisper '{options.whisper_model}')..."
        logger.info(f"[{job_id}] {job.current_step}")

        def on_transcribe_progress(current_sec: float, total_sec: float):
            if total_sec > 0:
                pct = min(99, int((current_sec / total_sec) * 100))
                # Interpola progresso entre 35% e 70%
                job.progress = 35 + int((pct / 100) * 35)
                job.current_step = f"Transcrevendo: {pct}% ({current_sec/60:.1f}/{total_sec/60:.1f} min)..."

        transcription_raw = TranscriptionService.transcribe(
            audio_path=wav_path,
            model_size=options.whisper_model,
            language=options.language,
            progress_callback=on_transcribe_progress
        )

        # Passo 4: Alinhamento de transcrição e locutores
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
        raise HTTPException(status_code=500, detail=f"Falha ao baixar modelo '{model_name}': {str(e)}")


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
FRONTEND_DIR = settings.ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
