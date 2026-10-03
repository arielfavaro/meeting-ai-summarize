import os
import sys
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configurar caminhos para bibliotecas CUDA (CTranslate2 / Faster-Whisper)
for _dir in [Path("/usr/local/lib/python3.10/site-packages"), Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"]:
    _cublas_dir = _dir / "nvidia" / "cublas" / "lib"
    _cudnn_dir = _dir / "nvidia" / "cudnn" / "lib"
    if _cublas_dir.exists() or _cudnn_dir.exists():
        _cur = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_cublas_dir}:{_cudnn_dir}:{_cur}"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.config import settings
from backend.database import MeetingRepository
from backend.dependencies import (
    get_file_store, get_jobs, get_llm_client, get_minutes_generator, get_pipeline, get_repo,
)
from backend.models.schemas import (
    JobStatus, MeetingContext, MeetingDetail, ProcessOptions, RegenerateSummaryRequest, UpdateSpeakersRequest,
)
from backend.services.audio_service import AudioService
from backend.services.exporter import ExporterService
from backend.services.file_store import FileStore, FileTooLargeError, InvalidFileError
from backend.services.jobs import FINAL_STATES, JobRegistry
from backend.services.llm.ollama_client import OllamaClient
from backend.services.minutes.generator import MinutesGenerator, MinutesRequest
from backend.services.pipeline import MeetingPipeline, apply_speaker_suggestions
from backend.services.speakers import present_meeting, sync_display_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("meeting_ai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    get_repo()  # aplica migrações do banco na subida
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Sistema de IA Local para Transcrição, Diarização e Atas de Reunião em PT-BR",
    lifespan=lifespan,
)

# Sem credenciais: allow_origins=["*"] + allow_credentials=True é inseguro (e inválido pela especificação CORS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_meeting(repo: MeetingRepository, meeting_id: str) -> MeetingDetail:
    meeting = repo.get(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")
    return meeting


def _meeting_date(meeting: MeetingDetail) -> datetime:
    try:
        return datetime.fromisoformat(meeting.created_at)
    except ValueError:
        return datetime.now().astimezone()


async def _regenerate(meeting: MeetingDetail, minutes: MinutesGenerator, model: Optional[str],
                      custom_prompt: Optional[str] = None) -> None:
    meeting.summary = await minutes.generate(MinutesRequest(
        segments=meeting.segments,
        title=meeting.title,
        duration_minutes=meeting.audio_duration / 60.0,
        meeting_date=_meeting_date(meeting),
        context=meeting.context,
        speaker_map=meeting.speaker_map,
        speaker_name_sources=meeting.speaker_name_sources,
        model=model,
        custom_prompt=custom_prompt,
    ))
    apply_speaker_suggestions(meeting)


# ----------------------------------------------------------------------------
# Upload e processamento
# ----------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...), files: FileStore = Depends(get_file_store)):
    """Recebe o arquivo em streaming (sem carregar tudo na memória) e armazena localmente."""
    try:
        file_id, size = await files.save_upload(file)
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except InvalidFileError as e:
        raise HTTPException(status_code=400, detail=str(e))

    info = await asyncio.to_thread(AudioService.get_audio_info, files.upload_path(file_id))
    return {"file_id": file_id, "original_name": file.filename, "duration": info.get("duration", 0.0), "size_bytes": size}


@app.post("/api/process")
async def start_processing(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    title: str = Form("Reunião"),
    whisper_model: Optional[str] = Form(None),
    language: str = Form("pt"),
    ollama_model: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    objective: Optional[str] = Form(None),
    meeting_type: str = Form("geral"),
    participants: Optional[str] = Form(None),
    glossary: Optional[str] = Form(None),
    files: FileStore = Depends(get_file_store),
    jobs: JobRegistry = Depends(get_jobs),
    pipeline: MeetingPipeline = Depends(get_pipeline),
):
    """Inicia a esteira completa de diarização, transcrição e geração de ata."""
    try:
        files.upload_path(file_id)  # valida antes de enfileirar
    except InvalidFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        options = ProcessOptions(
            whisper_model=whisper_model or settings.WHISPER_MODEL_SIZE,
            language=language,
            ollama_model=ollama_model,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=hf_token,
            context=MeetingContext(
                objective=(objective or "").strip() or None,
                meeting_type=meeting_type,
                participants=participants,
                glossary=glossary,
                custom_prompt=(custom_prompt or "").strip() or None,
            ),
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))

    job = jobs.create()
    background_tasks.add_task(pipeline.run, job.job_id, file_id, title, options)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str, jobs: JobRegistry = Depends(get_jobs)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")
    return job


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_status(job_id: str, jobs: JobRegistry = Depends(get_jobs)):
    """Server-Sent Events (SSE): envia o estado apenas quando ele muda."""
    if not jobs.get(job_id):
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")

    async def event_generator():
        last_seen = -1.0
        while True:
            job = jobs.get(job_id)
            if not job:
                break
            if job.updated_at != last_seen:
                last_seen = job.updated_at
                yield f"data: {job.model_dump_json()}\n\n"
            else:
                yield ": keep-alive\n\n"
            if job.status in FINAL_STATES:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ----------------------------------------------------------------------------
# Reuniões e histórico
# ----------------------------------------------------------------------------
@app.get("/api/meetings")
async def list_meetings(repo: MeetingRepository = Depends(get_repo)):
    return repo.list()


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    return present_meeting(_load_meeting(repo, meeting_id))


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str, repo: MeetingRepository = Depends(get_repo)):
    if not repo.delete(meeting_id):
        raise HTTPException(status_code=404, detail="Reunião não encontrada.")
    return {"deleted": True, "id": meeting_id}


@app.put("/api/meetings/{meeting_id}/speakers")
async def update_meeting_speakers(
    meeting_id: str,
    req: UpdateSpeakersRequest,
    repo: MeetingRepository = Depends(get_repo),
    minutes: MinutesGenerator = Depends(get_minutes_generator),
):
    """
    Renomeia locutores pelo ID estável (ex.: {"Locutor 1": "Mariana"}).
    A ata e as exportações refletem o novo nome imediatamente, sem regerar via LLM.
    """
    meeting = _load_meeting(repo, meeting_id)
    ids = set(meeting.speaker_ids())
    by_display = {meeting.display_name(sid): sid for sid in ids}

    for key, new_name in req.speaker_map.items():
        sid = key if key in ids else by_display.get(key)  # aceita o nome atual por compatibilidade
        if not sid:
            raise HTTPException(status_code=400, detail=f"Locutor '{key}' não existe nesta reunião.")
        new_name = (new_name or "").strip()
        if new_name and new_name != sid:
            if new_name != meeting.speaker_map.get(sid):
                meeting.speaker_name_sources[sid] = "user"
            meeting.speaker_map[sid] = new_name
        else:
            meeting.speaker_map[sid] = sid
            meeting.speaker_name_sources[sid] = "default"
    sync_display_names(meeting)

    if req.regenerate_summary:
        await _regenerate(meeting, minutes, req.ollama_model)

    repo.save(meeting)
    return present_meeting(meeting)


@app.post("/api/meetings/{meeting_id}/regenerate-summary")
async def regenerate_summary(
    meeting_id: str,
    req: RegenerateSummaryRequest,
    repo: MeetingRepository = Depends(get_repo),
    minutes: MinutesGenerator = Depends(get_minutes_generator),
):
    meeting = _load_meeting(repo, meeting_id)
    await _regenerate(meeting, minutes, req.ollama_model, req.custom_prompt)
    repo.save(meeting)
    return present_meeting(meeting)


# ----------------------------------------------------------------------------
# Exportação
# ----------------------------------------------------------------------------
def _safe_title(title: str) -> str:
    return "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip() or "Reuniao"


@app.get("/api/meetings/{meeting_id}/export/{format}")
async def export_meeting(meeting_id: str, format: str, repo: MeetingRepository = Depends(get_repo)):
    """Exporta a ata para Markdown (.md), Word (.docx) ou Texto (.txt)."""
    meeting = present_meeting(_load_meeting(repo, meeting_id))
    filename_base = f"Ata_{_safe_title(meeting.title)}_{datetime.now().strftime('%Y%m%d')}"

    if format == "md":
        return Response(content=ExporterService.to_markdown(meeting), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'})
    if format == "txt":
        return Response(content=ExporterService.to_plain_text(meeting), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_base}.txt"'})
    if format == "docx":
        return StreamingResponse(
            ExporterService.to_docx_bytes(meeting),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'})
    raise HTTPException(status_code=400, detail=f"Formato '{format}' não suportado (use 'md', 'docx' ou 'txt').")


@app.get("/api/meetings/{meeting_id}/export-transcript/{format}")
async def export_meeting_transcript(meeting_id: str, format: str, repo: MeetingRepository = Depends(get_repo)):
    """Exporta a transcrição para Markdown (.md), Word (.docx), Texto (.txt) ou Legenda (.srt)."""
    meeting = present_meeting(_load_meeting(repo, meeting_id))
    if not meeting.segments:
        raise HTTPException(status_code=400, detail="Reunião não possui transcrição disponível.")
    filename_base = f"Transcricao_{_safe_title(meeting.title)}_{datetime.now().strftime('%Y%m%d')}"

    if format == "md":
        return Response(content=ExporterService.transcript_to_markdown(meeting), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_base}.md"'})
    if format == "txt":
        return Response(content=ExporterService.transcript_to_plain_text(meeting), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_base}.txt"'})
    if format == "docx":
        return StreamingResponse(
            ExporterService.transcript_to_docx_bytes(meeting),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'})
    if format == "srt":
        return Response(content=ExporterService.transcript_to_srt(meeting), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename_base}.srt"'})
    raise HTTPException(status_code=400,
                        detail=f"Formato '{format}' não suportado para transcrição (use 'md', 'docx', 'txt' ou 'srt').")


# ----------------------------------------------------------------------------
# Áudio para o player
# ----------------------------------------------------------------------------
@app.get("/api/audio/{filename}")
async def serve_audio(filename: str, files: FileStore = Depends(get_file_store)):
    try:
        path = files.audio_path(filename)
    except InvalidFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo de áudio não encontrado.")
    return FileResponse(path, media_type="audio/wav")


# ----------------------------------------------------------------------------
# Modelos e diagnóstico
# ----------------------------------------------------------------------------
@app.get("/api/models/whisper")
async def list_whisper_models():
    return [
        {"id": "tiny", "name": "Tiny (Mais rápido, 39M params)", "recommended_for": "Testes rápidos"},
        {"id": "base", "name": "Base (Equilibrado e leve, 74M params)", "recommended_for": "CPUs comuns"},
        {"id": "small", "name": "Small (244M params)", "recommended_for": "Boa precisão / velocidade"},
        {"id": "medium", "name": "Medium (Recomendado para PT-BR, 769M params)", "recommended_for": "Reuniões complexas / termos técnicos"},
        {"id": "large-v3", "name": "Large v3 (Precisão máxima, 1550M params)", "recommended_for": "Uso com GPU"},
    ]


@app.get("/api/models/ollama")
async def list_ollama_models(llm: OllamaClient = Depends(get_llm_client)):
    models = await llm.list_models()
    return {"ollama_url": settings.OLLAMA_BASE_URL, "available_models": models, "connected": len(models) > 0}


@app.post("/api/models/ollama/pull")
async def pull_ollama_model(payload: dict, llm: OllamaClient = Depends(get_llm_client)):
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Nome do modelo não informado.")
    try:
        res = await llm.pull_model(model_name)
        return {"status": "success", "model": model_name, "detail": res}
    except Exception as e:
        logger.error("Erro ao baixar modelo %s: %s", model_name, e)
        raise HTTPException(status_code=400, detail=f"Falha ao baixar modelo '{model_name}': {e}")


@app.get("/api/health")
async def health_check(llm: OllamaClient = Depends(get_llm_client)):
    ollama_models = await llm.list_models()
    return {
        "status": "online",
        "version": settings.APP_VERSION,
        "whisper_device": settings.WHISPER_DEVICE,
        "ollama_connected": len(ollama_models) > 0,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "ollama_models_count": len(ollama_models),
    }


# Frontend estático
if settings.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")
