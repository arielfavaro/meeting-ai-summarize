"""
Download ÚNICO dos modelos de IA para uso 100% offline.

Rode uma vez (com internet), depois deixe HF_HUB_OFFLINE=1: em processamento nenhum
áudio, texto ou metadado sai da máquina.

    docker compose run --rm -e HF_HUB_OFFLINE=0 app \
        python -m backend.scripts.download_models --whisper medium --hf-token hf_xxx

- Whisper (Faster-Whisper/CTranslate2) → data/models_cache (mesmo cache usado pela aplicação)
- SpeechBrain ECAPA-TDNN (público)     → data/models_cache/speechbrain_ecapa
- pyannote speaker-diarization-community-1 (gratuito, mas exige aceitar os termos em
  https://huggingface.co/pyannote/speaker-diarization-community-1 e um token de leitura)
                                        → data/models_cache/pyannote-speaker-diarization-community-1

O token é usado só aqui, para o download, e não é gravado em lugar nenhum.
"""
import argparse
import os
import sys

# O download precisa de rede: desliga o modo offline APENAS neste processo.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from backend.config import settings  # noqa: E402

PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"
SPEECHBRAIN_REPO = "speechbrain/spkrec-ecapa-voxceleb"


def download_whisper(sizes):
    from faster_whisper.utils import download_model
    for size in sizes:
        print(f"→ Whisper '{size}'...")
        path = download_model(size, cache_dir=str(settings.MODELS_CACHE_DIR))
        print(f"  ok: {path}")


def download_speechbrain():
    from huggingface_hub import snapshot_download
    target = settings.speechbrain_model_dir
    print(f"→ SpeechBrain ECAPA-TDNN → {target}")
    snapshot_download(SPEECHBRAIN_REPO, local_dir=str(target))
    print("  ok")


def download_pyannote(token):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    target = settings.pyannote_model_dir
    print(f"→ pyannote community-1 → {target}")
    try:
        snapshot_download(PYANNOTE_REPO, local_dir=str(target), token=token)
    except GatedRepoError:
        print("  ✗ Acesso negado: aceite os termos em https://huggingface.co/" + PYANNOTE_REPO +
              " com a mesma conta do token e rode de novo.")
        return False
    except HfHubHTTPError as e:
        print(f"  ✗ Falha no download: {e}")
        return False
    if not (target / "config.yaml").exists():
        print("  ✗ config.yaml não encontrado após o download.")
        return False
    print("  ok")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Baixa os modelos para uso 100% offline.")
    parser.add_argument("--whisper", default=settings.WHISPER_MODEL_SIZE,
                        help="Tamanhos do Whisper separados por vírgula (ex.: medium,large-v3). Vazio = pular.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="Token do Hugging Face (só para o pyannote, que é 'gated').")
    parser.add_argument("--skip-speechbrain", action="store_true")
    parser.add_argument("--skip-pyannote", action="store_true")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    ok = True
    sizes = [s.strip() for s in args.whisper.split(",") if s.strip()]
    if sizes:
        download_whisper(sizes)
    if not args.skip_speechbrain:
        download_speechbrain()
    if not args.skip_pyannote:
        if args.hf_token:
            ok = download_pyannote(args.hf_token) and ok
        else:
            print("→ pyannote ignorado (sem --hf-token). A diarização usará o SpeechBrain.")

    print("\nPronto. Para garantir que nada saia da máquina, mantenha HF_HUB_OFFLINE=1 (padrão do docker-compose).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
