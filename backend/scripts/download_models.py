"""
Download ÚNICO dos modelos de IA para uso 100% offline.

Rode uma vez (com internet), depois deixe HF_HUB_OFFLINE=1: em processamento nenhum
áudio, texto ou metadado sai da máquina.

    docker compose run --rm -e HF_HUB_OFFLINE=0 app \
        python -m backend.scripts.download_models --whisper medium --hf-token hf_xxx

- Whisper (Faster-Whisper/CTranslate2) → data/models_cache (mesmo cache usado pela aplicação)
- SpeechBrain ECAPA-TDNN (público)     → data/models_cache/speechbrain_ecapa
- wav2vec2 XLSR-53 português (público, alinhamento forçado das palavras)
                                        → data/models_cache/wav2vec2-large-xlsr-53-portuguese
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
        try:  # já no disco? (sem rede)
            path = download_model(size, cache_dir=str(settings.MODELS_CACHE_DIR), local_files_only=True)
            print(f"  já presente no disco (nada a baixar): {path}")
            continue
        except Exception:  # noqa: BLE001 - não está no cache: baixa
            pass
        path = download_model(size, cache_dir=str(settings.MODELS_CACHE_DIR))
        print(f"  ok: {path}")
    return True


SPEECHBRAIN_REQUIRED = ("hyperparams.yaml", "embedding_model.ckpt")


def _files_ready(target, names) -> bool:
    """Arquivos presentes e legíveis (symlinks apontando para algo que existe)."""
    return all((target / n).exists() for n in names)


def _replace_symlinks(target) -> int:
    """
    Versões antigas do app deixavam symlinks do SpeechBrain apontando para o cache do
    Hugging Face; o snapshot_download não consegue copiar um arquivo "sobre ele mesmo"
    (SameFileError). Remove só os links — o conteúdo continua no cache e é recopiado.
    """
    removed = 0
    if target.exists():
        for entry in target.iterdir():
            if entry.is_symlink():
                entry.unlink()
                removed += 1
    return removed


def download_speechbrain(token=None):
    from huggingface_hub import snapshot_download
    target = settings.speechbrain_model_dir
    print(f"→ SpeechBrain ECAPA-TDNN → {target}")
    if _files_ready(target, SPEECHBRAIN_REQUIRED):
        print("  já presente no disco (nada a baixar)")
        return True
    if _replace_symlinks(target):
        print("  links antigos para o cache substituídos por cópias locais")
    snapshot_download(SPEECHBRAIN_REPO, local_dir=str(target), token=token or None)
    print("  ok")
    return True


ALIGNER_REQUIRED = ("config.json", "vocab.json", "preprocessor_config.json")


def download_aligner(token=None):
    from huggingface_hub import snapshot_download
    target = settings.alignment_model_dir
    print(f"→ wav2vec2 (alinhamento de palavras) → {target}")
    has_weights = any(target.glob("*.bin")) or any(target.glob("*.safetensors"))
    if _files_ready(target, ALIGNER_REQUIRED) and has_weights:
        print("  já presente no disco (nada a baixar)")
        return True
    _replace_symlinks(target)
    repo = settings.ALIGNMENT_MODEL_REPO
    # Só um formato de pesos (safetensors se houver) e nada do modelo de linguagem (KenLM), que não é usado.
    try:
        from huggingface_hub import list_repo_files
        files = set(list_repo_files(repo, token=token or None))
    except Exception:  # noqa: BLE001 - sem a listagem, baixa o .bin (formato mais comum nesses repositórios)
        files = set()
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    snapshot_download(repo, local_dir=str(target), token=token or None,
                      allow_patterns=["*.json", weights], ignore_patterns=["language_model/*"])
    if not (target / "config.json").exists() or not (target / weights).exists():
        print(f"  ✗ arquivos do modelo incompletos em {target}")
        return False
    print("  ok")
    return True


def download_pyannote(token):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    target = settings.pyannote_model_dir
    print(f"→ pyannote community-1 → {target}")
    if (target / "config.yaml").exists():
        print("  já presente no disco (nada a baixar)")
        return True
    _replace_symlinks(target)
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
    parser = argparse.ArgumentParser(
        description="Baixa os modelos para uso 100% offline. Idempotente: o que já está no disco é "
                    "verificado e pulado, então o mesmo comando serve para instalar do zero ou completar "
                    "o que faltar. As opções --skip-* só servem para economizar download/tempo.")
    parser.add_argument("--whisper", default=settings.WHISPER_MODEL_SIZE,
                        help="Tamanhos do Whisper separados por vírgula (padrão: WHISPER_MODEL_SIZE, "
                             f"hoje '{settings.WHISPER_MODEL_SIZE}'). Ex.: medium,large-v3.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="Token do Hugging Face (só para o pyannote, que é 'gated').")
    parser.add_argument("--skip-whisper", action="store_true", help="Não verificar/baixar o Whisper.")
    parser.add_argument("--skip-speechbrain", action="store_true", help="Não verificar/baixar o SpeechBrain.")
    parser.add_argument("--skip-pyannote", action="store_true", help="Não verificar/baixar o pyannote.")
    parser.add_argument("--skip-aligner", action="store_true", help="Não verificar/baixar o wav2vec2 de alinhamento.")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    results = {}

    def step(name, fn, *fn_args):
        # Cada modelo é independente: a falha de um não impede o download dos outros.
        try:
            results[name] = fn(*fn_args) is not False
        except Exception as e:  # noqa: BLE001 - relatório amigável no fim
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            results[name] = False

    sizes = [] if args.skip_whisper else [s.strip() for s in args.whisper.split(",") if s.strip()]
    if sizes:
        step("Whisper", download_whisper, sizes)
    if not args.skip_speechbrain:
        step("SpeechBrain", download_speechbrain, args.hf_token)
    if not args.skip_aligner and settings.ALIGNMENT_ENGINE.lower() == "wav2vec2":
        step("Alinhamento (wav2vec2)", download_aligner, args.hf_token)
    if not args.skip_pyannote:
        if args.hf_token or (settings.pyannote_model_dir / "config.yaml").exists():
            step("pyannote", download_pyannote, args.hf_token)  # já presente: só confirma, sem token
        else:
            print("→ pyannote ignorado (sem --hf-token e não está no disco). A diarização usará o SpeechBrain.")

    print("\nResumo: " + ", ".join(f"{k} {'ok' if v else 'FALHOU'}" for k, v in results.items()))
    print("Para garantir que nada saia da máquina, mantenha HF_HUB_OFFLINE=1 (padrão do docker-compose).")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
