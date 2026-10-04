"""
Relógio da aplicação no fuso configurado (APP_TIMEZONE, padrão America/Sao_Paulo).

O container roda em UTC; sem isso, logs, data da ata e prazos ("amanhã", "sexta") ficavam
3 horas adiantados em relação ao horário do usuário.
"""
import logging
from datetime import datetime, tzinfo
from functools import lru_cache

from backend.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def app_timezone() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(settings.APP_TIMEZONE)
    except Exception:  # fuso inválido ou base de fusos ausente: usa o do sistema
        logger.warning("Fuso horário '%s' indisponível; usando o do sistema.", settings.APP_TIMEZONE)
        return datetime.now().astimezone().tzinfo


def now() -> datetime:
    return datetime.now(app_timezone())


def to_app_tz(value: datetime) -> datetime:
    """Converte datas com fuso para o fuso da aplicação (datas sem fuso ficam como estão)."""
    return value.astimezone(app_timezone()) if value.tzinfo else value
