"""
Resolve prazos relativos em PT-BR ("sexta-feira", "amanhã", "dia 15", "em 2 semanas")
para datas absolutas, usando a data da reunião como referência.

Estratégia conservadora: se a expressão for ambígua, devolve None e o texto original
é mantido na ata.
"""
import calendar
import re
import unicodedata
from datetime import date, timedelta
from typing import Optional

WEEKDAYS = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4, "sabado": 5, "domingo": 6}
MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower()).strip()


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _next_weekday(ref: date, weekday: int) -> date:
    delta = (weekday - ref.weekday()) % 7
    return ref + timedelta(days=delta or 7)


def resolve_deadline(text: str, reference: date) -> Optional[date]:
    t = _norm(text)
    if not t or t in ("a definir", "nao definido", "sem prazo"):
        return None

    if "depois de amanha" in t:
        return reference + timedelta(days=2)
    if re.search(r"\bamanha\b", t):
        return reference + timedelta(days=1)
    if re.search(r"\bhoje\b", t):
        return reference

    m = re.search(r"\b(?:em|daqui a|dentro de)\s+(\d{1,3})\s+(dia|dias|semana|semanas|mes|meses)\b", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("dia"):
            return reference + timedelta(days=n)
        if unit.startswith("semana"):
            return reference + timedelta(weeks=n)
        month = reference.month - 1 + n
        year = reference.year + month // 12
        month = month % 12 + 1
        day = min(reference.day, calendar.monthrange(year, month)[1])
        return date(year, month, day)

    if re.search(r"\b(fim|final) d[ao] mes\b", t):
        return date(reference.year, reference.month, calendar.monthrange(reference.year, reference.month)[1])
    if re.search(r"\b(fim|final) da semana\b", t):
        return _next_weekday(reference, 4) if reference.weekday() != 4 else reference

    # dd/mm[/aaaa]
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else reference.year
        if y < 100:
            y += 2000
        candidate = _safe_date(y, mo, d)
        if candidate and not m.group(3) and candidate < reference:
            candidate = _safe_date(y + 1, mo, d)
        return candidate

    # "15 de outubro"
    m = re.search(r"\b(\d{1,2}) de (" + "|".join(MONTHS) + r")\b", t)
    if m:
        d, mo = int(m.group(1)), MONTHS[m.group(2)]
        candidate = _safe_date(reference.year, mo, d)
        if candidate and candidate < reference:
            candidate = _safe_date(reference.year + 1, mo, d)
        return candidate

    # "dia 15"
    m = re.search(r"\bdia (\d{1,2})\b", t)
    if m:
        d = int(m.group(1))
        candidate = _safe_date(reference.year, reference.month, d)
        if candidate and candidate < reference:
            nm = reference.month % 12 + 1
            ny = reference.year + (1 if reference.month == 12 else 0)
            candidate = _safe_date(ny, nm, d)
        return candidate

    # Dias da semana ("sexta", "sexta-feira", "próxima segunda", "terça que vem")
    m = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", t)
    if m:
        return _next_weekday(reference, WEEKDAYS[m.group(1)])

    return None
