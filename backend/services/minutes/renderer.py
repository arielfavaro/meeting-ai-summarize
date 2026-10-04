"""Renderização da ata (Markdown / texto) a partir dos dados estruturados."""
from datetime import datetime
from typing import Dict, List, Optional

from backend.clock import to_app_tz
from backend.models.schemas import MeetingDetail

OBJECTIVE_STATUS_LABEL = {
    "atingido": "✅ Atingido",
    "parcial": "🟡 Parcial",
    "nao_atingido": "❌ Não atingido",
    "indefinido": "⚪ Indefinido",
}


def format_datetime(value: Optional[str]) -> str:
    """ISO 8601 -> 'dd/mm/YYYY HH:MM'. Valores legados são devolvidos como estão."""
    if not value:
        return "--"
    try:
        return to_app_tz(datetime.fromisoformat(value)).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return value


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def evidence_refs(evidence: List[int], seg_start: Dict[int, float]) -> str:
    stamps = [format_timestamp(seg_start[e]) for e in evidence if e in seg_start]
    return f" _(ref. {', '.join(stamps[:4])})_" if stamps else ""


def _md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render_minutes_markdown(meeting: MeetingDetail) -> str:
    """Espera uma reunião já apresentada (nomes resolvidos)."""
    s = meeting.summary
    if not s:
        return "# Ata não disponível."
    seg_start = {seg.id: seg.start for seg in meeting.segments}
    ref = lambda ev: evidence_refs(ev, seg_start)  # noqa: E731

    md = [
        f"# 📋 Ata de Reunião: {s.title}",
        "",
        f"**📅 Data e Hora:** {format_datetime(s.date or meeting.created_at)}  ",
        f"**⏱️ Duração:** {meeting.audio_duration / 60:.1f} minutos  ",
        f"**👥 Participantes:** {', '.join(s.participants)}",
        "",
    ]
    if s.source == "heuristic":
        md += ["> ⚠️ Ata gerada pelo modo de contingência (sem LLM). Revise antes de compartilhar.", ""]
    for w in s.warnings:
        md += [f"> ⚠️ {w}", ""]

    md += ["---", ""]

    if s.objectives:
        md.append("## 🎯 Objetivos da Reunião")
        for o in s.objectives:
            label = OBJECTIVE_STATUS_LABEL.get(o.status, o.status)
            line = f"- **{o.description}** — {label}"
            if o.notes:
                line += f": {o.notes}"
            md.append(line + ref(o.evidence))
        md.append("")

    md += ["## 📌 Resumo Executivo", s.executive_summary or "_Sem resumo._", ""]

    md.append("## 💬 Tópicos Discutidos")
    for idx, t in enumerate(s.main_topics, 1):
        md.append(f"### {idx}. {t.title}")
        md.append(f"**Discussão:** {t.discussion}{ref(t.evidence)}")
        if t.conclusions:
            md.append(f"**Conclusão:** {t.conclusions}")
        md.append("")

    md.append("## ⚖️ Decisões Tomadas")
    if s.decisions:
        for d in s.decisions:
            flag = "" if d.grounded else " ⚠️ _sem evidência na transcrição_"
            md.append(f"- ✅ {d.description}{ref(d.evidence)}{flag}")
    else:
        md.append("_Nenhuma decisão formal registrada._")
    md.append("")

    md.append("## 🚀 Plano de Ação (Tarefas)")
    if s.action_items:
        md.append("| Tarefa / Ação | Responsável | Prazo | Status |")
        md.append("| :--- | :--- | :--- | :--- |")
        for a in s.action_items:
            prazo = a.deadline
            if a.due_date:
                prazo = f"{a.deadline} ({format_date(a.due_date)})" if a.deadline not in ("", "A definir") else format_date(a.due_date)
            task = a.task + ("" if a.grounded else " ⚠️")
            md.append(f"| {_md_cell(task)} | **{_md_cell(a.owner)}** | {_md_cell(prazo)} | `{a.status}` |")
    else:
        md.append("_Nenhuma tarefa atribuída especificamente._")
    md.append("")

    if s.open_points:
        md.append("## ❓ Pontos em Aberto / Próximos Passos")
        md += [f"- ⏳ {p.description}{ref(p.evidence)}" for p in s.open_points]
        md.append("")

    if s.risks:
        md.append("## 🚩 Riscos e Impedimentos")
        md += [f"- {p.description}{ref(p.evidence)}" for p in s.risks]
        md.append("")

    return "\n".join(md)


def format_date(iso_date: str) -> str:
    try:
        return datetime.fromisoformat(iso_date).strftime("%d/%m/%Y")
    except ValueError:
        return iso_date
