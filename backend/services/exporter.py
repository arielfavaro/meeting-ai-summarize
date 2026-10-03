import io
from pathlib import Path
from typing import Optional
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from backend.models.schemas import MeetingDetail
from backend.services.minutes.renderer import (
    OBJECTIVE_STATUS_LABEL, format_date, format_datetime, format_timestamp, render_minutes_markdown,
)


class ExporterService:
    @staticmethod
    def to_markdown(meeting: MeetingDetail) -> str:
        """Ata completa em Markdown (espera uma reunião já apresentada: nomes resolvidos)."""
        if not meeting.summary:
            return "# Ata não disponível."

        content = [render_minutes_markdown(meeting)]
        content.append("\n\n---\n## 🎙️ Transcrição Completa dos Diálogos\n")
        for seg in meeting.segments:
            content.append(f"**[{format_timestamp(seg.start)}] {seg.speaker}:** {seg.text}  ")
        return "\n".join(content)

    @staticmethod
    def to_plain_text(meeting: MeetingDetail) -> str:
        """Gera versão em texto simples."""
        if not meeting.summary:
            return "Ata não disponível."

        s = meeting.summary
        lines = [
            f"ATA DE REUNIÃO: {s.title.upper()}",
            f"Data: {format_datetime(s.date or meeting.created_at)}",
            f"Duração: {meeting.audio_duration / 60:.1f} minutos",
            f"Participantes: {', '.join(s.participants)}",
        ]
        if s.source == "heuristic":
            lines.append("AVISO: ata gerada em modo de contingência (sem LLM). Revise antes de compartilhar.")
        lines += [f"AVISO: {w}" for w in s.warnings]
        lines.append("=" * 60)

        n = 1
        if s.objectives:
            lines.append(f"\n{n}. OBJETIVOS DA REUNIÃO:")
            for o in s.objectives:
                status = OBJECTIVE_STATUS_LABEL.get(o.status, o.status)
                lines.append(f"  - {o.description} [{status}]" + (f" — {o.notes}" if o.notes else ""))
            n += 1

        lines += [f"\n{n}. RESUMO EXECUTIVO:", s.executive_summary]
        n += 1

        lines.append(f"\n{n}. TÓPICOS DISCUTIDOS:")
        for i, t in enumerate(s.main_topics, 1):
            lines.append(f"  {i}. {t.title}")
            lines.append(f"     Discussão: {t.discussion}")
            if t.conclusions:
                lines.append(f"     Conclusão: {t.conclusions}")
        n += 1

        lines.append(f"\n{n}. DECISÕES TOMADAS:")
        lines += [f"  - {d.description}" for d in s.decisions] or ["  (nenhuma)"]
        n += 1

        lines.append(f"\n{n}. PLANO DE AÇÃO:")
        for a in s.action_items:
            prazo = a.deadline + (f" ({format_date(a.due_date)})" if a.due_date else "")
            lines.append(f"  - [ ] {a.task} | Responsável: {a.owner} | Prazo: {prazo} | Status: {a.status}")
        n += 1

        if s.open_points:
            lines.append(f"\n{n}. PONTOS EM ABERTO:")
            lines += [f"  - {p.description}" for p in s.open_points]
            n += 1
        if s.risks:
            lines.append(f"\n{n}. RISCOS E IMPEDIMENTOS:")
            lines += [f"  - {p.description}" for p in s.risks]

        lines.append("\n" + "=" * 60)
        lines.append("TRANSCRIÇÃO DETALHADA:")
        for seg in meeting.segments:
            lines.append(f"[{format_timestamp(seg.start)}] {seg.speaker}: {seg.text}")
        return "\n".join(lines)

    @staticmethod
    def to_docx_bytes(meeting: MeetingDetail) -> io.BytesIO:
        """Gera um arquivo DOCX (Microsoft Word) estilizado profissionalmente."""
        doc = Document()
        for section in doc.sections:
            section.top_margin = section.bottom_margin = Inches(1)
            section.left_margin = section.right_margin = Inches(1)

        summary = meeting.summary

        title_p = doc.add_paragraph()
        title_run = title_p.add_run(summary.title if summary else meeting.title)
        title_run.font.name = "Arial"
        title_run.font.size = Pt(20)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(30, 41, 59)
        title_p.paragraph_format.space_after = Pt(4)

        sub_p = doc.add_paragraph()
        date_str = format_datetime(summary.date if summary and summary.date else meeting.created_at)
        sub_run = sub_p.add_run(f"Data: {date_str} | Duração: {meeting.audio_duration / 60:.1f} min")
        sub_run.font.size = Pt(10)
        sub_run.font.italic = True
        sub_run.font.color.rgb = RGBColor(100, 116, 139)
        sub_p.paragraph_format.space_after = Pt(14)

        if summary:
            for warning in ([
                "Ata gerada em modo de contingência (sem LLM). Revise antes de compartilhar."
            ] if summary.source == "heuristic" else []) + summary.warnings:
                w_run = doc.add_paragraph().add_run(f"⚠ {warning}")
                w_run.font.size = Pt(9)
                w_run.font.color.rgb = RGBColor(180, 83, 9)

            p_part = doc.add_paragraph()
            p_part.add_run("Participantes: ").bold = True
            p_part.add_run(", ".join(summary.participants))
            p_part.paragraph_format.space_after = Pt(12)

            n = 1
            if summary.objectives:
                doc.add_heading(f"{n}. Objetivos da Reunião", level=1)
                for o in summary.objectives:
                    p = doc.add_paragraph(style="List Bullet")
                    p.add_run(o.description).bold = True
                    p.add_run(f" — {OBJECTIVE_STATUS_LABEL.get(o.status, o.status)}")
                    if o.notes:
                        p.add_run(f": {o.notes}")
                n += 1

            h1 = doc.add_heading(f"{n}. Resumo Executivo", level=1)
            h1.paragraph_format.space_before = Pt(12)
            doc.add_paragraph(summary.executive_summary).paragraph_format.space_after = Pt(12)
            n += 1

            doc.add_heading(f"{n}. Tópicos Discutidos", level=1)
            for idx, topic in enumerate(summary.main_topics, 1):
                doc.add_paragraph().add_run(f"{idx}. {topic.title}").bold = True
                doc.add_paragraph(f"Discussão: {topic.discussion}")
                if topic.conclusions:
                    doc.add_paragraph(f"Conclusão: {topic.conclusions}")
            n += 1

            doc.add_heading(f"{n}. Decisões Tomadas", level=1)
            for dec in summary.decisions:
                doc.add_paragraph(dec.description, style="List Bullet")
            n += 1

            doc.add_heading(f"{n}. Plano de Ação", level=1)
            if summary.action_items:
                table = doc.add_table(rows=1, cols=4)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for i, title in enumerate(["Tarefa / Ação", "Responsável", "Prazo", "Status"]):
                    cell = table.rows[0].cells[i]
                    cell.text = title
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.bold = True
                            run.font.size = Pt(10)
                for item in summary.action_items:
                    row = table.add_row().cells
                    row[0].text = item.task
                    row[1].text = item.owner
                    row[2].text = item.deadline + (f" ({format_date(item.due_date)})" if item.due_date else "")
                    row[3].text = item.status
            n += 1

            if summary.open_points:
                doc.add_heading(f"{n}. Pontos em Aberto", level=1)
                for pt in summary.open_points:
                    doc.add_paragraph(pt.description, style="List Bullet")
                n += 1
            if summary.risks:
                doc.add_heading(f"{n}. Riscos e Impedimentos", level=1)
                for pt in summary.risks:
                    doc.add_paragraph(pt.description, style="List Bullet")

        doc.add_page_break()
        doc.add_heading("Transcrição Integral da Reunião", level=1)
        for seg in meeting.segments:
            p = doc.add_paragraph()
            time_run = p.add_run(f"[{format_timestamp(seg.start)}] ")
            time_run.font.color.rgb = RGBColor(148, 163, 184)
            time_run.font.size = Pt(9)
            p.add_run(f"{seg.speaker}: ").bold = True
            p.add_run(seg.text)

        target_stream = io.BytesIO()
        doc.save(target_stream)
        target_stream.seek(0)
        return target_stream

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Converte segundos para o formato de tempo do SRT (HH:MM:SS,mmm)."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis = 999
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def transcript_to_markdown(meeting: MeetingDetail) -> str:
        """Retorna a transcrição dos diálogos em formato Markdown."""
        title = meeting.summary.title if meeting.summary else meeting.title
        date_str = format_datetime(meeting.summary.date if meeting.summary and meeting.summary.date else meeting.created_at)
        duration_min = meeting.audio_duration / 60.0
        unique_speakers = list(dict.fromkeys(seg.speaker for seg in meeting.segments))

        content = [
            f"# Transcrição: {title}",
            "",
            f"**Data:** {date_str}  ",
            f"**Duração:** {duration_min:.1f} minutos  ",
            f"**Locutores:** {', '.join(unique_speakers)}  ",
            "",
            "---",
            "",
            "## 🎙️ Diálogos Integrais",
            ""
        ]

        for seg in meeting.segments:
            m = int(seg.start // 60)
            s = int(seg.start % 60)
            time_str = f"{m:02d}:{s:02d}"
            content.append(f"**[{time_str}] {seg.speaker}:** {seg.text}  ")

        return "\n".join(content)

    @staticmethod
    def transcript_to_plain_text(meeting: MeetingDetail) -> str:
        """Gera versão da transcrição em texto simples (.txt)."""
        title = meeting.summary.title if meeting.summary else meeting.title
        date_str = format_datetime(meeting.summary.date if meeting.summary and meeting.summary.date else meeting.created_at)
        duration_min = meeting.audio_duration / 60.0
        unique_speakers = list(dict.fromkeys(seg.speaker for seg in meeting.segments))

        lines = [
            f"TRANSCRIÇÃO DE REUNIÃO: {title.upper()}",
            f"Data: {date_str}",
            f"Duração: {duration_min:.1f} minutos",
            f"Locutores: {', '.join(unique_speakers)}",
            "=" * 60,
            ""
        ]

        for seg in meeting.segments:
            m = int(seg.start // 60)
            s = int(seg.start % 60)
            lines.append(f"[{m:02d}:{s:02d}] {seg.speaker}: {seg.text}")

        return "\n".join(lines)

    @staticmethod
    def transcript_to_docx_bytes(meeting: MeetingDetail) -> io.BytesIO:
        """Gera um arquivo DOCX elegante contendo a transcrição integral dos diálogos."""
        doc = Document()

        for section in doc.sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)

        title = meeting.summary.title if meeting.summary else meeting.title
        date_str = format_datetime(meeting.summary.date if meeting.summary and meeting.summary.date else meeting.created_at)
        duration_min = meeting.audio_duration / 60.0
        unique_speakers = list(dict.fromkeys(seg.speaker for seg in meeting.segments))

        # Título
        title_p = doc.add_paragraph()
        title_run = title_p.add_run(f"Transcrição: {title}")
        title_run.font.name = "Arial"
        title_run.font.size = Pt(20)
        title_run.font.bold = True
        title_run.font.color.rgb = RGBColor(30, 41, 59)
        title_p.paragraph_format.space_after = Pt(4)

        # Subtítulo com metadados
        sub_p = doc.add_paragraph()
        sub_run = sub_p.add_run(
            f"Data: {date_str} | Duração: {duration_min:.1f} min | Locutores: {', '.join(unique_speakers)}"
        )
        sub_run.font.name = "Arial"
        sub_run.font.size = Pt(10)
        sub_run.font.italic = True
        sub_run.font.color.rgb = RGBColor(100, 116, 139)
        sub_p.paragraph_format.space_after = Pt(16)

        # Heading Diálogos
        h1 = doc.add_heading("Diálogos da Reunião", level=1)
        h1.paragraph_format.space_before = Pt(8)
        h1.paragraph_format.space_after = Pt(12)

        for seg in meeting.segments:
            m = int(seg.start // 60)
            s = int(seg.start % 60)
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15

            time_run = p.add_run(f"[{m:02d}:{s:02d}] ")
            time_run.font.name = "Arial"
            time_run.font.size = Pt(9.5)
            time_run.font.color.rgb = RGBColor(148, 163, 184)

            speaker_run = p.add_run(f"{seg.speaker}: ")
            speaker_run.font.name = "Arial"
            speaker_run.font.size = Pt(10.5)
            speaker_run.bold = True
            speaker_run.font.color.rgb = RGBColor(37, 99, 235)

            text_run = p.add_run(seg.text)
            text_run.font.name = "Arial"
            text_run.font.size = Pt(10.5)
            text_run.font.color.rgb = RGBColor(30, 41, 59)

        target_stream = io.BytesIO()
        doc.save(target_stream)
        target_stream.seek(0)
        return target_stream

    @classmethod
    def transcript_to_srt(cls, meeting: MeetingDetail) -> str:
        """Gera legendas sincronizadas no formato universal SRT."""
        srt_blocks = []
        for idx, seg in enumerate(meeting.segments, 1):
            start_str = cls._format_srt_time(seg.start)
            end_str = cls._format_srt_time(seg.end)
            text = f"[{seg.speaker}] {seg.text}"
            srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        return "\n".join(srt_blocks)

