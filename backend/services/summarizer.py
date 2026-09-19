import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from backend.config import settings
from backend.models.schemas import MeetingMinutes, TopicItem, ActionItem, SpeakerSegment

logger = logging.getLogger(__name__)


class SummarizerService:
    @staticmethod
    async def get_available_ollama_models() -> List[str]:
        """Obtém a lista de modelos baixados no Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Não foi possível conectar ao Ollama ({settings.OLLAMA_BASE_URL}): {e}")
        return []

    @classmethod
    async def generate_minutes(
        cls,
        segments: List[SpeakerSegment],
        title: str = "Reunião de Alinhamento",
        model_name: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        duration_minutes: float = 0.0
    ) -> MeetingMinutes:
        """
        Gera a Ata de Reunião completa e estruturada em Português do Brasil.
        Usa o Ollama com prompt especializado ou o gerador heurístico local se Ollama estiver indisponível.
        """
        model = model_name or settings.OLLAMA_MODEL

        # Formata o diálogo da reunião
        dialogue_lines = []
        participants_set = set()
        for s in segments:
            dialogue_lines.append(f"[{s.speaker}]: {s.text}")
            participants_set.add(s.speaker)
        dialogue_text = "\n".join(dialogue_lines)
        participants_list = sorted(list(participants_set))

        # Tenta gerar com Ollama
        try:
            return await cls._generate_with_ollama(
                dialogue_text=dialogue_text,
                participants=participants_list,
                title=title,
                model=model,
                custom_prompt=custom_prompt,
                duration_minutes=duration_minutes
            )
        except Exception as e:
            logger.warning(f"Falha na geração via Ollama ({e}). Usando gerador estruturado de contingência...")
            return cls._generate_fallback(
                segments=segments,
                title=title,
                participants=participants_list,
                duration_minutes=duration_minutes
            )

    @classmethod
    async def pull_model(cls, model_name: str):
        """Dispara download de um modelo no Ollama."""
        async with httpx.AsyncClient(timeout=3600.0) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/pull",
                json={"model": model_name, "name": model_name, "stream": True}
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        err_json = json.loads(body.decode("utf-8"))
                        err_msg = err_json.get("error") or body.decode("utf-8")
                    except Exception:
                        err_msg = body.decode("utf-8")
                    raise Exception(f"Ollama retornou status {resp.status_code}: {err_msg}")

                last_status = "Download em andamento..."
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            raise Exception(data["error"])
                        if "status" in data:
                            last_status = data["status"]
                            if "completed" in data and "total" in data and data["total"] > 0:
                                pct = (data["completed"] / data["total"]) * 100
                                logger.info(f"Ollama pull {model_name}: {last_status} ({pct:.1f}%)")
                    except json.JSONDecodeError:
                        pass
                return {"status": "success", "detail": last_status}

    @classmethod
    async def _generate_with_ollama(
        cls,
        dialogue_text: str,
        participants: List[str],
        title: str,
        model: str,
        custom_prompt: Optional[str],
        duration_minutes: float
    ) -> MeetingMinutes:
        system_instruction = (
            "Você é um secretário executivo de alto nível e especialista em governança corporativa e gestão ágil. "
            "Sua tarefa é analisar a transcrição de uma reunião em Português do Brasil e produzir uma ATA DE REUNIÃO "
            "profissional, clara, concisa e altamente estruturada.\n\n"
            "DICA DE LOCUTORES: Analise as falas para identificar se os participantes se chamam pelos nomes próprios "
            "(ex: 'Oi Ariel', 'Carlos, você pode falar?', 'Aqui é a Mariana'). Se conseguir inferir com clareza o nome real de algum Locutor, "
            "preencha no campo 'suggested_speakers' associando o identificador ao nome real (ex: 'Locutor 1': 'Ariel'). "
            "Se não for possível inferir, mantenha o identificador original.\n\n"
            "Retorne APENAS um objeto JSON válido estritamente com a seguinte estrutura:\n"
            "{\n"
            '  "title": "Título claro e objetivo da reunião",\n'
            '  "executive_summary": "Resumo executivo em 1 a 2 parágrafos com o propósito e principais conclusões",\n'
            '  "suggested_speakers": {\n'
            '    "Locutor 1": "Nome Real 1 ou Locutor 1",\n'
            '    "Locutor 2": "Nome Real 2 ou Locutor 2"\n'
            '  },\n'
            '  "main_topics": [\n'
            '    {\n'
            '      "title": "Nome do tópico",\n'
            '      "discussion": "O que foi debatido e argumentos apresentados",\n'
            '      "conclusions": "Conclusão deste tópico"\n'
            '    }\n'
            '  ],\n'
            '  "decisions": [\n'
            '    "Decisão tomada 1",\n'
            '    "Decisão tomada 2"\n'
            '  ],\n'
            '  "action_items": [\n'
            '    {\n'
            '      "task": "Ação específica a ser realizada",\n'
            '      "owner": "Nome do responsável ou Quem se comprometeu",\n'
            '      "deadline": "Prazo acordado ou A definir",\n'
            '      "status": "Pendente"\n'
            '    }\n'
            '  ],\n'
            '  "open_points": [\n'
            '    "Dúvida ou ponto que ficou para a próxima reunião"\n'
            '  ]\n'
            "}"
        )

        user_content = f"TRANSCRICÃO DA REUNIÃO:\n{dialogue_text}\n\nPARTICIPANTES IDENTIFICADOS:\n{', '.join(participants)}"
        if custom_prompt:
            user_content += f"\n\nINSTRUÇÕES ADICIONAIS DO USUÁRIO:\n{custom_prompt}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        # Dimensionamento adaptativo da janela de contexto
        estimated_tokens = int(len(dialogue_text) / 2.8) + 2500
        max_ctx = getattr(settings, "OLLAMA_NUM_CTX", 32768)
        num_ctx = min(max_ctx, max(4096, estimated_tokens))
        logger.info(f"Ollama num_ctx adaptativo dimensionado para {num_ctx} tokens (teto configurado: {max_ctx})")

        async with httpx.AsyncClient(timeout=600.0) as client:
            raw_response = ""
            try:
                # 1. Chamada via /api/chat: o Ollama aplica automaticamente o template nativo correto (Gemma, Llama, Qwen)
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_ctx": num_ctx
                        }
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("message", {}).get("content", "").strip()
            except Exception as e_chat:
                logger.warning(f"Chamada via /api/chat falhou ({e_chat}). Tentando fallback via /api/generate...")
                # Fallback para /api/generate caso o endpoint /api/chat não responda
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model,
                        "system": system_instruction,
                        "prompt": user_content,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_ctx": num_ctx
                        }
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("response", "").strip()

            # Parser seguro de JSON
            parsed = cls._parse_llm_json(raw_response)

            # Construir os itens tipados
            topics = [TopicItem(**t) for t in parsed.get("main_topics", [])]
            actions = [ActionItem(**a) for a in parsed.get("action_items", [])]
            decisions = [str(d) for d in parsed.get("decisions", [])]
            open_points = [str(o) for o in parsed.get("open_points", [])]
            suggested_speakers = parsed.get("suggested_speakers", {})
            if not isinstance(suggested_speakers, dict):
                suggested_speakers = {}

            now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
            final_title = parsed.get("title") or title
            exec_summary = parsed.get("executive_summary", "Reunião realizada com sucesso.")

            # Montar Markdown formatado elegante
            markdown_content = cls._build_markdown(
                title=final_title,
                date=now_str,
                duration_minutes=duration_minutes,
                participants=participants,
                executive_summary=exec_summary,
                topics=topics,
                decisions=decisions,
                actions=actions,
                open_points=open_points
            )

            return MeetingMinutes(
                title=final_title,
                date=now_str,
                duration_minutes=duration_minutes,
                participants=participants,
                executive_summary=exec_summary,
                main_topics=topics,
                decisions=decisions,
                action_items=actions,
                open_points=open_points,
                suggested_speakers=suggested_speakers,
                raw_markdown=markdown_content
            )

    @staticmethod
    def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
        """Tenta fazer parse de JSON puro ou dentro de blocos ```json."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            logger.warning("Falha ao fazer parse de JSON retornado pelo LLM.")
            return {}

    @classmethod
    def _generate_fallback(
        cls,
        segments: List[SpeakerSegment],
        title: str,
        participants: List[str],
        duration_minutes: float
    ) -> MeetingMinutes:
        """
        Gerador Heurístico Local:
        Gera uma ata estruturada extraindo pontos-chave mesmo sem LLM online.
        """
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

        # Palavras-chave indicativas de ação e decisão em PT-BR
        action_keywords = ["vamos", "vou", "precisamos", "fazer", "tarefa", "enviar", "entregar", "verificar", "alinhar", "criar", "desenvolver"]
        decision_keywords = ["decidido", "fechado", "combinado", "aprovado", "concordo", "definido", "escolhemos"]

        actions: List[ActionItem] = []
        decisions: List[str] = []
        topics: List[TopicItem] = []

        for seg in segments:
            text_lower = seg.text.lower()
            if any(k in text_lower for k in decision_keywords):
                decisions.append(f"{seg.speaker}: {seg.text}")
            elif any(k in text_lower for k in action_keywords):
                actions.append(ActionItem(
                    task=seg.text,
                    owner=seg.speaker,
                    deadline="A definir",
                    status="Pendente"
                ))

        if not decisions:
            decisions = ["Alinhamento geral dos itens discutidos durante a sessão."]
        if not actions:
            actions = [ActionItem(task="Revisão dos pontos alinhados", owner=participants[0] if participants else "Equipe")]

        topics.append(TopicItem(
            title="Debates Gerais e Alinhamento",
            discussion=f"A reunião contou com a participação de {len(participants)} interlocutor(es). Foram abordados os temas registrados na transcrição integral.",
            conclusions="Continuidade das ações designadas."
        ))

        exec_summary = (
            f"Reunião com duração estimada de {duration_minutes:.1f} minutos entre {', '.join(participants)}. "
            f"Foram discutidos temas centrais com foco em alinhamento e definição de próximos passos operacionais."
        )

        markdown_content = cls._build_markdown(
            title=title,
            date=now_str,
            duration_minutes=duration_minutes,
            participants=participants,
            executive_summary=exec_summary,
            topics=topics,
            decisions=decisions,
            actions=actions,
            open_points=[]
        )

        return MeetingMinutes(
            title=title,
            date=now_str,
            duration_minutes=duration_minutes,
            participants=participants,
            executive_summary=exec_summary,
            main_topics=topics,
            decisions=decisions,
            action_items=actions,
            open_points=[],
            raw_markdown=markdown_content
        )

    @staticmethod
    def _build_markdown(
        title: str,
        date: str,
        duration_minutes: float,
        participants: List[str],
        executive_summary: str,
        topics: List[TopicItem],
        decisions: List[str],
        actions: List[ActionItem],
        open_points: List[str]
    ) -> str:
        md = [
            f"# 📋 Ata de Reunião: {title}",
            "",
            f"**📅 Data e Hora:** {date}  ",
            f"**⏱️ Duração:** {duration_minutes:.1f} minutos  ",
            f"**👥 Participantes:** {', '.join(participants)}",
            "",
            "---",
            "",
            "## 📌 Resumo Executivo",
            executive_summary,
            "",
            "## 💬 Tópicos Discutidos",
        ]

        for idx, t in enumerate(topics, 1):
            md.append(f"### {idx}. {t.title}")
            md.append(f"**Discussão:** {t.discussion}")
            if t.conclusions:
                md.append(f"**Conclusão:** {t.conclusions}")
            md.append("")

        md.append("## ⚖️ Decisões Tomadas")
        if decisions:
            for d in decisions:
                md.append(f"- ✅ {d}")
        else:
            md.append("_Nenhuma decisão formal registrada._")
        md.append("")

        md.append("## 🚀 Plano de Ação (Tarefas)")
        if actions:
            md.append("| Tarefa / Ação | Responsável | Prazo | Status |")
            md.append("| :--- | :--- | :--- | :--- |")
            for a in actions:
                md.append(f"| {a.task} | **{a.owner}** | {a.deadline} | `{a.status}` |")
        else:
            md.append("_Nenhuma tarefa atribuída especificamente._")
        md.append("")

        if open_points:
            md.append("## ❓ Pontos em Aberto / Próximos Passos")
            for p in open_points:
                md.append(f"- ⏳ {p}")
            md.append("")

        return "\n".join(md)
