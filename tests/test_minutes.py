"""Testes do motor de ata (sem LLM real): saída estruturada, verificação, map-reduce e contingência."""
import _env  # noqa: F401  (configura ambiente isolado)

import json
import unittest
from datetime import date, datetime

from backend.models.schemas import MeetingContext
from backend.services.llm.base import LLMResponse
from backend.services.minutes.deadline_resolver import resolve_deadline
from backend.services.minutes.generator import MinutesConfig, MinutesGenerator, MinutesRequest
from backend.services.minutes.llm_schemas import max_output_tokens
from backend.services.minutes.transcript_formatter import chunk_segments, format_segment
from fakes import FakeLLM, minutes_payload, sample_segments, seg

MEETING_DATE = datetime(2026, 10, 2, 10, 0)  # sexta-feira


def make_request(**kw) -> MinutesRequest:
    segments = kw.pop("segments", sample_segments())
    ids = list(dict.fromkeys(s.speaker_id for s in segments))
    return MinutesRequest(
        segments=segments, title="Reunião", duration_minutes=10, meeting_date=MEETING_DATE,
        speaker_map={i: i for i in ids}, speaker_name_sources={i: "default" for i in ids}, **kw,
    )


def make_generator(llm, **cfg) -> MinutesGenerator:
    return MinutesGenerator(llm, MinutesConfig(default_model="fake", **cfg))


class TestSinglePass(unittest.IsolatedAsyncioTestCase):
    async def test_structured_minutes_are_verified(self):
        llm = FakeLLM([minutes_payload()])
        minutes = await make_generator(llm).generate(make_request())

        self.assertEqual(minutes.source, "llm")
        self.assertEqual(minutes.strategy, "single_pass")
        self.assertEqual(minutes.title, "Definição da data de deploy")
        # JSON Schema enviado ao Ollama (saída estruturada)
        self.assertIn("properties", llm.calls[0]["schema"])
        # Transcrição com IDs referenciáveis e rótulos estáveis
        self.assertIn("[#3 00:08] Locutor 3:", llm.calls[0]["messages"][1]["content"])

        # Duplicata de tarefa unida, evidências preservadas
        tasks = [a.task for a in minutes.action_items]
        self.assertEqual(len(minutes.action_items), 2, tasks)
        self.assertEqual(minutes.action_items[0].evidence, [2, 3])
        # Prazo relativo resolvido a partir da data da reunião
        self.assertEqual(minutes.action_items[0].due_date, "2026-10-09")
        # Evidência inexistente descartada -> item marcado sem evidência + aviso
        self.assertFalse(minutes.action_items[1].grounded)
        self.assertEqual(minutes.action_items[1].owner, "Não atribuído")
        self.assertTrue(any("sem evidência" in w for w in minutes.warnings))

        self.assertEqual(minutes.objectives[0].status, "atingido")
        self.assertEqual(minutes.decisions[0].evidence, [4])

    async def test_speaker_names_are_validated(self):
        minutes = await make_generator(FakeLLM([minutes_payload()])).generate(make_request())
        applied = {s.speaker_id: s.name for s in minutes.speaker_suggestions if s.applied}
        # Auto-apresentação e "chamado pelo nome + respondeu em seguida" são aceitos;
        # o Locutor 2 (quem CHAMOU a Mariana) é rejeitado.
        self.assertEqual(applied, {"Locutor 1": "Carlos", "Locutor 3": "Mariana"})

    async def test_name_called_but_no_reply_is_rejected(self):
        payload = minutes_payload(speaker_names=[
            {"speaker_id": "Locutor 1", "name": "Mariana", "kind": "chamado_pelo_nome",
             "confidence": 0.95, "evidence": [2]},
        ])
        minutes = await make_generator(FakeLLM([payload])).generate(make_request())
        self.assertFalse(minutes.speaker_suggestions[0].applied)

    async def test_user_names_are_not_overridden(self):
        req = make_request()
        req.speaker_map["Locutor 1"] = "Beto"
        req.speaker_name_sources["Locutor 1"] = "user"
        llm = FakeLLM([minutes_payload()])
        minutes = await make_generator(llm).generate(req)
        sug = next(s for s in minutes.speaker_suggestions if s.speaker_id == "Locutor 1")
        self.assertFalse(sug.applied)
        # O nome confirmado é informado ao modelo
        self.assertIn("Locutor 1 — nome confirmado: Beto", llm.calls[0]["messages"][1]["content"])

    async def test_owner_given_as_confirmed_name_is_normalized_to_label(self):
        req = make_request()
        req.speaker_map["Locutor 3"] = "Mariana"
        payload = minutes_payload(action_items=[
            {"task": "Validar testes", "owner": "mariana", "deadline": "amanhã", "evidence": [3]}])
        minutes = await make_generator(FakeLLM([payload])).generate(req)
        self.assertEqual(minutes.action_items[0].owner, "Locutor 3")
        self.assertEqual(minutes.action_items[0].due_date, "2026-10-03")

    async def test_informed_objective_is_always_present(self):
        req = make_request(context=MeetingContext(objective="Aprovar orçamento", meeting_type="planejamento"))
        llm = FakeLLM([minutes_payload()])
        minutes = await make_generator(llm).generate(req)
        self.assertEqual(minutes.objectives[0].origin, "informado")
        self.assertIn("Aprovar orçamento", minutes.objectives[0].description)
        # Objetivo e orientação do tipo de reunião chegam ao prompt
        self.assertIn("Objetivo/pauta informado pelo usuário: Aprovar orçamento", llm.calls[0]["messages"][1]["content"])
        self.assertIn("TIPO: Planejamento", llm.calls[0]["messages"][0]["content"])

    async def test_retry_on_invalid_json(self):
        llm = FakeLLM(["isto não é json", {"title": "faltam campos"}, minutes_payload()])
        minutes = await make_generator(llm, max_retries=2).generate(make_request())
        self.assertEqual(minutes.source, "llm")
        self.assertEqual(len(llm.calls), 3)
        # O erro de validação é devolvido ao modelo na nova tentativa
        self.assertIn("não passou na validação", llm.calls[2]["messages"][-1]["content"])

    async def test_window_is_planned_from_schema_before_calling(self):
        """Orçamento da resposta vem do pior caso do schema: sem gerar, estourar e refazer."""
        llm = FakeLLM([minutes_payload()])
        progress = []
        await make_generator(llm, num_predict=1000).generate(
            make_request(on_progress=lambda m, lvl="info": progress.append(m)))
        call = llm.calls[0]
        expected_budget = max_output_tokens(call["schema"])
        self.assertEqual(call["num_predict"], expected_budget)
        self.assertGreater(expected_budget, 4096)                 # o antigo 4096 cortava atas grandes
        self.assertEqual(call["num_ctx"] % 2048, 0)                # janela estável (evita recarga do modelo)
        self.assertGreaterEqual(call["num_ctx"], expected_budget)
        self.assertTrue(any(m.startswith("Plano da ata: passada única") for m in progress), progress)
        # Schema ajustado à reunião: rótulos reais como enum e listas limitadas
        speaker_schema = call["schema"]["$defs"]["LLMSpeakerName"]["properties"]["speaker_id"]
        self.assertEqual(speaker_schema["enum"], ["Locutor 1", "Locutor 2", "Locutor 3"])
        self.assertEqual(call["schema"]["properties"]["topics"]["maxItems"], 8)
        self.assertIn("LIMITES DE TAMANHO", call["messages"][0]["content"])

    async def test_retries_keep_the_same_window(self):
        llm = FakeLLM(["isto não é json", minutes_payload()])
        await make_generator(llm).generate(make_request())
        self.assertEqual(llm.calls[0]["num_ctx"], llm.calls[1]["num_ctx"])

    async def test_truncated_output_is_salvaged_without_regenerating(self):
        full = json.dumps(minutes_payload(), ensure_ascii=False)
        cut = full[: full.index('"open_points"') + 30]  # cortada no meio dos pontos em aberto
        llm = FakeLLM([LLMResponse(content=cut, done_reason="length")])
        minutes = await make_generator(llm).generate(make_request())
        self.assertEqual(len(llm.calls), 1)  # nada de refazer do zero
        self.assertEqual(minutes.source, "llm")
        self.assertTrue(minutes.decisions and minutes.action_items)
        self.assertTrue(any("aproveitado" in w for w in minutes.warnings))

    async def test_loop_retries_with_anti_repetition_and_same_window(self):
        loop = LLMResponse(content='{"objectives": [' + " " * 200, done_reason="loop", loop_reason="espaços")
        llm = FakeLLM([loop, minutes_payload()])
        minutes = await make_generator(llm).generate(make_request())
        self.assertEqual(minutes.source, "llm")
        first, second = llm.calls
        self.assertIsNone(first["options"])
        self.assertEqual(second["options"]["repeat_penalty"], 1.15)
        self.assertGreater(second["temperature"], first["temperature"])
        # janela e orçamento fixos: sem recarregar o modelo e sem gerar mais do mesmo laço
        self.assertEqual((first["num_ctx"], first["num_predict"]), (second["num_ctx"], second["num_predict"]))

    async def test_lists_are_trimmed_if_backend_ignores_schema_limits(self):
        payload = minutes_payload(decisions=[{"description": f"Decisão {i}", "evidence": [4]} for i in range(40)])
        minutes = await make_generator(FakeLLM([payload])).generate(make_request())
        self.assertLessEqual(len(minutes.decisions), 12)

    async def test_fallback_when_llm_unavailable(self):
        progress = []
        req = make_request(context=MeetingContext(objective="Decidir deploy"),
                           on_progress=lambda m, lvl="info": progress.append((m, lvl)))
        minutes = await make_generator(FakeLLM(unavailable=True)).generate(req)
        self.assertEqual(progress[-1][1], "warning")
        self.assertIn("modo de contingência", progress[-1][0])
        self.assertEqual(minutes.source, "heuristic")
        self.assertTrue(minutes.warnings)
        self.assertTrue(any(d.evidence == [4] for d in minutes.decisions))
        self.assertEqual(minutes.objectives[0].origin, "informado")

    async def test_fallback_after_exhausting_retries(self):
        llm = FakeLLM(["{}", "{}", "{}"])
        minutes = await make_generator(llm, max_retries=2).generate(make_request())
        self.assertEqual(minutes.source, "heuristic")


class TestMapReduce(unittest.IsolatedAsyncioTestCase):
    async def test_long_meeting_uses_map_reduce(self):
        segments = [seg(i, i * 5.0, f"Locutor {1 + i % 2}", "Discussão longa sobre o projeto " * 8)
                    for i in range(1, 201)]
        extraction = {k: v for k, v in minutes_payload().items() if k not in ("title", "executive_summary")}
        extraction["speaker_names"] = []
        n_chunks = len(chunk_segments(segments, 3000))
        llm = FakeLLM([extraction] * n_chunks + [minutes_payload(speaker_names=[])])

        gen = make_generator(llm, max_ctx=16384, chunk_tokens=3000)
        progress = []
        minutes = await gen.generate(make_request(segments=segments, on_progress=lambda m, lvl="info": progress.append((m, lvl))))

        self.assertGreater(n_chunks, 1)
        self.assertEqual(minutes.strategy, "map_reduce")
        self.assertTrue(any(m.startswith(f"Analisando bloco {n_chunks}/{n_chunks}") for m, _ in progress), progress)
        self.assertTrue(any(m.startswith("Consolidando") for m, _ in progress))
        self.assertEqual(progress[-1][1], "success")
        self.assertEqual(len(llm.calls), n_chunks + 1)
        self.assertIn("EXTRAÇÕES DOS BLOCOS", llm.calls[-1]["messages"][1]["content"])
        # Todos os blocos usam a MESMA janela (o Ollama não recarrega o modelo a cada bloco)
        block_ctx = {call["num_ctx"] for call in llm.calls[:-1]}
        self.assertEqual(len(block_ctx), 1)
        self.assertEqual(llm.calls[-1]["num_ctx"], block_ctx.pop())
        for call in llm.calls:
            self.assertLessEqual(call["num_ctx"], 16384)
            self.assertLessEqual(call["num_predict"], call["num_ctx"])
        self.assertIn("até 3 objetivos, 4 tópicos", llm.calls[0]["messages"][0]["content"])  # limites por bloco

    def _long_segments(self):
        return [seg(i, i * 5.0, f"Locutor {1 + i % 2}", "Discussão longa sobre o projeto " * 8) for i in range(1, 201)]

    async def test_failed_block_is_skipped_not_the_whole_minutes(self):
        segments = self._long_segments()
        n_chunks = len(chunk_segments(segments, 3000))
        extraction = {k: v for k, v in minutes_payload().items() if k not in ("title", "executive_summary")}
        extraction["speaker_names"] = []
        loop = LLMResponse(content="{", done_reason="loop", loop_reason="repetição")
        # bloco 1 falha nas 3 tentativas; demais blocos e consolidação funcionam
        llm = FakeLLM([loop, loop, loop] + [extraction] * (n_chunks - 1) + [minutes_payload(speaker_names=[])])
        progress = []
        minutes = await make_generator(llm, max_ctx=16384, chunk_tokens=3000).generate(
            make_request(segments=segments, on_progress=lambda m, lvl="info": progress.append((m, lvl))))
        self.assertEqual(minutes.source, "llm")
        self.assertTrue(any("bloco 1" in w for w in minutes.warnings), minutes.warnings)
        self.assertTrue(any(m.startswith(f"Consolidando {n_chunks - 1} blocos") for m, _ in progress))

    async def test_failed_consolidation_builds_minutes_from_blocks(self):
        segments = self._long_segments()
        n_chunks = len(chunk_segments(segments, 3000))
        extraction = {k: v for k, v in minutes_payload().items() if k not in ("title", "executive_summary")}
        extraction["speaker_names"] = []
        bad = LLMResponse(content="{", done_reason="loop", loop_reason="repetição")
        llm = FakeLLM([extraction] * n_chunks + [bad, bad, bad])
        minutes = await make_generator(llm, max_ctx=16384, chunk_tokens=3000).generate(make_request(segments=segments))
        self.assertEqual(minutes.source, "llm")  # não caiu no modo de contingência por palavras-chave
        self.assertEqual(minutes.title, "Reunião")
        self.assertTrue(minutes.decisions)
        self.assertEqual(len(minutes.action_items), 2)  # itens repetidos entre blocos foram unidos
        self.assertTrue(any("consolidação" in w for w in minutes.warnings))

    def test_chunks_overlap_and_respect_budget(self):
        segments = [seg(i, float(i), "Locutor 1", "palavra " * 30) for i in range(1, 41)]
        chunks = chunk_segments(segments, max_tokens=400, overlap_segments=2)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([s.id for s in chunks[1][:2]], [s.id for s in chunks[0][-2:]])
        self.assertEqual(format_segment(segments[0]), "[#1 00:01] Locutor 1: " + "palavra " * 30)


class TestDeadlineResolver(unittest.TestCase):
    REF = date(2026, 10, 2)  # sexta-feira

    def test_cases(self):
        cases = {
            "amanhã": date(2026, 10, 3),
            "depois de amanhã": date(2026, 10, 4),
            "hoje": self.REF,
            "sexta-feira": date(2026, 10, 9),
            "até segunda que vem": date(2026, 10, 5),
            "próxima terça": date(2026, 10, 6),
            "em 2 semanas": date(2026, 10, 16),
            "daqui a 3 dias": date(2026, 10, 5),
            "dia 15": date(2026, 10, 15),
            "dia 1": date(2026, 11, 1),
            "15/10": date(2026, 10, 15),
            "10/01": date(2027, 1, 10),
            "20/12/2026": date(2026, 12, 20),
            "30 de novembro": date(2026, 11, 30),
            "fim do mês": date(2026, 10, 31),
            "A definir": None,
            "quando der": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(resolve_deadline(text, self.REF), expected)


if __name__ == "__main__":
    unittest.main()
