import os
import sys
import unittest
import io
from pathlib import Path

# Adicionar pasta raiz ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.models.schemas import SpeakerSegment, MeetingMinutes, MeetingDetail, TopicItem, ActionItem
from backend.services.alignment import align_transcription_with_diarization
from backend.services.summarizer import SummarizerService
from backend.services.exporter import ExporterService
from backend import database


class TestMeetingPipeline(unittest.TestCase):
    def setUp(self):
        self.segments = [
            SpeakerSegment(id=1, start=0.0, end=4.5, speaker="Locutor 1", text="Bom dia a todos, vamos iniciar nossa reunião de alinhamento trimestral."),
            SpeakerSegment(id=2, start=5.0, end=9.8, speaker="Locutor 2", text="Perfeito Carlos. O time de engenharia já finalizou a migração da arquitetura para Docker."),
            SpeakerSegment(id=3, start=10.2, end=15.0, speaker="Locutor 1", text="Excelente notícia. Vamos definir que o deploy em produção será feito até sexta-feira."),
            SpeakerSegment(id=4, start=15.5, end=20.0, speaker="Locutor 2", text="Combinado, eu vou ficar responsável por validar os testes de carga e monitorar os servidores.")
        ]

        self.summary = MeetingMinutes(
            title="Reunião de Alinhamento Trimestral",
            date="05/09/2026 15:30",
            duration_minutes=0.33,
            participants=["Locutor 1", "Locutor 2"],
            executive_summary="Alinhamento sobre a migração de arquitetura para Docker e plano de deploy em produção.",
            main_topics=[
                TopicItem(
                    title="Migração para Docker",
                    discussion="O time de engenharia reportou conclusão bem-sucedida da migração.",
                    conclusions="Arquitetura validada e estável."
                )
            ],
            decisions=[
                "Deploy em produção agendado para sexta-feira."
            ],
            action_items=[
                ActionItem(
                    task="Validar testes de carga e monitoramento",
                    owner="Locutor 2",
                    deadline="Sexta-feira",
                    status="Pendente"
                )
            ],
            open_points=[],
            raw_markdown="# Ata de Teste"
        )

        self.meeting = MeetingDetail(
            id="test-meeting-123",
            title="Reunião de Alinhamento Trimestral",
            created_at="05/09/2026 15:30",
            audio_filename="test_meeting.wav",
            audio_duration=20.0,
            audio_url="/api/audio/test_meeting.wav",
            segments=self.segments,
            summary=self.summary,
            speaker_map={"Locutor 1": "Locutor 1", "Locutor 2": "Locutor 2"}
        )

    def test_alignment_logic(self):
        """Testa se a correspondência entre transcrição e diarização funciona corretamente."""
        transcription_raw = [
            {"start": 0.5, "end": 4.0, "text": "Primeira fala"},
            {"start": 5.2, "end": 9.5, "text": "Segunda fala"}
        ]
        diarization_raw = [
            {"start": 0.0, "end": 4.8, "speaker": "Locutor A"},
            {"start": 5.0, "end": 10.0, "speaker": "Locutor B"}
        ]

        aligned = align_transcription_with_diarization(transcription_raw, diarization_raw)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0].speaker, "Locutor A")
        self.assertEqual(aligned[1].speaker, "Locutor B")
        self.assertEqual(aligned[0].text, "Primeira fala")

    def test_markdown_and_text_export(self):
        """Testa geração de exportações em Markdown e Texto."""
        md = ExporterService.to_markdown(self.meeting)
        self.assertIn("Transcrição Completa dos Diálogos", md)
        self.assertIn("Locutor 1", md)

        txt = ExporterService.to_plain_text(self.meeting)
        self.assertIn("ATA DE REUNIÃO", txt)
        self.assertIn("RESUMO EXECUTIVO", txt)
        self.assertIn("PLANO DE AÇÃO", txt)

    def test_docx_export(self):
        """Testa geração de arquivo DOCX em memória."""
        docx_bytes = ExporterService.to_docx_bytes(self.meeting)
        self.assertIsInstance(docx_bytes, io.BytesIO)
        self.assertGreater(docx_bytes.getbuffer().nbytes, 500)

    def test_fallback_summarizer(self):
        """Testa gerador heurístico em PT-BR para situações sem LLM disponível."""
        summary = SummarizerService._generate_fallback(
            segments=self.segments,
            title="Reunião Fallback",
            participants=["Locutor 1", "Locutor 2"],
            duration_minutes=0.33
        )
        self.assertEqual(summary.title, "Reunião Fallback")
        self.assertGreater(len(summary.action_items), 0)
        self.assertIn("Ata de Reunião", summary.raw_markdown)

    def test_database_persistence(self):
        """Testa inserção, busca e exclusão no banco SQLite."""
        database.save_meeting(self.meeting)
        retrieved = database.get_meeting(self.meeting.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, self.meeting.title)
        self.assertEqual(len(retrieved.segments), 4)

        # Testar listagem
        all_meetings = database.list_meetings()
        self.assertTrue(any(m.id == self.meeting.id for m in all_meetings))

        # Testar exclusão
        database.delete_meeting(self.meeting.id)
        deleted = database.get_meeting(self.meeting.id)
        self.assertIsNone(deleted)

    def test_local_diarization(self):
        """Testa motor de diarização 100% local e offline com áudio sintético."""
        import numpy as np
        import soundfile as sf
        from backend.services.diarization import DiarizationService

        sr = 16000
        t = np.linspace(0, 4, sr * 4)
        y1 = 0.8 * np.sin(2 * np.pi * 150 * t[:sr*2])
        y2 = 0.8 * np.sin(2 * np.pi * 500 * t[sr*2:])
        y = np.concatenate([y1, y2])
        test_wav = ROOT_DIR / "data" / "test_diar_synth.wav"
        sf.write(str(test_wav), y, sr)

        try:
            diarizer = DiarizationService()
            results = diarizer.diarize(test_wav)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            self.assertIn("speaker", results[0])
            self.assertTrue(results[0]["speaker"].startswith("Locutor"))
        finally:
            if test_wav.exists():
                test_wav.unlink()

    def test_local_diarization_four_speakers(self):
        """Testa separação de 4 interlocutores distintos com timbres e frequências diferentes."""
        import numpy as np
        import soundfile as sf
        from backend.services.diarization import DiarizationService

        sr = 16000
        # Cria 4 interlocutores com frequências fundamentais distintas
        # e múltiplos turnos de fala
        def make_voice(f0, dur_sec):
            t = np.linspace(0, dur_sec, int(sr * dur_sec))
            # fundamental + harmônicos para criar timbre rico
            sig = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            silence = np.zeros(int(sr * 0.4))
            return np.concatenate([sig, silence])

        chunks = []
        # Turnos alternados entre os 4 participantes
        pitches = [120, 220, 320, 420]
        for _ in range(3):
            for p in pitches:
                chunks.append(make_voice(p, 1.2))

        y = np.concatenate(chunks)
        test_wav = ROOT_DIR / "data" / "test_diar_4spk.wav"
        sf.write(str(test_wav), y, sr)

        try:
            diarizer = DiarizationService()
            # Testar com especificação explícita de 4 participantes
            results = diarizer.diarize(test_wav, min_speakers=4, max_speakers=4)
            speakers = set(r["speaker"] for r in results)
            self.assertEqual(len(speakers), 4, f"Esperado 4 oradores, detectou: {speakers}")
        finally:
            if test_wav.exists():
                test_wav.unlink()


if __name__ == "__main__":
    unittest.main()
