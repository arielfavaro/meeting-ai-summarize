"""Fuso horário da aplicação (container em UTC, usuário em America/Sao_Paulo)."""
import _env  # noqa: F401

import unittest

from backend import clock
from backend.services.jobs import JobRegistry
from backend.services.minutes.renderer import format_datetime


class TestClock(unittest.TestCase):
    def test_now_is_in_app_timezone(self):
        self.assertEqual(clock.now().utcoffset().total_seconds(), -3 * 3600)

    def test_utc_dates_are_shown_in_app_timezone(self):
        # reuniões gravadas antes da correção têm created_at em UTC
        self.assertEqual(format_datetime("2026-10-04T17:55:00+00:00"), "04/10/2026 14:55")
        self.assertEqual(format_datetime("05/09/2026 15:30"), "05/09/2026 15:30")  # legado, sem fuso

    def test_job_log_has_epoch_for_the_browser(self):
        jobs = JobRegistry()
        job = jobs.create()
        jobs.log(job.job_id, "teste")
        entry = jobs.get(job.job_id).logs[-1]
        self.assertGreater(entry.ts, 1_700_000_000)
        self.assertRegex(entry.time, r"^\d{2}:\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main()
