"""
Detecção de geração degenerada durante o streaming do LLM.

Com saída estruturada (JSON Schema), alguns modelos entram em laço: emitem quebras de linha/
espaços sem fim ou repetem o mesmo trecho até esgotar `num_predict`. Esperar o limite custa
minutos e a resposta é descartada. O guard observa o texto enquanto chega e manda abortar
assim que o padrão aparece — a nova tentativa começa em segundos, não em minutos.
"""
import re
from typing import Optional

_TRAILING_WS = re.compile(r"\s+$")


class LoopGuard:
    def __init__(self, max_trailing_whitespace: int = 64, window: int = 800, min_period: int = 4,
                 max_period: int = 200, min_repeated_chars: int = 240, check_every: int = 200):
        self.max_trailing_whitespace = max_trailing_whitespace
        self.window = window
        self.min_period = min_period
        self.max_period = max_period
        self.min_repeated_chars = min_repeated_chars
        self.check_every = check_every
        self._last_checked = 0

    def check(self, text: str) -> Optional[str]:
        """Motivo do aborto ("espaços em branco sem fim", "repetição") ou None."""
        if len(text) - self._last_checked < self.check_every:
            return None
        self._last_checked = len(text)

        trailing = _TRAILING_WS.search(text[-(self.max_trailing_whitespace + 1):])
        if trailing and len(trailing.group(0)) > self.max_trailing_whitespace:
            return "espaços em branco sem fim"

        tail = text[-self.window:]
        if len(tail) < self.min_repeated_chars:
            return None
        for period in range(self.min_period, min(self.max_period, len(tail) // 3) + 1):
            unit = tail[-period:]
            if not unit.strip():
                continue
            repeats = self.min_repeated_chars // period + 1
            if repeats < 3:
                repeats = 3
            if len(tail) >= period * repeats and tail.endswith(unit * repeats):
                return "repetição do mesmo trecho"
        return None
