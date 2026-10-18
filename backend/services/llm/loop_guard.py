"""
Detecção de geração degenerada durante o streaming do LLM.

Com saída estruturada (JSON Schema), alguns modelos entram em laço: emitem quebras de linha/
espaços sem fim, repetem o mesmo trecho ou repetem o mesmo ITEM com pequenas variações
(números de evidência, IDs) até esgotar `num_predict`. Esperar o limite custa minutos e a
resposta é descartada. O guard observa o texto enquanto chega e manda abortar assim que o
padrão aparece — a nova tentativa começa em segundos, não em minutos.

Três detectores:
1. espaços em branco sem fim no final do texto;
2. repetição periódica do final do texto, com DÍGITOS NORMALIZADOS (pega itens quase
   idênticos que só mudam números, e listas numéricas longas como "evidence": [1, 2, 3, ...]);
3. o mesmo valor textual (description/task/title/...) repetido em vários itens.
"""
import re
import unicodedata
from collections import Counter
from typing import Optional

_TRAILING_WS = re.compile(r"\s+$")
_DIGITS = re.compile(r"\d+")
# Valores dos campos de texto dos itens da ata (strings JSON completas).
# "conclusions"/"notes" ficam de fora: "Sem conclusão" repetido em vários tópicos é legítimo.
_ITEM_TEXT = re.compile(r'"(?:description|task|title|discussion)"\s*:\s*"((?:[^"\\]|\\.){20,})"')
MIN_ITEM_CHARS = 20


def _norm_value(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z ]", "", re.sub(r"\s+", " ", _DIGITS.sub("#", text))).strip()


class LoopGuard:
    def __init__(self, max_trailing_whitespace: int = 64, window: int = 800, min_period: int = 4,
                 max_period: int = 200, min_repeated_chars: int = 240, check_every: int = 200,
                 max_same_item: int = 3):
        self.max_trailing_whitespace = max_trailing_whitespace
        self.window = window
        self.min_period = min_period
        self.max_period = max_period
        self.min_repeated_chars = min_repeated_chars
        self.check_every = check_every
        self.max_same_item = max_same_item
        self._last_checked = 0

    def check(self, text: str) -> Optional[str]:
        """Motivo do aborto ou None."""
        if len(text) - self._last_checked < self.check_every:
            return None
        self._last_checked = len(text)

        trailing = _TRAILING_WS.search(text[-(self.max_trailing_whitespace + 1):])
        if trailing and len(trailing.group(0)) > self.max_trailing_whitespace:
            return "espaços em branco sem fim"

        if self._periodic_tail(text[-self.window:]):
            return "repetição do mesmo trecho"
        if self._duplicated_items(text):
            return "itens repetidos"
        return None

    def _periodic_tail(self, tail: str) -> bool:
        # Dígitos viram "#": "evidence": [12], [13], [14]... ou "1, 2, 3, 4..." passam a ser periódicos.
        tail = _DIGITS.sub("#", tail)
        if len(tail) < self.min_repeated_chars:
            return False
        for period in range(self.min_period, min(self.max_period, len(tail) // 3) + 1):
            unit = tail[-period:]
            if not unit.strip() or not any(c.isalnum() or c == "#" for c in unit):
                continue
            repeats = max(3, self.min_repeated_chars // period + 1)
            if len(tail) >= period * repeats and tail.endswith(unit * repeats):
                return True
        return False

    def _duplicated_items(self, text: str) -> bool:
        values = [_norm_value(m.group(1)) for m in _ITEM_TEXT.finditer(text)]
        counts = Counter(v for v in values if len(v) >= MIN_ITEM_CHARS)
        return bool(counts) and counts.most_common(1)[0][1] >= self.max_same_item
