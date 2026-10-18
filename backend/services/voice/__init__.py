"""
Assinaturas de voz (embeddings de locutor) 100% locais.

- `embedder`: extrai um vetor por locutor a partir dos trechos em que ele fala (SpeechBrain ECAPA).
- `clustering`: funde locutores que a diarização separou indevidamente (dentro de uma faixa e
  entre faixas de uma gravação multi-faixa).
- `library`: biblioteca de vozes conhecidas — nomeia automaticamente quem já foi identificado
  pelo usuário em reuniões anteriores.
- `service`: fachada usada pelo pipeline e pela API.
"""
