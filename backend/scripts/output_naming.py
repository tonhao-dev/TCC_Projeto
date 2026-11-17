"""Utilitário de nomenclatura dos arquivos de saída (JSON/Markdown).

Os arquivos de resultado passam a carregar, como PREFIXO do nome, o timestamp
da escrita e o nome do modelo utilizado. O formato pedido para o timestamp é
``YYYY-MM-DD HH:mm:ss.SSS``; como ``:`` e espaço são inválidos em nomes de
arquivo no Windows, no NOME do arquivo esse timestamp é serializado de forma
segura para filesystem (``:`` -> ``-``, espaço -> ``_``), preservando a mesma
informação. Dentro do conteúdo (campo ``timestamp`` do JSON / cabeçalho do MD)
o formato literal com ``:`` é mantido.

Exemplo de nome gerado::

    2026-10-13_15-26-21-507_llama3.1_resultados_bertscore_seed_hibrido.json

Exemplo do campo de conteúdo::

    2026-10-13 15:26:21.507
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Formato legível (usado DENTRO do conteúdo: JSON "timestamp" e cabeçalho MD).
_FMT_CONTEUDO = "%Y-%m-%d %H:%M:%S"


def timestamp_conteudo(agora: datetime | None = None) -> str:
    """Timestamp no formato literal pedido: ``YYYY-MM-DD HH:mm:ss.SSS``.

    Usado no CONTEÚDO dos arquivos (não no nome), onde ``:`` é permitido.
    """
    agora = agora or datetime.now()
    # strftime("%f") = microssegundos (6 dígitos); fatiar para milissegundos (3).
    return agora.strftime(_FMT_CONTEUDO) + "." + agora.strftime("%f")[:3]


def _timestamp_nome(agora: datetime) -> str:
    """Mesmo instante do timestamp de conteúdo, porém válido para nome de arquivo.

    ``2026-10-13 15:26:21.507`` -> ``2026-10-13_15-26-21-507`` (sem ``:`` nem espaço).
    """
    return agora.strftime("%Y-%m-%d_%H-%M-%S-") + agora.strftime("%f")[:3]


def _sanitizar_modelo(modelo: str) -> str:
    """Normaliza o nome do modelo para uso em nome de arquivo (ex.: ``llama3.1``).

    Mantém pontos (parte da versão), troca o resto por ``-`` e remove ``:``
    (tag do Ollama, ex. ``llama3.1:latest`` -> ``llama3.1-latest``).
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", modelo.replace(":", "-")).strip("-")


def nome_com_prefixo(nome_base: str, modelo: str, agora: datetime | None = None) -> str:
    """Monta ``<timestamp>_<modelo>_<nome_base>`` com timestamp válido p/ filesystem.

    Args:
        nome_base: nome final desejado, com extensão (ex.: ``resultados.json``).
        modelo: nome do modelo utilizado (ex.: ``llama3.1``).
        agora: instante a usar; ``None`` usa ``datetime.now()``.

    Returns:
        O nome de arquivo prefixado.
    """
    agora = agora or datetime.now()
    return f"{_timestamp_nome(agora)}_{_sanitizar_modelo(modelo)}_{nome_base}"


def caminho_com_prefixo(
    diretorio: Path, nome_base: str, modelo: str, agora: datetime | None = None
) -> Path:
    """Igual a :func:`nome_com_prefixo`, mas devolve um :class:`~pathlib.Path`."""
    return Path(diretorio) / nome_com_prefixo(nome_base, modelo, agora)
