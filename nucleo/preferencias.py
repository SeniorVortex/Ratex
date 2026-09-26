"""
Pares de preferência para o DPO: "nessa conversa, esta resposta é melhor que aquela".

Formato (arquivos .txt em dados_preferencia/):

    Pessoa: quem é o ser mais poderoso de touhou?
    Ruim: O ser mais poderoso é a Luna Child, a deusa da lua...
    Boa: Não existe um ranking oficial, mas a candidata mais citada é a Hecatia...

Antes do par pode vir o resto da conversa (Pessoa:/Xselo: alternando), pra ensinar
comportamento no meio do papo, tipo aceitar uma correção:

    Pessoa: quem é o ser mais poderoso de touhou?
    Xselo: É a Luna Child.
    Pessoa: não, a luna child só apaga o som, é uma fada fraquinha
    Ruim: A Luna Child é quase invencível, mas...
    Boa: Você tem razão, eu errei feio...

Como no resto do dataset, respostas podem ter vários parágrafos, e uma linha em branco
seguida de "Pessoa:" começa um par novo.
"""

from __future__ import annotations

import re
from pathlib import Path

from .dados_chat import ler_textos

_MARCAS = {"Pessoa:": "user", "Xselo:": "assistant", "Ruim:": "ruim", "Boa:": "boa"}


def _blocos(texto: str) -> list[str]:
    """Separa por linha em branco, mas junta de volta os parágrafos que continuam uma fala."""
    blocos: list[str] = []
    for bloco in re.split(r"\n\s*\n", texto):
        bloco = bloco.strip()
        if not bloco or bloco.startswith("=="):
            continue
        if blocos and not bloco.startswith(tuple(_MARCAS)):
            blocos[-1] += "\n\n" + bloco
        else:
            blocos.append(bloco)
    return blocos


def _falas(bloco: str) -> list[tuple[str, str]]:
    falas: list[list[str]] = []
    for linha in bloco.splitlines():
        limpa = linha.strip()
        marca = next((m for m in _MARCAS if limpa.startswith(m)), None)
        if marca:
            falas.append([_MARCAS[marca], limpa[len(marca):].strip()])
        elif falas:
            falas[-1][1] += "\n" + limpa
    return [(papel, re.sub(r"\n{3,}", "\n\n", conteudo).strip()) for papel, conteudo in falas]


def pares_do_texto(texto: str) -> list[dict]:
    """[{"conversa": [mensagens até a última fala da pessoa], "boa": str, "ruim": str}]"""
    pares = []
    for bloco in _blocos(texto):
        falas = _falas(bloco)
        conversa = [{"role": p, "content": c} for p, c in falas if p in ("user", "assistant")]
        boa = next((c for p, c in falas if p == "boa"), None)
        ruim = next((c for p, c in falas if p == "ruim"), None)
        if boa and ruim and conversa and conversa[-1]["role"] == "user":
            pares.append({"conversa": conversa, "boa": boa, "ruim": ruim})
    return pares


def ler_pares(fontes: list[str], raiz: Path) -> tuple[list[dict], list[Path]]:
    texto, arquivos = ler_textos(fontes, raiz)
    return pares_do_texto(texto), arquivos
