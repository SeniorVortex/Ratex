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

A arena também vira par automaticamente (pares_da_arena): quem venceu é a Boa, quem perdeu
é a Ruim; e quando você escreve a resposta certa ("nenhuma prestou"), ela é a Boa contra
as duas. Empates não ensinam nada e ficam de fora.
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


def _cabe_no_formato(texto: str) -> bool:
    """A resposta pode ser escrita no .txt sem confundir o leitor? (linha começando com
    Pessoa:/Boa:... ou parágrafo começando com == quebraria o par)"""
    return bool(texto.strip()) and not any(
        linha.strip().startswith(tuple(_MARCAS)) or linha.strip().startswith("==") for linha in texto.splitlines())


def pares_da_arena(partidas: list[dict]) -> list[dict]:
    """Transforma as partidas da arena (nucleo.arena) em pares de preferência.
    Respostas que citam fonte da busca ([1], [2]...) ficam de fora: no treino não haveria
    busca nenhuma no prompt, e o modelo aprenderia a citar fonte inventada."""
    pares, vistos = [], set()
    for p in partidas:
        pergunta = (p.get("pergunta") or "").strip()
        respostas = [(p.get("resposta_a") or "").strip(), (p.get("resposta_b") or "").strip()]
        correcao = (p.get("correcao") or "").strip()
        if correcao:
            candidatos = [(correcao, r) for r in dict.fromkeys(respostas) if r and r != correcao]
        elif p.get("resultado") in ("a", "b"):
            boa, ruim = respostas if p["resultado"] == "a" else respostas[::-1]
            candidatos = [(boa, ruim)] if boa != ruim else []
        else:
            candidatos = []
        for boa, ruim in candidatos:
            chave = (pergunta, boa, ruim)
            if (not pergunta or chave in vistos or re.search(r"\[\d+\]", boa + ruim)
                    or not all(map(_cabe_no_formato, (pergunta, boa, ruim)))):
                continue
            vistos.add(chave)
            pares.append({"conversa": [{"role": "user", "content": pergunta}], "boa": boa, "ruim": ruim})
    return pares


def escrever_pares(pares: list[dict], titulo: str) -> str:
    """O texto no formato de dados_preferencia/ (dá pra abrir, revisar e editar à mão)."""
    blocos = [f"== {titulo} =="]
    for par in pares:
        linhas = [f"{'Pessoa' if m['role'] == 'user' else 'Xselo'}: {m['content']}" for m in par["conversa"]]
        blocos.append("\n".join(linhas + [f"Ruim: {par['ruim']}", f"Boa: {par['boa']}"]))
    return "\n\n".join(blocos) + "\n"


def arena_para_arquivo(arq_arena: str | Path, destino: str | Path) -> int:
    """Lê o arena.json e grava os pares em `destino` (sobrescreve: o .json é a fonte da verdade).
    Retorna quantos pares saíram."""
    import json

    partidas = json.loads(Path(arq_arena).read_text(encoding="utf-8")).get("partidas", [])
    pares = pares_da_arena(partidas)
    destino = Path(destino)
    if pares:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(escrever_pares(pares, f"Preferências vindas da arena ({len(partidas)} partidas)"),
                           encoding="utf-8")
    return len(pares)
