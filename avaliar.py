#!/usr/bin/env python3
"""
Prova fixa do Xselo: as mesmas perguntas para todas as versões, com nota por categoria.

    python avaliar.py                      # avalia o modelo mais novo (igual ao gerar.py)
    python avaliar.py --modelo 0.1         # o micro-Transformer feito do zero
    python avaliar.py --modelo 0.3 --salvar avaliacoes/0.3.json
    python avaliar.py --mostrar            # imprime cada resposta
    python avaliar.py --modelo base:qwen-0.5b   # o modelo base puro, sem o LoRA (comparação)

Categorias:
    touhou      fatos de Touhou (palavras-chave esperadas na resposta)
    geral       conhecimento geral básico (perguntas diferentes das do treino)
    matematica  contas geradas com uma semente diferente da do treino; acerto = número
                certo na resposta

É uma prova simples, por palavra-chave: serve para comparar versões entre si, não
para medir "inteligência" de forma absoluta.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch

PROVA = {
    "touhou": [
        ("quem criou touhou?", [["zun"]]),
        ("quem é a Reimu Hakurei?", [["sacerdotisa", "miko"], ["santuario"]]),
        ("qual o poder da Sakuya Izayoi?", [["tempo"]]),
        ("quem é a irmã mais nova da Remilia?", [["flandre"]]),
        ("qual número é o símbolo da Cirno?", [["9", "nove"]]),
        ("o que a Marisa faz com os livros da Patchouli?", [["emprestad", "rouba", "pega"]]),
        ("quem causou o incidente da névoa vermelha?", [["remilia"]]),
        ("quem trabalha de jardineira pra Yuyuko?", [["youmu"]]),
        ("a Nazrin é que tipo de youkai?", [["rata", "rato"]]),
        ("qual o nome do Touhou 6?", [["embodiment", "scarlet devil"]]),
        ("o que quer dizer danmaku?", [["bala", "tiro"]]),
        ("qual o poder da Yukari Yakumo?", [["fronteira"]]),
    ],
    "geral": [
        ("qual é a capital da França?", [["paris"]]),
        ("quantos planetas existem no sistema solar?", [["8", "oito"]]),
        ("que gás as plantas liberam na fotossíntese?", [["oxigenio"]]),
        ("em que ano o Brasil ficou independente de Portugal?", [["1822"]]),
        ("qual o maior planeta do sistema solar?", [["jupiter"]]),
        ("o que significa a palavra saudade?", [["falta"]]),
        ("antibiótico funciona contra vírus?", [["nao"]]),
        ("qual o maior oceano do mundo?", [["pacifico"]]),
        ("quem escreveu Dom Casmurro?", [["machado"]]),
        ("qual a fórmula química da água?", [["h2o"]]),
        ("quantos lados tem um hexágono?", [["6", "seis"]]),
        ("o que é um verbo?", [["acao"]]),
    ],
}
N_MATEMATICA = 12
SEMENTE_PROVA = 4242


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.replace("₂", "2")


def questoes_matematica(seed_treino: int = 1337, n_treino: int | None = None) -> list[tuple[str, str]]:
    """Contas que NÃO apareceram no treino (mesmo gerador, outra semente)."""
    from nucleo.hibrido import VERSOES
    from nucleo.matematica import gerar_exercicios

    n_treino = n_treino or max(v["matematica"] for v in VERSOES.values())
    vistas = {p for p, _ in gerar_exercicios(n_treino, seed=seed_treino)}
    saida = []
    for p, r in gerar_exercicios(200, seed=SEMENTE_PROVA):
        if p not in vistas:
            saida.append((p, r.split("Resposta:")[1].strip().rstrip(".")))
        if len(saida) == N_MATEMATICA:
            break
    return saida


def nota_palavras(resposta: str, grupos: list[list[str]]) -> float:
    r = normalizar(resposta)
    return sum(any(re.search(rf"(?<![0-9a-z]){re.escape(normalizar(k))}", r) for k in g) for g in grupos) / len(grupos)


def nota_conta(resposta: str, esperado: str) -> float:
    """Acerto se o número principal da resposta esperada aparece na resposta do modelo."""
    numero = re.search(r"-?[\d.]+(?:,\d+)?(?:/\d+)?", esperado).group(0)
    variantes = {numero, numero.replace(".", "")}  # 2.500 ou 2500
    if re.search(r",\d0$", numero):
        variantes.add(numero[:-1])  # R$ 10,50 ou 10,5
    r = resposta.replace(" ", "")
    return float(any(re.search(rf"(?<![\d,./]){re.escape(v)}(?![\d]|[.,]\d)", r) for v in variantes))


class ModeloHibrido:
    memoria = None

    def __init__(self, pasta, device):
        from nucleo.hibrido import carregar_hibrido

        self.modelo, self.tok, self.cfg = carregar_hibrido(pasta, device=device)
        self.nome = self.cfg["nome"]

    def responder(self, pergunta: str) -> str:
        from nucleo.hibrido import responder

        return responder(self.modelo, self.tok, [{"role": "user", "content": pergunta}],
                         system_prompt=self.cfg.get("system_prompt"), stream=False, temperatura=0,
                         max_novos_tokens=200, penalidade_repeticao=1.05, memoria=self.memoria)


class ModeloBase(ModeloHibrido):
    """O modelo base sem LoRA, com o mesmo system prompt da 0.3: mostra o que o LoRA acrescentou."""

    def __init__(self, base, device):
        from nucleo.dados_chat import SYSTEM_PROMPT_03
        from nucleo.hibrido import carregar_base, escolher_device, resolver_base

        device = escolher_device(device)
        base, _ = resolver_base(base, device)
        self.modelo, self.tok = carregar_base(base, device)
        self.modelo.eval()
        self.cfg = {"system_prompt": SYSTEM_PROMPT_03}
        self.nome = f"{base} (sem LoRA)"


class ModeloV1:
    def __init__(self, pasta):
        from gerar import Gerador
        from nucleo import carregar_modelo

        modelo, tok, cfg, _ = carregar_modelo(pasta, device="cpu")
        self.nome = cfg.get("nome", "ratex/xselo-0-1/v1")
        self.g = Gerador(modelo, tok, {"temperatura": 0, "top_k": None, "top_p": None,
                                       "penalidade_repeticao": 1.0, "janela_repeticao": 64}, "cpu")

    def responder(self, pergunta: str) -> str:
        return self.g.gerar(f"Pessoa: {pergunta}\nXselo:", 300, ["\n"], stream=False).strip()


def main() -> None:
    from nucleo import PASTA_PADRAO
    from nucleo.hibrido import VERSOES, eh_hibrido, pasta_da_versao, pasta_mais_nova

    p = argparse.ArgumentParser(description="Prova fixa do Xselo.")
    p.add_argument("--modelo", default="auto", help="auto, 0.1, 0.2, 0.3, uma pasta, ou base:<modelo> (base pura)")
    p.add_argument("--device", default="auto")
    p.add_argument("--memoria", choices=["auto", "sim", "nao"], default="auto",
                   help="consulta o dataset antes de responder (RAG). auto = sim nos híbridos, não na base pura")
    p.add_argument("--salvar", help="salva as respostas e as notas neste .json")
    p.add_argument("--mostrar", action="store_true", help="imprime cada pergunta e resposta")
    args = p.parse_args()

    if args.modelo == "auto":
        pasta = pasta_mais_nova() or PASTA_PADRAO
    elif args.modelo == "0.1":
        pasta = PASTA_PADRAO
    elif args.modelo in VERSOES:
        pasta = pasta_da_versao(args.modelo)
    else:
        pasta = Path(args.modelo)
    torch.manual_seed(0)
    if args.modelo.startswith("base:"):
        modelo = ModeloBase(args.modelo[len("base:"):], args.device)
    elif eh_hibrido(pasta):
        modelo = ModeloHibrido(pasta, args.device)
    else:
        modelo = ModeloV1(pasta)
    if isinstance(modelo, ModeloHibrido):
        usar = args.memoria == "sim" or (args.memoria == "auto" and not isinstance(modelo, ModeloBase))
        if usar:
            from nucleo.memoria import Memoria

            modelo.memoria = Memoria()
            modelo.nome += " + memória"
    print(f"== prova do {modelo.nome} ({pasta}) ==", file=sys.stderr)

    itens = [(cat, q, g, "palavras") for cat, lista in PROVA.items() for q, g in lista]
    itens += [("matematica", q, r, "conta") for q, r in questoes_matematica()]
    resultados, t0 = [], time.time()
    for i, (cat, pergunta, gabarito, tipo) in enumerate(itens, 1):
        resposta = modelo.responder(pergunta)
        nota = nota_palavras(resposta, gabarito) if tipo == "palavras" else nota_conta(resposta, gabarito)
        resultados.append({"categoria": cat, "pergunta": pergunta, "gabarito": gabarito,
                           "resposta": resposta, "nota": nota})
        if args.mostrar:
            print(f"\n[{cat}] {pergunta}  (nota {nota:.2f})\n  -> {resposta[:400]}")
        else:
            print(f"\r{i}/{len(itens)} perguntas", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    notas = {}
    for cat in [*PROVA, "matematica"]:
        ns = [r["nota"] for r in resultados if r["categoria"] == cat]
        notas[cat] = round(100 * sum(ns) / len(ns), 1)
    notas["total"] = round(sum(notas.values()) / len(notas), 1)
    print(f"\nnotas do {modelo.nome} (0 a 100):")
    for cat, n in notas.items():
        print(f"  {cat:<11} {n:5.1f}  {'#' * int(n // 5)}")
    print(f"({len(itens)} perguntas em {time.time() - t0:.0f}s)")

    if args.salvar:
        Path(args.salvar).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salvar).write_text(json.dumps({"modelo": modelo.nome, "notas": notas, "respostas": resultados},
                                                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"relatório salvo em {args.salvar}")


if __name__ == "__main__":
    main()
