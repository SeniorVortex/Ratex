"""
Memória de consulta do Xselo (RAG): antes de responder, ele procura no próprio
dataset (dataset.txt, dados_extras/, dados_gerais/) os trechos que mais têm a ver
com a pergunta e lê esses trechos junto com o system prompt.

Um modelo pequeno não consegue decorar todos os fatos de Touhou só com o LoRA; com
a memória ele não precisa: o fato certo chega pronto no contexto. E qualquer texto
novo colado em dados_extras/ já vale na hora, sem retreinar nada.

A busca é BM25 em Python puro (sem dependências), sobre:
* cada parágrafo de texto, com o título da seção;
* cada par Pessoa/Xselo dos diálogos.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .dados_chat import _secoes, ler_textos
from .tokenizador import normalizar_texto

RAIZ = Path(__file__).resolve().parent.parent
FONTES_PADRAO = ["dataset.txt", "dados_extras", "dados_gerais"]

_STOP = set("""
a o as os um uma uns umas de do da dos das no na nos nas em por pra pro para com sem sobre
que qual quais quem quando onde como porque por que e ou mas se nao sim ja mais menos muito
muita muitos muitas eh e foi ser sao era esta estao esse essa isso este esta isto aquele aquela
ele ela eles elas eu voce voces me te se lhe nos meu minha seu sua dele dela tem ter tinha ha
ai la aqui entao tambem so ate ao aos tipo coisa coisas fala me explica sabe conta tudo
""".split())


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def termos(texto: str) -> list[str]:
    saida = []
    for p in re.findall(r"[a-z0-9]+", _norm(texto)):
        if p in _STOP or (len(p) < 2 and not p.isdigit()):
            continue
        if len(p) > 4 and p.endswith("s"):  # plural simples: personagens -> personagen
            p = p[:-1]
        saida.append(p)
    return saida


def _trechos(texto: str) -> list[str]:
    trechos = []
    for titulo, paragrafos in _secoes(normalizar_texto(texto)):
        for p in paragrafos:
            if "Pessoa:" in p and "Xselo:" in p:
                pergunta = None
                for linha in p.splitlines():
                    linha = linha.strip()
                    if linha.startswith("Pessoa:"):
                        pergunta = linha[len("Pessoa:"):].strip()
                    elif linha.startswith("Xselo:") and pergunta:
                        trechos.append(f"Pergunta: {pergunta}\nResposta: {linha[len('Xselo:'):].strip()}")
                        pergunta = None
            else:
                trechos.append(f"[{titulo}] {p}" if titulo else p)
    return trechos


class Memoria:
    def __init__(self, fontes: list[str] | None = None, k1: float = 1.5, b: float = 0.75):
        texto, self.arquivos = ler_textos(fontes or FONTES_PADRAO, RAIZ)
        self.trechos = _trechos(texto)
        self.docs = [Counter(termos(t)) for t in self.trechos]
        # "cabeça" do trecho: o nome da ficha ("Reimu Hakurei: ...") ou a pergunta do diálogo
        self.cabecas = [set(termos(self._cabeca(t))) for t in self.trechos]
        self.tam = [sum(d.values()) for d in self.docs]
        self.media = sum(self.tam) / max(len(self.tam), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.idf_max = max(self.idf.values(), default=1.0)
        self.k1, self.b = k1, b

    @staticmethod
    def _cabeca(trecho: str) -> str:
        if trecho.startswith("Pergunta:"):
            return trecho.split("\n", 1)[0]
        corpo = re.sub(r"^\[[^\]]*\]\s*", "", trecho)
        m = re.match(r"^([^:.]{2,60}):", corpo)
        return m.group(1) if m else ""

    def pontuar(self, pergunta: str) -> list[tuple[float, float, int]]:
        """[(nota BM25, cobertura, índice)], da maior nota para a menor. Cobertura = fração do
        peso (IDF) da pergunta que aparece no trecho; palavra que não existe em lugar nenhum do
        caderno (ex.: "França") pesa mais que qualquer outra, porque é justamente o assunto."""
        q = set(termos(pergunta))
        peso = {t: self.idf.get(t, 1.5 * self.idf_max) for t in q}
        total = sum(peso.values()) or 1.0
        notas = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.tam[i] / self.media))
                if t in self.cabecas[i]:
                    s += self.idf[t]  # bônus: a pergunta cita o nome da ficha / bate com a pergunta do diálogo
            if s > 0:
                notas.append((s, sum(peso[t] for t in q if t in d) / total, i))
        notas.sort(reverse=True)
        return notas

    def buscar(self, pergunta: str, k: int = 3, minimo: float = 3.0, relativo: float = 0.6,
               cobertura: float = 0.6) -> list[str]:
        """Os k trechos mais relevantes. Descarta os que pontuam abaixo de `minimo`, de
        `relativo` × a nota do melhor trecho, ou que cobrem menos de `cobertura` da pergunta
        (trecho fraco atrapalha mais do que ajuda)."""
        notas = [n for n in self.pontuar(pergunta) if n[1] >= cobertura]
        if not notas:
            return []
        corte = max(minimo, relativo * notas[0][0])
        return [self.trechos[i] for s, _, i in notas[:k] if s >= corte]


def prompt_com_memoria(system_prompt: str, trechos: list[str]) -> str:
    if not trechos:
        return system_prompt
    notas = "\n\n".join(trechos)
    return (f"{system_prompt}\n\nAnotações do seu caderno sobre o assunto (use se ajudarem a responder "
            f"certo; se não tiverem a ver com a pergunta, ignore e responda com o que você sabe):\n\n{notas}")
