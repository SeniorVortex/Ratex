"""
Tokenizadores do Xselo.

Dois sabores, ambos em Python puro e salvos num único `vocab.json`:

* ``char``: cada caractere vira um token. Simples e robusto para datasets
  pequenos (é o padrão).
* ``bpe``: Byte-Pair Encoding leve em cima dos caracteres. Junta pedaços de
  palavras frequentes ("Rei", "mu", " Gensokyo"...) em tokens únicos, então
  a janela de contexto enxerga mais texto. Vale a pena quando o dataset
  crescer (algumas centenas de KB para cima).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# Troca caracteres "chatos" que aparecem quando se cola texto da internet por
# equivalentes simples. Mantém o vocabulário pequeno e consistente entre o
# treino e a geração.
_SUBSTITUICOES = {
    "\r\n": "\n",
    "\r": "\n",
    "\t": "    ",
    " ": " ",  # espaço não separável
    "​": "",  # espaço de largura zero
    "﻿": "",  # BOM
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "´": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "⑨": "9",  # o lendário (9) da Cirno
}


def normalizar_texto(texto: str) -> str:
    """Normaliza unicode (NFC) e troca pontuação exótica por ASCII simples."""
    texto = unicodedata.normalize("NFC", texto)
    for velho, novo in _SUBSTITUICOES.items():
        texto = texto.replace(velho, novo)
    return texto


class TokenizadorChar:
    tipo = "char"

    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def treinar(cls, texto: str, **_) -> "TokenizadorChar":
        return cls(sorted(set(texto)))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, texto: str) -> list[int]:
        texto = normalizar_texto(texto)
        return [self.stoi[c] for c in texto if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.tokens[i] for i in ids)

    def desconhecidos(self, texto: str) -> set[str]:
        return {c for c in normalizar_texto(texto) if c not in self.stoi}

    def para_dict(self) -> dict:
        return {"tipo": self.tipo, "tokens": self.tokens}


# Pré-tokenização estilo GPT-2: palavras (com o espaço da frente), números,
# pontuação e espaços em branco. O BPE nunca junta pedaços de grupos diferentes.
_PADRAO_BPE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


class TokenizadorBPE:
    tipo = "bpe"

    def __init__(self, tokens: list[str], merges: list[tuple[str, str]]):
        self.tokens = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def treinar(cls, texto: str, vocab_size: int = 512, verbose: bool = True) -> "TokenizadorBPE":
        base = sorted(set(texto))
        palavras = Counter(_PADRAO_BPE.findall(texto))
        # cada palavra vira uma lista de símbolos que vão sendo fundidos
        seqs = {p: list(p) for p in palavras}
        tokens = list(base)
        conhecidos = set(tokens)
        merges: list[tuple[str, str]] = []
        while len(tokens) < vocab_size:
            pares: Counter = Counter()
            for p, freq in palavras.items():
                s = seqs[p]
                for a, b in zip(s, s[1:]):
                    pares[(a, b)] += freq
            if not pares:
                break
            (a, b), freq = pares.most_common(1)[0]
            if freq < 2:
                break
            novo = a + b
            merges.append((a, b))
            if novo not in conhecidos:  # dois merges diferentes podem dar a mesma string
                tokens.append(novo)
                conhecidos.add(novo)
            for p, s in seqs.items():
                if len(s) < 2:
                    continue
                i, saida = 0, []
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        saida.append(novo)
                        i += 2
                    else:
                        saida.append(s[i])
                        i += 1
                seqs[p] = saida
            if verbose and len(merges) % 100 == 0:
                print(f"  bpe: {len(tokens)}/{vocab_size} tokens (último: {novo!r})")
        return cls(tokens, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _encode_palavra(self, palavra: str) -> list[int]:
        if palavra in self._cache:
            return self._cache[palavra]
        s = [c for c in palavra if c in self.stoi]
        while len(s) > 1:
            melhor = min(
                ((self.ranks.get((a, b), float("inf")), i) for i, (a, b) in enumerate(zip(s, s[1:]))),
                default=(float("inf"), -1),
            )
            if melhor[0] == float("inf"):
                break
            i = melhor[1]
            s = s[:i] + [s[i] + s[i + 1]] + s[i + 2 :]
        ids = [self.stoi[t] for t in s]
        self._cache[palavra] = ids
        return ids

    def encode(self, texto: str) -> list[int]:
        texto = normalizar_texto(texto)
        ids: list[int] = []
        for palavra in _PADRAO_BPE.findall(texto):
            ids.extend(self._encode_palavra(palavra))
        return ids

    def decode(self, ids) -> str:
        return "".join(self.tokens[i] for i in ids)

    def desconhecidos(self, texto: str) -> set[str]:
        base = {t for t in self.tokens if len(t) == 1}
        return {c for c in normalizar_texto(texto) if c not in base}

    def para_dict(self) -> dict:
        return {"tipo": self.tipo, "tokens": self.tokens, "merges": [list(m) for m in self.merges]}


TIPOS = {"char": TokenizadorChar, "bpe": TokenizadorBPE}


def treinar_tokenizador(tipo: str, texto: str, vocab_size: int = 512):
    if tipo not in TIPOS:
        raise ValueError(f"tokenizador desconhecido: {tipo!r} (use um de {list(TIPOS)})")
    return TIPOS[tipo].treinar(texto, vocab_size=vocab_size)


def salvar_tokenizador(tok, caminho: Path) -> None:
    dados = tok.para_dict()
    dados["vocab_size"] = tok.vocab_size
    Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")


def carregar_tokenizador(caminho: Path):
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    tipo = dados.get("tipo", "char")
    if tipo == "char":
        return TokenizadorChar(dados["tokens"])
    if tipo == "bpe":
        return TokenizadorBPE(dados["tokens"], dados["merges"])
    raise ValueError(f"vocab.json com tipo desconhecido: {tipo!r}")
