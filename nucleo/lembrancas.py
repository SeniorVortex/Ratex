"""
Memória de conversa longa: o Xselo lembra de você entre uma conversa e outra.

São duas coisas guardadas por pessoa (num .json):

    fatos    frases curtas sobre a pessoa ("se chama Ana", "gosta de Touhou", "mora em Recife").
             Entram sozinhas quando a pessoa conta ("meu nome é...", "eu gosto de...", "moro em..."),
             ou na mão, com /lembrar <fato>. /esquecer apaga tudo.
    resumo   quando a conversa fica comprida, as mensagens mais antigas saem do prompt, mas antes
             o próprio Xselo escreve um resumo delas. Assim ele não esquece o começo do papo.

Tudo isso entra no system prompt como "o que você lembra desta pessoa".

    lembrancas = Lembrancas("~/.cache/ratex/lembrancas.json")
    conversa = Conversa("ana", lembrancas, resumir=resumidor(modelo, tok))
    conversa.usuario("oi, meu nome é Ana e eu moro em Recife")
    resposta = responder(modelo, tok, conversa.historico, system_prompt=conversa.system(cfg["system_prompt"]), ...)
    conversa.xselo(resposta)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ARQ_PADRAO = Path.home() / ".cache" / "ratex" / "lembrancas.json"
MAX_FATOS = 30

# (padrão, como vira fato). O grupo "x" é o que a pessoa contou; cortado no fim da frase.
_PADROES = [
    (r"\b(?:meu nome é|eu me chamo|me chamo|pode me chamar de)\s+(?P<x>[^\s,.!?]+(?:\s+[A-ZÀ-Ú][^\s,.!?]*)?)",
     "se chama {x}"),
    (r"\beu (?:moro|vivo) (?:em|no|na|nos|nas)\s+(?P<x>[^,.!?\n]{2,40})", "mora em {x}"),
    (r"\b(?:eu )?tenho (?P<x>\d{1,3}) anos\b", "tem {x} anos"),
    (r"\beu (?:trabalho|trampo) (?:como|de)\s+(?P<x>[^,.!?\n]{2,40})", "trabalha como {x}"),
    (r"\beu (?:estudo|faço faculdade de|curso)\s+(?P<x>[^,.!?\n]{2,40})", "estuda {x}"),
    (r"\beu (?:gosto muito de|gosto de|amo|adoro)\s+(?P<x>[^,.!?\n]{2,40})", "gosta de {x}"),
    (r"\beu (?:odeio|detesto|não gosto de|nao gosto de)\s+(?P<x>[^,.!?\n]{2,40})", "não gosta de {x}"),
    (r"\bmeu (?:personagem|jogo|anime|livro|filme) (?:favorito|preferido) é\s+(?:o |a )?(?P<x>[^,.!?\n]{2,40})",
     "tem como favorito: {x}"),
    (r"\bminha (?:personagem|música|série|banda) (?:favorita|preferida) é\s+(?:o |a )?(?P<x>[^,.!?\n]{2,40})",
     "tem como favorita: {x}"),
]
# "eu gosto de você" / "eu amo isso" não dizem nada sobre a pessoa
_VAZIO = re.compile(r"^(você|vc|voce|isso|isto|aquilo|ele|ela|tu|te|muito|demais|quando|que|se)\b", re.I)


def fatos_da_mensagem(texto: str) -> list[str]:
    """Fatos sobre a pessoa que dá pra tirar da mensagem sem modelo nenhum (padrões fixos)."""
    fatos = []
    for padrao, molde in _PADROES:
        for m in re.finditer(padrao, texto, flags=re.I):
            # "gosta de Touhou e de café" fica inteiro; nos outros, o "e" já começa outro assunto
            corte = r"\s+(?:mas|porque|pq|só que)\s+" if "gosta" in molde else r"\s+(?:e|mas|porque|pq|só que)\s+"
            x = re.split(corte, m.group("x").strip(), maxsplit=1)[0].strip(" .,!?")
            if molde.startswith("se chama"):  # sobrenome só se vier com maiúscula
                palavras = x.split()
                x = " ".join(palavras[:1] + [w for w in palavras[1:2] if w[:1].isupper()])
            if x and not _VAZIO.match(x):
                fatos.append(molde.format(x=x))
    return fatos


class Lembrancas:
    """As lembranças de todo mundo, num arquivo .json: {pessoa: {"fatos": [...], "resumo": str}}."""

    def __init__(self, arquivo: str | Path | None = ARQ_PADRAO):
        self.arquivo = Path(arquivo).expanduser() if arquivo else None
        self.dados: dict[str, dict] = {}
        if self.arquivo and self.arquivo.exists():
            self.dados = json.loads(self.arquivo.read_text(encoding="utf-8"))

    def de(self, pessoa: str) -> dict:
        return self.dados.setdefault(str(pessoa), {"fatos": [], "resumo": ""})

    def lembrar(self, pessoa: str, fato: str) -> None:
        fato = fato.strip().rstrip(".")
        if not fato:
            return
        fatos = self.de(pessoa)["fatos"]
        # um fato novo do mesmo tipo substitui o antigo ("mora em X" -> "mora em Y")
        tipo = next((m.split("{x}")[0] for _, m in _PADROES if fato.startswith(m.split("{x}")[0])), None)
        if tipo and tipo not in ("gosta de ", "não gosta de "):
            fatos[:] = [f for f in fatos if not f.startswith(tipo)]
        if fato.lower() not in (f.lower() for f in fatos):
            fatos.append(fato)
        del fatos[:-MAX_FATOS]
        self.salvar()

    def guardar_resumo(self, pessoa: str, resumo: str) -> None:
        self.de(pessoa)["resumo"] = resumo.strip()
        self.de(pessoa)["atualizado"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.salvar()

    def esquecer(self, pessoa: str) -> None:
        self.dados.pop(str(pessoa), None)
        self.salvar()

    def salvar(self) -> None:
        if self.arquivo:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            self.arquivo.write_text(json.dumps(self.dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def texto(self, pessoa: str) -> str:
        d = self.dados.get(str(pessoa)) or {}
        partes = []
        if d.get("fatos"):
            partes.append("O que você sabe sobre a pessoa com quem está falando:\n" +
                          "\n".join(f"- {f}" for f in d["fatos"]))
        if d.get("resumo"):
            partes.append(f"Resumo do que vocês já conversaram antes:\n{d['resumo']}")
        return "\n\n".join(partes)


def prompt_com_lembrancas(system_prompt: str, texto: str) -> str:
    if not texto:
        return system_prompt
    return (f"{system_prompt}\n\n{texto}\n\nUse essas lembranças com naturalidade, só quando fizerem sentido; "
            "não repita a lista pra pessoa.")


PEDIDO_RESUMO = (
    "Resuma a conversa abaixo em no máximo 6 frases curtas, em português, pra você mesmo lembrar depois: "
    "quem é a pessoa, do que falaram, o que ela pediu ou prometeu voltar a falar. Só o que aparece na "
    "conversa, sem inventar nada.{anterior}\n\n{conversa}"
)


def resumidor(modelo, tok, max_novos_tokens: int = 220) -> Callable[[str, list[dict]], str]:
    """Função que usa o próprio Xselo pra resumir mensagens antigas (temperatura 0: sem invenção)."""
    from .hibrido import responder

    def resumir(resumo_anterior: str, mensagens: list[dict]) -> str:
        conversa = "\n".join(f"{'Pessoa' if m['role'] == 'user' else 'Xselo'}: {m['content']}" for m in mensagens)
        anterior = f"\n\nResumo de antes, pra juntar com o novo:\n{resumo_anterior}" if resumo_anterior else ""
        pedido = PEDIDO_RESUMO.format(anterior=anterior, conversa=conversa[-6000:])
        return responder(modelo, tok, [{"role": "user", "content": pedido}],
                         system_prompt="Você resume conversas com fidelidade e poucas palavras.",
                         stream=False, max_novos_tokens=max_novos_tokens, temperatura=0,
                         penalidade_repeticao=1.05)
    return resumir


class Conversa:
    """Uma conversa com uma pessoa: histórico recente + lembranças de longo prazo."""

    def __init__(self, pessoa: str = "voce", lembrancas: Lembrancas | None = None,
                 resumir: Callable[[str, list[dict]], str] | None = None, max_mensagens: int = 12):
        self.pessoa = str(pessoa)
        self.lembrancas = lembrancas if lembrancas is not None else Lembrancas(None)
        self.resumir = resumir
        self.max_mensagens = max_mensagens
        self.historico: list[dict] = []

    def usuario(self, texto: str, imagens: list | None = None) -> None:
        for fato in fatos_da_mensagem(texto):
            self.lembrancas.lembrar(self.pessoa, fato)
        msg = {"role": "user", "content": texto}
        if imagens:
            msg["imagens"] = list(imagens)
        self.historico.append(msg)

    def xselo(self, resposta: str) -> None:
        self.historico.append({"role": "assistant", "content": resposta})
        if len(self.historico) > self.max_mensagens:
            # sai a metade mais antiga, mas antes vira resumo
            corte = len(self.historico) - self.max_mensagens // 2
            antigas, self.historico = self.historico[:corte], self.historico[corte:]
            for m in self.historico:  # imagem antiga não volta pro prompt: pesa e já foi comentada
                m.pop("imagens", None)
            if self.resumir:
                anterior = self.lembrancas.de(self.pessoa).get("resumo", "")
                try:
                    self.lembrancas.guardar_resumo(self.pessoa, self.resumir(anterior, antigas))
                except Exception as erro:  # o resumo é um extra: se falhar, a conversa segue
                    print(f"(não consegui resumir a conversa: {erro})")

    def system(self, system_prompt: str) -> str:
        return prompt_com_lembrancas(system_prompt, self.lembrancas.texto(self.pessoa))

    def comando(self, msg: str) -> str | None:
        """Trata /lembrar, /esquecer, /lembrancas e /novo. Retorna o aviso, ou None se não era comando."""
        if msg.startswith("/lembrar "):
            self.lembrancas.lembrar(self.pessoa, msg[len("/lembrar "):])
            return "(anotado, vou lembrar disso)"
        if msg == "/esquecer":
            self.lembrancas.esquecer(self.pessoa)
            self.historico.clear()
            return "(pronto, esqueci tudo sobre você)"
        if msg in ("/lembrancas", "/lembranças"):
            return self.lembrancas.texto(self.pessoa) or "(ainda não lembro de nada sobre você)"
        if msg == "/novo":
            self.historico.clear()
            return "(conversa zerada; as lembranças continuam)"
        return None
