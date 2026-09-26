"""
Converte o dataset em texto corrido do Xselo v1 (dataset.txt + dados_extras/)
em conversas no formato de chat, usadas para treinar o LoRA do xselo-0-2.

De onde saem as conversas:
* linhas `Pessoa:` / `Xselo:`           -> conversas prontas (uma ou várias trocas)
* fichas `Nome: descrição` por seção    -> "quem é Nome?", "qual o poder de Nome?", ...
* seções de jogos, lugares, mecânica,
  músicas, memes, crônicas e analogias  -> perguntas no tom de um leigo
* qualquer outro parágrafo              -> pergunta genérica sobre o título da seção

Assim nenhum texto do v1 é jogado fora: as respostas são sempre parágrafos
escritos no estilo do Xselo, e só as perguntas são montadas automaticamente.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from .tokenizador import normalizar_texto

SYSTEM_PROMPT = (
    "Você é o Xselo, a inteligência artificial da Ratex (modelo ratex/xselo-0-2/v1), "
    "especialista em Touhou Project: Gensokyo, danmaku, os jogos e as músicas do ZUN, "
    "os incidentes e as personagens. Responda sempre em português do Brasil, com uma "
    "prosa fluida, carismática e descontraída, sem academicismo e sem jeito de robô. "
    "Explique tudo de forma simples, divertida e com analogias do dia a dia, como se "
    "estivesse falando com um amigo que nunca ouviu falar de Touhou."
)

SYSTEM_PROMPT_03 = (
    "Você é o Xselo, a inteligência artificial da Ratex (modelo ratex/xselo-0-3/v1). Sua especialidade é "
    "Touhou Project, mas você também conversa sobre qualquer assunto do dia a dia: ciência, matemática, "
    "português, história, tecnologia e cultura geral. Responda sempre em português do Brasil, com uma prosa "
    "fluida, carismática e descontraída, sem academicismo e sem jeito de robô, usando analogias simples. "
    "Em contas, mostre o passo a passo e termine com 'Resposta: ...'. Se não souber algo ou não tiver "
    "certeza, diga isso com honestidade em vez de inventar, e em assuntos de saúde, dinheiro ou lei, "
    "recomende procurar uma fonte confiável ou um profissional."
)

_TITULO = re.compile(r"^==\s*(.+?)\s*==\s*$")
_FICHA = re.compile(r"^([0-9A-ZÀ-Ú][^:\n]{0,59}):\s+(\S.*)$", re.S)
_NAO_E_NOME = ("A personalidade", "Analogia", "Meme", "Memes", "O meme", "Outro meme", "Pra ", "Resumindo", "Spoiler")

# (prefixo do título da seção, tipo)
_TIPOS_SECAO = [
    ("Conversas", "dialogo"),
    ("Dicionário de personagens", "personagem"),
    ("Os moradores", "personagem"),
    ("O pessoal", "personagem"),
    ("As fadas", "personagem"),
    ("A Montanha dos Youkai", "personagem"),
    ("O subterrâneo", "personagem"),
    ("O Templo Myouren", "personagem"),
    ("Os santos", "personagem"),
    ("Os deuses", "personagem"),
    ("Os humanos", "personagem"),
    ("Os lugares", "lugar"),
    ("Como funciona o jogo", "mecanica"),
    ("As músicas", "musica"),
    ("Memes", "meme"),
    ("Crônicas", "cronica"),
    ("Analogias", "analogia"),
    ("Touhou ", "jogo"),
]

_PERGUNTAS_GERAIS = [
    "me fala sobre {t}",
    "me explica isso aqui como se eu fosse um neandertal: {t}",
    "{t}? explica aí",
    "conta pra mim: {t}",
    "o que você sabe sobre {t}?",
]


# perguntas de leigo para as seções "soltas" (as que não são fichas de personagem/jogo/lugar)
_PERGUNTAS_SECAO = {
    "quem é o xselo": ["quem é você?", "se apresenta aí", "quem é o Xselo?"],
    "o que é touhou": ["o que é touhou?", "me explica touhou como se eu fosse um neandertal", "touhou é o quê, afinal?"],
    "por que tanta gente": ["por que tanta gente é viciada em touhou?", "o que touhou tem de tão bom?"],
    "zun": ["quem é o ZUN?", "quem criou touhou?", "me fala do criador de touhou"],
    "gensokyo": ["o que é Gensokyo?", "onde se passa touhou?"],
    "como funciona o jogo": ["como funciona o jogo?", "como se joga touhou?", "me explica o danmaku"],
    "por que os tiros": ["por que os tiros de touhou são tão bonitos?", "como funcionam os duelos de spell card?"],
    "incidentes": ["o que é um incidente em touhou?", "como são as histórias de touhou?"],
    "a era pc-98": ["quais são os jogos antigos de touhou?", "o que são os jogos de PC-98?"],
    "a série continua": ["touhou ainda lança jogo novo?", "quais são os spin-offs de touhou?"],
    "os livros e mangás": ["touhou tem mangá?", "quais são os mangás oficiais de touhou?"],
    "as músicas": ["como são as músicas de touhou?", "me fala das músicas do ZUN"],
    "a comunidade de fãs": ["como é a comunidade de touhou?", "o que os fãs de touhou fazem?"],
    "por onde começar": ["por onde eu começo em touhou?", "me dá dicas pra começar a jogar", "sou iniciante, o que eu faço?"],
    "como explicar touhou": ["como eu explico touhou pro meu amigo?", "meu amigo não entende touhou, o que eu falo pra ele?"],
}

# O dataset foi escrito para o v1 (feito do zero); nas versões híbridas o Xselo se apresenta direito.
_IDENTIDADE = [
    ("Sou um modelo pequenininho, feito do zero, e fui treinado pra explicar",
     "Sou um modelo de linguagem ajustado com LoRA em cima de um modelo base maior, e fui treinado pra explicar"),
    ("Sou um modelo pequenininho, treinado do zero, pra explicar",
     "Fui ajustado com LoRA em cima de um modelo base maior pra explicar"),
    ("Eu sou um modelo minúsculo, feito do zero, então às vezes eu falo besteira.",
     "Eu ainda sou uma IA pequena, então às vezes eu falo besteira."),
]


# no 0.3 ele também deixa de dizer que só sabe de Touhou
_IDENTIDADE_GERAL = [
    ("Eu sou pequeno e ainda estou aprendendo, então fora de Touhou eu sou meio perdido. "
     "Mas se o assunto for garota mágica desviando de tiro, pode perguntar.",
     "E fora de Touhou eu também encaro o básico de ciência, matemática, português e tecnologia. "
     "Pode perguntar."),
    ("Enquanto os modelos grandões sabem de tudo um pouco, eu sei de uma coisa só, e essa coisa é Touhou.",
     "Eu sei o básico de muita coisa, mas Touhou é onde eu brilho."),
    ("Pra Touhou, eu me viro. Pra resto, eu sou meio Cirno.",
     "Pra Touhou, eu me viro bem. Pro resto, eu sei o básico."),
    ("Mas estou aprendendo e vou melhorar nas próximas versões.",
     "Por isso, em coisa importante, vale conferir."),
]


def adaptar_identidade(texto: str, nome_modelo: str = "ratex/xselo-0-2/v1", geral: bool = False) -> str:
    texto = texto.replace("ratex/xselo-0-1/v1", nome_modelo)
    for velho, novo in _IDENTIDADE + (_IDENTIDADE_GERAL if geral else []):
        texto = texto.replace(velho, novo)
    return texto


def _pergunta_da_secao(titulo: str, rnd: random.Random) -> str | None:
    t = titulo.lower()
    for prefixo, perguntas in _PERGUNTAS_SECAO.items():
        if t.startswith(prefixo):
            return rnd.choice(perguntas)
    return None


def _tipo_secao(titulo: str) -> str:
    for prefixo, tipo in _TIPOS_SECAO:
        if titulo.startswith(prefixo):
            if tipo == "jogo" and not re.match(r"Touhou \d", titulo):
                continue
            return tipo
    return "geral"


def _minusculo(t: str) -> str:
    """Primeira letra minúscula, a não ser que a primeira palavra seja uma sigla (ZUN, PC-98)."""
    primeira = t.split(" ")[0] if t else ""
    if not t or (len(primeira) > 1 and primeira.isupper()):
        return t
    return t[:1].lower() + t[1:]


def _ficha(paragrafo: str):
    m = _FICHA.match(paragrafo)
    if not m or m.group(1).startswith(_NAO_E_NOME):
        return None
    return m.group(1).strip(), m.group(2).strip()


def _frase_de_ficha(nome: str, resto: str) -> str:
    """'Reimu Hakurei' + 'a protagonista...' -> 'A Reimu Hakurei é a protagonista...'"""
    verbo = "são" if re.search(r",| e ", nome) else "é"
    if re.match(r"^(a|o|as|os|uma|um) ", resto):
        return f"{nome} {verbo} {resto}"
    return f"{nome}: {resto}"


def _pergunta_seguinte(nome: str, paragrafo: str) -> str:
    p = paragrafo.lower()
    if p.startswith(("o meme", "meme", "memes", "outro meme")):
        return f"quais os memes de {nome}?"
    if p.startswith("analogia"):
        return f"explica {nome} com uma analogia pra leigo"
    if p.startswith(("o poder", "os poderes")):
        return f"qual o poder de {nome}?"
    if p.startswith(("a personalidade", "o jeito")):
        return f"como é a personalidade de {nome}?"
    if p.startswith("o incidente"):
        return f"qual foi o incidente do {nome}?"
    return f"me conta mais sobre {nome}"


def _conversa(pergunta: str, resposta: str) -> list[dict]:
    return [{"role": "user", "content": pergunta}, {"role": "assistant", "content": resposta}]


def _dialogos(bloco: str) -> list[list[dict]]:
    """Linhas 'Pessoa: ...' / 'Xselo: ...' -> uma conversa (lista de mensagens)."""
    msgs: list[dict] = []
    for linha in bloco.splitlines():
        linha = linha.strip()
        if linha.startswith("Pessoa:"):
            msgs.append({"role": "user", "content": linha[len("Pessoa:"):].strip()})
        elif linha.startswith("Xselo:") and msgs and msgs[-1]["role"] == "user":
            msgs.append({"role": "assistant", "content": linha[len("Xselo:"):].strip()})
    return [msgs] if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" else []


def _secoes(texto: str):
    titulo, paragrafos = "", []
    for bloco in re.split(r"\n\s*\n", texto):
        bloco = bloco.strip()
        if not bloco:
            continue
        m = _TITULO.match(bloco.splitlines()[0])
        if m:
            if paragrafos:
                yield titulo, paragrafos
            titulo, paragrafos = m.group(1), []
            resto = "\n".join(bloco.splitlines()[1:]).strip()
            if resto:
                paragrafos.append(resto)
        else:
            paragrafos.append(bloco)
    if paragrafos:
        yield titulo, paragrafos


def construir_conversas(texto: str, seed: int = 0, **kw) -> list[list[dict]]:
    """Transforma o texto do dataset em uma lista de conversas (sem o system prompt)."""
    return [c for c, _ in construir_conversas_com_origem(texto, seed, **kw)]


def construir_conversas_com_origem(texto: str, seed: int = 0, nome_modelo: str = "ratex/xselo-0-2/v1",
                                   geral: bool = False) -> list[tuple[list[dict], str]]:
    """Como construir_conversas, mas diz de onde veio cada conversa: 'dialogo' (escrita à
    mão no formato Pessoa/Xselo) ou 'texto' (pergunta montada a partir de um parágrafo)."""
    return _construir(texto, seed, nome_modelo, geral)


def conversas_de_matematica(n: int, seed: int = 0) -> list[tuple[list[dict], str]]:
    """Exercícios gerados com resolução passo a passo e resposta exata."""
    from .matematica import gerar_exercicios

    return [(_conversa(p, r), "matematica") for p, r in gerar_exercicios(n, seed=seed)]


def _construir(texto: str, seed: int, nome_modelo: str, geral: bool) -> list[tuple[list[dict], str]]:
    rnd = random.Random(seed)
    conversas: list[tuple[list[dict], str]] = []
    for titulo, paragrafos in _secoes(normalizar_texto(texto)):
        tipo = _tipo_secao(titulo)
        t = _minusculo(titulo)
        atual = None  # nome da ficha em andamento (personagem, lugar, jogo...)
        if tipo == "jogo":
            atual = re.sub(r"^Touhou [\d.]+:\s*", "", titulo)
        for i, p in enumerate(paragrafos):
            if "Pessoa:" in p and "Xselo:" in p:
                conversas += [(c, "dialogo") for c in _dialogos(p)]
                continue
            ficha = _ficha(p) if tipo in ("personagem", "lugar", "mecanica", "musica", "meme") else None
            if ficha:
                nome, resto = ficha
                atual = nome
                if tipo == "personagem":
                    quem = "quem são" if re.search(r",| e ", nome) else "quem é"
                    conversas.append((_conversa(f"{quem} {nome}?", _frase_de_ficha(nome, resto)), "texto"))
                elif tipo == "lugar":
                    conversas.append((_conversa(f"o que é {nome} em Gensokyo?", _frase_de_ficha(nome, resto)), "texto"))
                elif tipo == "mecanica":
                    conversas.append((_conversa(f"o que é {nome.lower()} no Touhou?", _frase_de_ficha(nome, resto)), "texto"))
                elif tipo == "musica":
                    conversas.append((_conversa(f"que música é {nome}?", f"{nome} é {resto}"), "texto"))
                else:
                    conversas.append((_conversa(f"qual é o meme {_minusculo(nome)}?", resto[:1].upper() + resto[1:]), "texto"))
            elif tipo == "cronica":
                m = re.match(r"^([^.]{3,60})\.\s+(.+)$", p, re.S)
                nome = m.group(1) if m else t
                conversas.append((_conversa(f"me conta uma história de Gensokyo: {_minusculo(nome)}", p), "texto"))
            elif tipo == "analogia":
                antes = p.split(" é tipo")[0] if " é tipo" in p else ""
                if antes and len(antes) < 40:
                    pergunta = f"me explica {_minusculo(antes)} com uma analogia"
                else:
                    pergunta = "me dá uma analogia pra explicar Touhou pra um leigo"
                conversas.append((_conversa(pergunta, p), "texto"))
            elif atual and tipo in ("personagem", "jogo"):
                if tipo == "jogo" and i == 0:
                    pergunta = f"o que é o {atual}?"
                else:
                    pergunta = _pergunta_seguinte(atual, p)
                resposta = re.sub(r"^Analogia pra leigo:\s*", "", p)
                conversas.append((_conversa(pergunta, resposta[:1].upper() + resposta[1:]), "texto"))
            else:
                pergunta = _pergunta_da_secao(titulo, rnd) or rnd.choice(_PERGUNTAS_GERAIS).format(t=t or "Touhou")
                conversas.append((_conversa(pergunta, p), "texto"))
    for conversa, _ in conversas:
        for msg in conversa:
            if msg["role"] == "assistant":
                msg["content"] = adaptar_identidade(msg["content"], nome_modelo, geral)
    return conversas


def ler_textos(fontes: list[str], raiz: Path) -> tuple[str, list[Path]]:
    arquivos: list[Path] = []
    for fonte in fontes:
        caminho = Path(fonte)
        if not caminho.is_absolute():
            caminho = raiz / caminho
        if caminho.is_dir():
            arquivos += sorted(caminho.rglob("*.txt"))
        elif caminho.is_file():
            arquivos.append(caminho)
    textos = [normalizar_texto(a.read_text(encoding="utf-8")).strip() for a in arquivos]
    return "\n\n".join(t for t in textos if t) + "\n", arquivos
