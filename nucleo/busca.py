"""
Busca na internet pro Xselo, usando o SearXNG (um buscador aberto que junta resultados
do Google, Bing, Wikipédia, DuckDuckGo... sem rastrear ninguém).

O Xselo busca quando a pergunta tem cara de coisa atual ("agora", "hoje", "notícia",
"preço", um ano recente...) ou quando você começa a mensagem com /buscar. Os melhores
resultados entram no prompt com o link, e ele responde citando [1], [2]...

    from nucleo.busca import Busca, iniciar_searxng
    url = iniciar_searxng()          # baixa, configura e liga um SearXNG local (1ª vez: ~1-2 min)
    busca = Busca(url)
    busca.buscar("capital da austrália")

Também dá pra apontar pra um SearXNG que já esteja rodando (ex.: no Docker do seu PC),
desde que ele aceite o formato JSON (search.formats: [html, json] no settings.yml).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

URL_PADRAO = os.environ.get("XSELO_BUSCA", "http://127.0.0.1:8888")
PASTA_SEARXNG = Path.home() / ".cache" / "ratex" / "searxng"

# palavras que indicam que a resposta depende de informação atual
_ATUAL = [
    "agora", "hoje", "ontem", "amanha", "atual", "atualmente", "ultimo", "ultima", "ultimos", "ultimas",
    "recente", "recentemente", "noticia", "noticias", "lancamento", "lancou", "vai lancar", "preco",
    "cotacao", "quanto custa", "quanto ta", "dolar", "bitcoin", "placar", "resultado do jogo",
    "quem ganhou", "quem venceu", "previsao do tempo", "clima em", "temperatura em", "esta semana",
    "este mes", "este ano", "essa semana", "esse mes", "esse ano", "no momento", "tendencia", "em alta",
]


def _sem_acento(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def precisa_buscar(pergunta: str) -> bool:
    """Heurística simples: a pergunta depende de informação recente?"""
    p = _sem_acento(pergunta)
    if p.lstrip().startswith("/buscar"):
        return True
    ano = date.today().year
    if re.search(rf"\b({ano - 1}|{ano}|{ano + 1})\b", p):
        return True
    return any(re.search(rf"\b{re.escape(palavra)}\b", p) for palavra in _ATUAL)


class Busca:
    def __init__(self, url: str = URL_PADRAO, idioma: str = "pt-BR", tempo_limite: float = 12.0):
        self.url = url.rstrip("/")
        self.idioma = idioma
        self.tempo_limite = tempo_limite

    def funcionando(self) -> bool:
        try:
            return bool(self.buscar("teste", n=1))
        except Exception:
            return False

    def buscar(self, consulta: str, n: int = 4) -> list[dict]:
        """[{"titulo", "url", "trecho"}] dos n melhores resultados (sem repetir site)."""
        consulta = re.sub(r"^\s*/buscar\s*", "", consulta).strip()
        parametros = urllib.parse.urlencode({"q": consulta, "format": "json", "language": self.idioma})
        pedido = urllib.request.Request(f"{self.url}/search?{parametros}",
                                        headers={"User-Agent": "Xselo/0.4", "X-Real-IP": "127.0.0.1"})
        with urllib.request.urlopen(pedido, timeout=self.tempo_limite) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
        saida, sites = [], set()
        for r in dados.get("results", []):
            site = urllib.parse.urlparse(r.get("url", "")).netloc
            trecho = re.sub(r"\s+", " ", r.get("content") or "").strip()
            if not trecho or site in sites:
                continue
            sites.add(site)
            saida.append({"titulo": (r.get("title") or "").strip(), "url": r.get("url", ""), "trecho": trecho[:500]})
            if len(saida) == n:
                break
        return saida


def prompt_com_busca(system_prompt: str, resultados: list[dict]) -> str:
    if not resultados:
        return system_prompt
    fontes = "\n\n".join(f"[{i}] {r['titulo']} ({r['url']})\n{r['trecho']}" for i, r in enumerate(resultados, 1))
    return (f"{system_prompt}\n\nResultados de uma busca na internet feita agora ({date.today():%d/%m/%Y}). "
            "Use o que for relevante, cite a fonte no formato [1], [2]..., e se os resultados não "
            f"responderem a pergunta, diga isso com honestidade:\n\n{fontes}")


def iniciar_searxng(pasta: Path = PASTA_SEARXNG, porta: int = 8888, esperar: float = 90.0) -> str:
    """Baixa, configura e liga um SearXNG só pra este computador. Retorna a URL.
    Precisa de git e internet. Da 2ª vez em diante é rápido (reaproveita a instalação)."""
    url = f"http://127.0.0.1:{porta}"
    if Busca(url).funcionando():
        return url
    pasta = Path(pasta)
    codigo, venv = pasta / "searxng", pasta / "venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not codigo.exists():
        pasta.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/searxng/searxng.git", str(codigo)],
                       check=True)
    if not python.exists():
        # ambiente separado: as versões fixas do SearXNG não bagunçam as bibliotecas do Xselo
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", "-U", "pip", "setuptools", "wheel"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", "-r", str(codigo / "requirements.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(codigo)],
                       check=True)
    config = pasta / "settings.yml"
    config.write_text(
        "use_default_settings: true\n"
        "general:\n  instance_name: \"Xselo busca\"\n"
        "search:\n  safe_search: 1\n  default_lang: \"pt-BR\"\n  formats: [html, json]\n"
        f"server:\n  bind_address: \"127.0.0.1\"\n  port: {porta}\n  secret_key: \"{secrets.token_hex(16)}\"\n"
        "  limiter: false\n  image_proxy: false\n", encoding="utf-8")
    log = open(pasta / "searxng.log", "a", encoding="utf-8")
    subprocess.Popen([str(python), "-m", "searx.webapp"], cwd=codigo, stdout=log, stderr=log,
                     env={**os.environ, "SEARXNG_SETTINGS_PATH": str(config)})
    inicio = time.time()
    while time.time() - inicio < esperar:
        time.sleep(3)
        if Busca(url).funcionando():
            return url
    raise RuntimeError(f"o SearXNG não respondeu em {esperar:.0f}s; veja {pasta / 'searxng.log'}")
