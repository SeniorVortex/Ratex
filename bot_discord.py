#!/usr/bin/env python3
"""
O Xselo no Discord.

Ele responde quando alguém menciona o bot (@Xselo), responde uma mensagem dele, ou manda
mensagem direta (DM). Com --canal, responde tudo num canal específico. Lembra de cada pessoa
entre uma conversa e outra (nucleo/lembrancas.py), enxerga imagens anexadas (Xselo 0.5+) e,
com --busca, pesquisa na internet quando a pergunta é sobre coisa atual.

Comandos no chat (com ! na frente, pra não brigar com os comandos / do Discord):
    !novo        começa a conversa do zero (as lembranças continuam)
    !lembrar X   anota um fato sobre você
    !lembrancas  mostra o que ele lembra de você
    !esquecer    apaga tudo o que ele lembra de você
    !buscar X    força uma pesquisa na internet (precisa de --busca)

Como criar o bot (uma vez só, de graça, direto no site oficial do Discord):
    1. https://discord.com/developers/applications -> New Application -> dê o nome "Xselo".
    2. Aba "Bot": clique em "Reset Token" e copie o token. NUNCA mostre o token pra ninguém
       (quem tem o token controla o bot). Na mesma aba, ligue "Message Content Intent".
    3. Aba "OAuth2" -> "URL Generator": marque "bot"; em permissões, marque "Send Messages",
       "Read Message History" e "Attach Files". Abra o link gerado e adicione o bot ao seu servidor.
    4. Rode (o token vai numa variável de ambiente, nunca no código):
           DISCORD_TOKEN=... python bot_discord.py --modelo ratex/xselo-0-5/v1
       No Colab, guarde o token nos Secrets (chave 🔑) como DISCORD_TOKEN e use a célula do bot.

O bot só fica online enquanto o programa estiver rodando (no Colab, enquanto a célula roda).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

LIMITE_DISCORD = 2000
TIPOS_IMAGEM = ("image/png", "image/jpeg", "image/webp", "image/gif")


def args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Xselo no Discord.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--modelo", default="auto", help="pasta do Xselo treinado (ou 'auto' = o mais novo do repositório)")
    p.add_argument("--canal", type=int, action="append", default=[],
                   help="id de canal onde ele responde tudo, sem precisar mencionar (pode repetir)")
    p.add_argument("--busca", nargs="?", const="auto", metavar="URL", help="liga a busca na internet (SearXNG)")
    p.add_argument("--lembrancas", default=str(Path.home() / ".cache" / "ratex" / "lembrancas_discord.json"),
                   help="arquivo onde ficam as lembranças de cada pessoa")
    p.add_argument("--sem-memoria", action="store_true", help="não consulta o dataset (memória de fatos)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def partes_da_mensagem(texto: str, limite: int = LIMITE_DISCORD) -> list[str]:
    """Quebra a resposta em pedaços que cabem no Discord, de preferência entre parágrafos."""
    texto = texto.strip() or "(fiquei sem palavras)"
    partes = []
    while len(texto) > limite:
        corte = max(texto.rfind("\n\n", 0, limite), texto.rfind("\n", 0, limite), texto.rfind(". ", 0, limite) + 1)
        if corte <= limite // 3:
            corte = texto.rfind(" ", 0, limite)
        if corte <= 0:
            corte = limite
        partes.append(texto[:corte].rstrip())
        texto = texto[corte:].lstrip()
    return partes + [texto]


class XseloDiscord:
    """A parte que não depende do Discord: guarda as conversas e gera as respostas."""

    def __init__(self, modelo, tok, cfg: dict, memoria=None, busca=None, lembrancas=None):
        from nucleo.lembrancas import Lembrancas, resumidor

        self.modelo, self.tok, self.cfg = modelo, tok, cfg
        self.memoria, self.busca = memoria, busca
        self.lembrancas = lembrancas if lembrancas is not None else Lembrancas(None)
        self.resumir = resumidor(modelo, tok)
        self.conversas: dict[tuple[int, int], object] = {}
        self.trava = asyncio.Lock()  # a GPU atende uma pergunta por vez

    def conversa(self, canal: int, pessoa: int, nome: str):
        from nucleo.lembrancas import Conversa

        chave = (canal, pessoa)
        if chave not in self.conversas:
            self.conversas[chave] = Conversa(str(pessoa), self.lembrancas, resumir=self.resumir)
            if not self.lembrancas.de(str(pessoa))["fatos"]:
                self.lembrancas.lembrar(str(pessoa), f"no Discord, aparece como {nome}")
        return self.conversas[chave]

    def responder(self, conversa, texto: str, imagens: list[str]) -> str:
        """Bloqueia (gera na GPU): chame por asyncio.to_thread."""
        from nucleo.hibrido import enxerga, responder

        aviso = conversa.comando("/" + texto[1:]) if texto.startswith("!") and not texto.startswith("!buscar") else None
        if aviso:
            return aviso
        if texto.startswith("!buscar"):
            texto = "/buscar" + texto[len("!buscar"):]
        if imagens and not enxerga(self.modelo):
            return "Ainda não enxergo imagens nesta versão, só texto. Me conta com palavras o que tem nela?"
        conversa.usuario(texto or "O que você vê nessa imagem?", imagens)
        resposta = responder(self.modelo, self.tok, conversa.historico,
                             system_prompt=conversa.system(self.cfg["system_prompt"]),
                             memoria=self.memoria, busca=self.busca, stream=False, **self.cfg.get("geracao", {}))
        conversa.xselo(resposta)
        return resposta


def criar_cliente(xselo: XseloDiscord, canais: list[int] | None = None):
    import discord

    intents = discord.Intents.default()
    intents.message_content = True
    cliente = discord.Client(intents=intents)
    canais = set(canais or [])

    @cliente.event
    async def on_ready():
        print(f"✅ Xselo online no Discord como {cliente.user} (em {len(cliente.guilds)} servidor(es))")

    @cliente.event
    async def on_message(msg: "discord.Message"):
        if msg.author.bot:
            return
        eh_dm = msg.guild is None
        mencionou = cliente.user in msg.mentions
        respondeu_ele = (msg.reference is not None and isinstance(msg.reference.resolved, discord.Message)
                         and msg.reference.resolved.author == cliente.user)
        if not (eh_dm or mencionou or respondeu_ele or msg.channel.id in canais):
            return
        texto = msg.content.replace(f"<@{cliente.user.id}>", "").replace(f"<@!{cliente.user.id}>", "").strip()
        imagens = [a.url for a in msg.attachments if (a.content_type or "").split(";")[0] in TIPOS_IMAGEM]
        if not texto and not imagens:
            return
        conversa = xselo.conversa(msg.channel.id, msg.author.id, msg.author.display_name)
        async with xselo.trava:
            async with msg.channel.typing():
                try:
                    resposta = await asyncio.to_thread(xselo.responder, conversa, texto, imagens)
                except Exception as erro:  # nunca derruba o bot por causa de uma mensagem
                    print(f"erro respondendo {msg.author}: {erro!r}", file=sys.stderr)
                    resposta = "Deu um tropeço aqui do meu lado. Tenta de novo daqui a pouco?"
        for i, parte in enumerate(partes_da_mensagem(resposta)):
            if i == 0 and not eh_dm:
                await msg.reply(parte, mention_author=False)
            else:
                await msg.channel.send(parte)

    return cliente


def preparar(modelo_pasta: str = "auto", device: str = "auto", busca: str | None = None,
             arq_lembrancas: str | None = None, com_memoria: bool = True) -> XseloDiscord:
    from nucleo.hibrido import carregar_hibrido, pasta_mais_nova
    from nucleo.lembrancas import Lembrancas

    pasta = pasta_mais_nova() if modelo_pasta == "auto" else Path(modelo_pasta)
    if pasta is None:
        raise SystemExit("nenhum Xselo treinado encontrado: passe --modelo /pasta/do/modelo")
    print(f"carregando {pasta}...")
    modelo, tok, cfg = carregar_hibrido(pasta, device=device)
    memoria = None
    if com_memoria:
        from nucleo.memoria import Memoria

        memoria = Memoria()
    busca_obj = None
    if busca:
        from nucleo.busca import Busca, iniciar_searxng

        busca_obj = Busca(iniciar_searxng() if busca == "auto" else busca)
    return XseloDiscord(modelo, tok, cfg, memoria=memoria, busca=busca_obj, lembrancas=Lembrancas(arq_lembrancas))


def main() -> None:
    args = args_cli()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        sys.exit("falta o token: rode com DISCORD_TOKEN=... python bot_discord.py (veja o passo a passo no topo do arquivo)")
    try:
        import discord  # noqa: F401
    except ImportError:
        sys.exit("instale a biblioteca do Discord: pip install discord.py")
    xselo = preparar(args.modelo, args.device, args.busca, args.lembrancas, not args.sem_memoria)
    criar_cliente(xselo, args.canal).run(token)


if __name__ == "__main__":
    main()
