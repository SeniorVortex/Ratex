#!/usr/bin/env python3
"""
O Xselo com uma API igual à da OpenAI: qualquer site ou app que aceite "OpenAI compatible"
(SillyTavern, Open WebUI, Janitor, extensões de navegador, bots...) conversa com ele usando:

    URL base   https://<endereço>/v1
    chave      a que você escolher (ou uma gerada na hora)
    modelo     xselo-0.5 (ou o nome da versão carregada)

O que tem:
    GET  /v1/models              lista o modelo
    POST /v1/chat/completions    conversa, com ou sem "stream" (texto aparecendo aos poucos),
                                 temperature, top_p, max_tokens, stop, e imagens (image_url com
                                 link http ou data:base64) se a base enxergar

Pra ter um endereço público de graça, o túnel rápido da Cloudflare (trycloudflare.com, sem
conta) aponta um link https pra este servidor enquanto ele estiver ligado.

    python servidor_api.py --modelo ratex/xselo-0-5/v1 --tunel         # no Kaggle/Colab/PC
    python servidor_api.py --modelo ... --chave minha-chave-secreta --porta 8000

O system prompt que o site mandar entra junto com o do Xselo (--system juntar, o padrão), ou
no lugar dele (--system substituir, bom pra personagens de roleplay).
"""


import argparse
import asyncio
import base64
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

CLOUDFLARED = Path.home() / ".cache" / "ratex" / "cloudflared"
URL_CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def _texto_e_imagens(conteudo) -> tuple[str, list]:
    """O "content" da OpenAI pode ser texto ou lista de partes (text / image_url)."""
    if conteudo is None:
        return "", []
    if isinstance(conteudo, str):
        return conteudo, []
    textos, imagens = [], []
    for parte in conteudo:
        if not isinstance(parte, dict):
            continue
        if parte.get("type") == "text":
            textos.append(parte.get("text", ""))
        elif parte.get("type") == "image_url":
            url = parte.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if url:
                imagens.append(_abrir_url_imagem(url))
    return "\n".join(t for t in textos if t), imagens


def _abrir_url_imagem(url: str):
    if url.startswith("data:"):
        from PIL import Image

        dados = base64.b64decode(url.split(",", 1)[1])
        return Image.open(io.BytesIO(dados)).convert("RGB")
    return url  # link http: o abrir_imagem do hibrido baixa


def converter_mensagens(mensagens: list[dict]) -> tuple[str, list[dict]]:
    """Mensagens da OpenAI -> (system do cliente, histórico no formato do Xselo).
    Junta falas seguidas do mesmo papel e, se a conversa começar com o assistente (a saudação
    de um personagem, por exemplo), guarda essa fala no system: vários chat templates exigem
    que a conversa comece pela pessoa."""
    system, historico = [], []
    for m in mensagens:
        papel = m.get("role")
        texto, imagens = _texto_e_imagens(m.get("content"))
        if papel in ("system", "developer"):
            system.append(texto)
            continue
        papel = "assistant" if papel == "assistant" else "user"
        if not historico and papel == "assistant":
            system.append(f"A conversa começou com você dizendo:\n{texto}")
            continue
        if historico and historico[-1]["role"] == papel:
            historico[-1]["content"] += "\n\n" + texto
            if imagens:
                historico[-1].setdefault("imagens", []).extend(imagens)
        else:
            historico.append({"role": papel, "content": texto, **({"imagens": imagens} if imagens else {})})
    if not historico or historico[-1]["role"] != "user":
        historico.append({"role": "user", "content": "(continue)"})
    return "\n\n".join(s for s in system if s), historico


def criar_app(modelo, tok, cfg: dict, chave: str, nome_modelo: str | None = None, memoria=None, busca=None,
              modo_system: str = "juntar", max_tokens_teto: int = 1024):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    from nucleo.hibrido import responder_em_partes

    nome_modelo = nome_modelo or f"xselo-{cfg.get('versao', '')}".rstrip("-")
    app = FastAPI(title="Xselo (API compatível com OpenAI)")
    # sites que chamam direto do navegador (Janitor, extensões...) precisam de CORS liberado
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    trava = threading.Lock()  # uma GPU: um pedido de cada vez, os outros esperam na fila

    def conferir(request: Request):
        recebida = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        recebida = recebida or request.headers.get("x-api-key", "")
        if not secrets.compare_digest(recebida, chave):
            raise HTTPException(401, detail={"error": {"message": "chave (API key) errada", "type": "invalid_api_key"}})

    @app.get("/")
    def raiz():
        return {"ok": True, "modelo": nome_modelo, "dica": "use a URL base terminando em /v1"}

    @app.get("/v1/models")
    def modelos(request: Request):
        conferir(request)
        return {"object": "list", "data": [{"id": nome_modelo, "object": "model", "created": int(time.time()),
                                            "owned_by": "ratex"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        conferir(request)
        pedido = await request.json()
        system_cliente, historico = converter_mensagens(pedido.get("messages") or [])
        if modo_system == "substituir" and system_cliente:
            system = system_cliente
        else:
            system = cfg["system_prompt"] + (f"\n\nInstruções de quem está usando:\n{system_cliente}"
                                             if system_cliente else "")
        ger = dict(cfg.get("geracao", {}))
        if pedido.get("temperature") is not None:
            ger["temperatura"] = float(pedido["temperature"])
        if pedido.get("top_p") is not None:
            ger["top_p"] = float(pedido["top_p"])
        limite = pedido.get("max_completion_tokens") or pedido.get("max_tokens")
        if limite:
            ger["max_novos_tokens"] = max(1, min(int(limite), max_tokens_teto))
        paradas = pedido.get("stop") or []
        paradas = [paradas] if isinstance(paradas, str) else [p for p in paradas if p]
        ident, criado = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        parar = threading.Event()
        fila: asyncio.Queue = asyncio.Queue()
        laco = asyncio.get_running_loop()

        def trabalhar():
            """Roda numa thread: gera e vai colocando os pedaços na fila (None = acabou)."""
            texto = ""
            try:
                with trava:
                    for pedaco in responder_em_partes(modelo, tok, historico, system_prompt=system, memoria=memoria,
                                                      busca=busca, parar=parar, **ger):
                        texto += pedaco
                        corte = min((texto.find(p) for p in paradas if p in texto), default=-1)
                        if corte >= 0:  # achou uma palavra de parada: entrega até ela e encerra
                            enviado = len(texto) - len(pedaco)
                            laco.call_soon_threadsafe(fila.put_nowait, ("texto", texto[enviado:corte]))
                            parar.set()
                            break
                        laco.call_soon_threadsafe(fila.put_nowait, ("texto", pedaco))
                laco.call_soon_threadsafe(fila.put_nowait, ("fim", "stop"))
            except Exception as erro:
                laco.call_soon_threadsafe(fila.put_nowait, ("erro", str(erro)))

        threading.Thread(target=trabalhar, daemon=True).start()

        def pedaco_sse(delta: dict, fim: str | None = None) -> str:
            corpo = {"id": ident, "object": "chat.completion.chunk", "created": criado, "model": nome_modelo,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fim}]}
            return f"data: {json.dumps(corpo, ensure_ascii=False)}\n\n"

        if pedido.get("stream"):
            async def eventos():
                yield pedaco_sse({"role": "assistant", "content": ""})
                try:
                    while True:
                        try:
                            tipo, valor = await asyncio.wait_for(fila.get(), timeout=15)
                        except asyncio.TimeoutError:
                            yield ": esperando a vez na fila\n\n"  # comentário SSE: mantém a conexão viva
                            continue
                        if tipo == "texto":
                            if valor:
                                yield pedaco_sse({"content": valor})
                        elif tipo == "erro":
                            yield pedaco_sse({"content": f"\n[erro no Xselo: {valor}]"}, "stop")
                            break
                        else:
                            yield pedaco_sse({}, valor)
                            break
                    yield "data: [DONE]\n\n"
                finally:
                    parar.set()  # quem pediu desistiu (fechou a aba): a GPU para de gerar
            return StreamingResponse(eventos(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        async def resposta_inteira():
            """Sem stream, a resposta só sai no fim. Enquanto isso, manda espaços em branco (JSON
            aceita espaço antes do conteúdo) pra o túnel da Cloudflare não cortar por demora (~100 s)."""
            partes = []
            while True:
                try:
                    tipo, valor = await asyncio.wait_for(fila.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield " "
                    continue
                if tipo == "texto":
                    partes.append(valor)
                elif tipo == "erro":
                    yield json.dumps({"error": {"message": valor, "type": "server_error"}}, ensure_ascii=False)
                    return
                else:
                    break
            texto = "".join(partes).strip()
            yield json.dumps({
                "id": ident, "object": "chat.completion", "created": criado, "model": nome_modelo,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": len(tok(texto)["input_ids"]), "total_tokens": 0},
            }, ensure_ascii=False)
        return StreamingResponse(resposta_inteira(), media_type="application/json")

    @app.exception_handler(HTTPException)
    async def erro_http(request, exc):
        return JSONResponse(exc.detail if isinstance(exc.detail, dict) else {"error": {"message": str(exc.detail)}},
                            status_code=exc.status_code)

    return app


def ligar_servidor(app, porta: int = 8000) -> threading.Thread:
    """Liga o servidor numa thread (serve no notebook, que já tem um laço de eventos rodando)."""
    import uvicorn

    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=porta, log_level="warning"))
    fio = threading.Thread(target=servidor.run, daemon=True)
    fio.start()
    for _ in range(100):
        if servidor.started:
            break
        time.sleep(0.1)
    return fio


def abrir_tunel(porta: int = 8000, esperar: float = 90.0) -> str:
    """Túnel rápido da Cloudflare (grátis, sem conta): devolve o https://....trycloudflare.com.
    Só devolve depois que a conexão com a Cloudflare está registrada (antes disso o link dá erro
    1033). Usa HTTP2, que passa em mais redes que o QUIC. O endereço muda toda vez que liga."""
    binario = shutil.which("cloudflared") or str(CLOUDFLARED)
    if not Path(binario).exists():
        CLOUDFLARED.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_CLOUDFLARED, CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
        binario = str(CLOUDFLARED)
    log = CLOUDFLARED.parent / "cloudflared.log"
    saida = open(log, "w", encoding="utf-8")
    subprocess.Popen([binario, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{porta}"],
                     stdout=saida, stderr=saida)
    inicio, url = time.time(), None
    while time.time() - inicio < esperar:
        time.sleep(1)
        texto = log.read_text(encoding="utf-8", errors="ignore")
        achou = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", texto)
        url = achou.group(0) if achou else url
        if url and "Registered tunnel connection" in texto:
            return url
    raise RuntimeError("o túnel da Cloudflare não conectou (a rede pode estar bloqueando a porta 7844). "
                       f"Veja {log}")


def main() -> None:
    p = argparse.ArgumentParser(description="Xselo com API compatível com OpenAI.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--modelo", default="auto", help="pasta do Xselo treinado")
    p.add_argument("--base", help="pasta local do modelo base (se não quiser baixar)")
    p.add_argument("--chave", default=os.environ.get("XSELO_CHAVE"), help="API key (padrão: uma gerada na hora)")
    p.add_argument("--nome", help="nome do modelo na API (padrão: xselo-<versão>)")
    p.add_argument("--porta", type=int, default=8000)
    p.add_argument("--tunel", action="store_true", help="abre um link público grátis (trycloudflare.com)")
    p.add_argument("--system", choices=["juntar", "substituir"], default="juntar",
                   help="o system prompt do site entra junto com o do Xselo, ou no lugar dele")
    p.add_argument("--busca", nargs="?", const="auto", metavar="URL", help="liga a busca na internet (SearXNG)")
    p.add_argument("--sem-memoria", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from nucleo.hibrido import carregar_hibrido, pasta_mais_nova

    pasta = pasta_mais_nova() if args.modelo == "auto" else Path(args.modelo)
    modelo, tok, cfg = carregar_hibrido(pasta, device=args.device, base=args.base)
    memoria = None
    if not args.sem_memoria:
        from nucleo.memoria import Memoria

        memoria = Memoria()
    busca = None
    if args.busca:
        from nucleo.busca import Busca, iniciar_searxng

        busca = Busca(iniciar_searxng() if args.busca == "auto" else args.busca)
    chave = args.chave or "xselo-" + secrets.token_urlsafe(18)
    app = criar_app(modelo, tok, cfg, chave, args.nome, memoria, busca, args.system)
    ligar_servidor(app, args.porta)
    url = abrir_tunel(args.porta) if args.tunel else f"http://127.0.0.1:{args.porta}"
    nome = args.nome or f"xselo-{cfg.get('versao', '')}"
    print(f"\n✅ Xselo no ar!\n   URL base:  {url}/v1\n   API key:   {chave}\n   modelo:    {nome}\n")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
