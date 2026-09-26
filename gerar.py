#!/usr/bin/env python3
"""
Conversa / gera texto com o Xselo.

Por padrão usa o ratex/xselo-0-2/v1 (modelo base + LoRA) se ele já foi treinado,
e cai para o ratex/xselo-0-1/v1 (micro-Transformer feito do zero) se não foi.

Exemplos:
    python gerar.py --chat                       # bate-papo com o Xselo (usa o xselo-0-2)
    python gerar.py "quem é a Cirno?"            # uma pergunta só
    python gerar.py --modelo ratex/xselo-0-1/v1 "Touhou é"   # o v1 continua um texto
    python gerar.py --prompt "Reimu" --amostras 3
    python gerar.py                              # modo interativo

Controles de amostragem:
    --temperatura 0.8   menor = mais conservador/repetitivo, maior = mais criativo/caótico
    --top-k 40          só sorteia entre os 40 tokens mais prováveis
    --top-p 0.95        ...e dentre eles só os que somam 95% da probabilidade
    --penalidade 1.1    penaliza tokens usados recentemente (bom com --tokenizador bpe)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nucleo import NOME_MODELO, PASTA_PADRAO, carregar_modelo
from nucleo.hibrido import PASTA_HIBRIDO, eh_hibrido

PREFIXO_PESSOA = "Pessoa:"
PREFIXO_XSELO = "Xselo:"


def args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Gera texto com o {NOME_MODELO}.")
    p.add_argument("prompt_posicional", nargs="?", metavar="PROMPT", help="texto inicial")
    p.add_argument("--prompt", "-p", help="texto inicial (mesmo que o argumento posicional)")
    p.add_argument("--chat", action="store_true", help="modo conversa: você pergunta, o Xselo responde")
    p.add_argument("--modelo", default="auto",
                   help="pasta do modelo; 'auto' = ratex/xselo-0-2/v1 se existir, senão ratex/xselo-0-1/v1")
    p.add_argument("--base", help="xselo-0-2: modelo base alternativo (ex.: pasta local já baixada)")
    p.add_argument("--tokens", type=int, help="máximo de tokens novos por geração")
    p.add_argument("--temperatura", "-t", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--penalidade", type=float, help="penalidade de repetição (1.0 = desligada)")
    p.add_argument("--amostras", "-n", type=int, default=1, help="quantas versões gerar para o mesmo prompt")
    p.add_argument("--parar-em", action="append", default=[],
                   help="interrompe a geração quando este texto aparecer (pode repetir a opção)")
    p.add_argument("--seed", type=int, help="semente para resultados reproduzíveis")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps...")
    return p.parse_args()


def em_loop(texto: str, repeticoes: int = 3, min_len: int = 8, max_len: int = 120) -> bool:
    """True se o final do texto é o mesmo trecho repetido várias vezes seguidas."""
    for n in range(min_len, max_len + 1):
        if len(texto) < n * repeticoes:
            break
        trecho = texto[-n:]
        if texto.endswith(trecho * repeticoes):
            return True
    return False


class Gerador:
    def __init__(self, modelo, tok, amostragem: dict, device: str):
        self.modelo = modelo
        self.tok = tok
        self.amostragem = amostragem
        self.device = device

    def avisar_desconhecidos(self, texto: str) -> None:
        faltando = self.tok.desconhecidos(texto)
        if faltando:
            print(f"(aviso: o modelo não conhece {''.join(sorted(faltando))!r}; esses caracteres serão ignorados)",
                  file=sys.stderr)

    @torch.no_grad()
    def gerar(self, contexto: str, max_novos: int, parar_em: list[str], stream: bool = True) -> str:
        ids = self.tok.encode(contexto)
        if not ids:
            ids = self.tok.encode("\n") or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        gerado, impresso = "", 0
        for _ in range(max_novos):
            prox = self.modelo.proximo_token(idx, **self.amostragem)
            idx = torch.cat([idx, prox], dim=1)
            gerado += self.tok.decode([prox.item()])

            parada = min((gerado.find(s) for s in parar_em if s in gerado), default=-1)
            if parada >= 0:
                gerado = gerado[:parada]
                break
            if em_loop(gerado):
                break
            if stream:
                # segura na tela o pedaço que pode ser o começo de uma sequência de parada
                seguro = len(gerado)
                for s in parar_em:
                    for k in range(min(len(s) - 1, len(gerado)), 0, -1):
                        if gerado.endswith(s[:k]):
                            seguro = min(seguro, len(gerado) - k)
                            break
                if seguro > impresso:
                    print(gerado[impresso:seguro], end="", flush=True)
                    impresso = seguro
        if stream:
            print(gerado[impresso:], end="", flush=True)
            print()
        return gerado


def modo_prompt(g: Gerador, prompt: str, args, max_novos: int) -> None:
    g.avisar_desconhecidos(prompt)
    for i in range(args.amostras):
        if args.amostras > 1:
            print(f"\n----- amostra {i + 1}/{args.amostras} -----")
        print(prompt, end="", flush=True)
        g.gerar(prompt, max_novos, args.parar_em)


def modo_interativo(g: Gerador, args, max_novos: int) -> None:
    print("Digite o começo de um texto e o Xselo continua. (Enter vazio ou Ctrl+C para sair)")
    while True:
        try:
            prompt = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt.strip():
            return
        modo_prompt(g, prompt, args, max_novos)


def modo_chat(g: Gerador, args, max_novos: int) -> None:
    print("Papo com o Xselo, especialista em Touhou. Comandos: /novo (esquece a conversa), /sair")
    historico: list[str] = []
    while True:
        try:
            msg = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg in ("/sair", "/exit", "/quit"):
            return
        if msg == "/novo":
            historico.clear()
            print("(conversa zerada)")
            continue
        g.avisar_desconhecidos(msg)
        historico.append(f"{PREFIXO_PESSOA} {msg}")
        contexto = "\n".join(historico) + f"\n{PREFIXO_XSELO}"
        print("xselo>", end="", flush=True)
        resposta = g.gerar(contexto, max_novos, ["\n", *args.parar_em]).strip()
        historico.append(f"{PREFIXO_XSELO} {resposta}")
        # o modelo só enxerga block_size tokens; guarda só o finalzinho da conversa
        historico[:] = historico[-8:]


def ler_linha(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def main_hibrido(args, pasta: Path) -> None:
    """xselo-0-2: modelo base instruído + LoRA do Xselo, com o chat template do modelo base."""
    from nucleo.hibrido import carregar_hibrido, responder

    print(f"carregando {pasta} (modelo base + LoRA)...", file=sys.stderr)
    modelo, tok, cfg = carregar_hibrido(pasta, device=args.device, base=args.base)
    ger = dict(cfg.get("geracao", {}))
    for chave, valor in (("temperatura", args.temperatura), ("top_k", args.top_k), ("top_p", args.top_p),
                         ("penalidade_repeticao", args.penalidade), ("max_novos_tokens", args.tokens)):
        if valor is not None:
            ger[chave] = valor
    print(f"[{cfg['nome']} | base {args.base or cfg['base']} + LoRA r={cfg['lora']['rank']} | "
          f"temp {ger.get('temperatura')} | top-k {ger.get('top_k')} | top-p {ger.get('top_p')}]", file=sys.stderr)
    system = cfg.get("system_prompt")

    def perguntar(historico: list[dict]) -> str:
        print("xselo> ", end="", flush=True)
        return responder(modelo, tok, historico, system_prompt=system, **ger)

    prompt = args.prompt or args.prompt_posicional
    if prompt and not args.chat:
        for i in range(args.amostras):
            if args.amostras > 1:
                print(f"\n----- amostra {i + 1}/{args.amostras} -----")
            print(f"você> {prompt}")
            perguntar([{"role": "user", "content": prompt}])
        return

    print("Papo com o Xselo 0.2, especialista em Touhou. Comandos: /novo (esquece a conversa), /sair")
    historico: list[dict] = []
    while True:
        msg = prompt if prompt else ler_linha("\nvocê> ")
        prompt = None
        if msg is None or msg in ("/sair", "/exit", "/quit"):
            return
        if not msg:
            continue
        if msg == "/novo":
            historico.clear()
            print("(conversa zerada)")
            continue
        historico.append({"role": "user", "content": msg})
        historico.append({"role": "assistant", "content": perguntar(historico)})
        historico[:] = historico[-12:]  # memória: as últimas 6 trocas


def main() -> None:
    args = args_cli()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.modelo == "auto":
        pasta = PASTA_HIBRIDO if eh_hibrido(PASTA_HIBRIDO) else PASTA_PADRAO
    else:
        pasta = Path(args.modelo)
    if eh_hibrido(pasta):
        return main_hibrido(args, pasta)
    if args.device == "auto":
        args.device = "cpu"  # o v1 é minúsculo: CPU basta
    try:
        modelo, tok, config, geracao = carregar_modelo(pasta, device=args.device)
    except FileNotFoundError as e:
        sys.exit(str(e))

    amostragem = {
        "temperatura": geracao["temperatura"] if args.temperatura is None else args.temperatura,
        "top_k": geracao["top_k"] if args.top_k is None else args.top_k,
        "top_p": geracao["top_p"] if args.top_p is None else args.top_p,
        "penalidade_repeticao": geracao["penalidade_repeticao"] if args.penalidade is None else args.penalidade,
        "janela_repeticao": geracao["janela_repeticao"],
    }
    max_novos = args.tokens or geracao["max_novos_tokens"]
    print(f"[{config.get('nome', NOME_MODELO)} | {config.get('n_parametros', 0) / 1e6:.2f}M parâmetros | "
          f"temp {amostragem['temperatura']} | top-k {amostragem['top_k']} | top-p {amostragem['top_p']}]",
          file=sys.stderr)

    g = Gerador(modelo, tok, amostragem, args.device)
    prompt = args.prompt or args.prompt_posicional
    if args.chat:
        modo_chat(g, args, max_novos)
    elif prompt:
        modo_prompt(g, prompt, args, max_novos)
    else:
        modo_interativo(g, args, max_novos)


if __name__ == "__main__":
    main()
