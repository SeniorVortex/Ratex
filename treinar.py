#!/usr/bin/env python3
"""
Treina do zero o ratex/xselo-0-1/v1, um micro-Transformer (estilo nanoGPT)
especialista em Touhou Project.

Uso rápido:
    python treinar.py                  # escolhe o preset sozinho (gpu se tiver CUDA, senão cpu)
    python treinar.py --preset teste   # treino relâmpago só para ver se tudo funciona
    python treinar.py --preset gpu --max-iters 8000
    python treinar.py --retomar        # continua de onde o último treino parou

Veja `python treinar.py --help` para todas as opções.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from nucleo import (
    NOME_MODELO,
    PASTA_PADRAO,
    ConfigXselo,
    XseloGPT,
    normalizar_texto,
    salvar_modelo,
    treinar_tokenizador,
)
from nucleo.tokenizador import TokenizadorBPE, TokenizadorChar

RAIZ = Path(__file__).resolve().parent
CHECKPOINT_PADRAO = RAIZ / "checkpoints" / "xselo-0-1-v1_treino.pt"

# Presets pensados para o tamanho do dataset (~centenas de KB de texto).
# Qualquer valor pode ser sobrescrito pela linha de comando.
PRESETS = {
    # só valida que o pipeline inteiro roda (segundos)
    "teste": dict(n_layer=2, n_head=2, n_embd=64, block_size=64, batch_size=16, max_iters=150,
                  eval_interval=50, eval_iters=5, lr=2e-3, dropout=0.0, warmup=15),
    # CPU comum (4-8 núcleos): ~3M de parâmetros, ~40 min em 4 núcleos
    "cpu": dict(n_layer=4, n_head=4, n_embd=256, block_size=192, batch_size=32, max_iters=2500,
                eval_interval=150, eval_iters=20, lr=1e-3, dropout=0.2, warmup=100),
    # GPU (CUDA/MPS): ~11M de parâmetros, contexto maior
    "gpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=3000,
                eval_interval=250, eval_iters=50, lr=1e-3, dropout=0.2, warmup=200),
}


def args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Treina o {NOME_MODELO} do zero.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("dados e saída")
    g.add_argument("--dados", nargs="+", default=["dataset.txt", "dados_extras"],
                   help="arquivos .txt e/ou pastas (todos os .txt dentro delas entram no treino)")
    g.add_argument("--saida", default=str(PASTA_PADRAO), help="pasta onde o modelo final é salvo")
    g.add_argument("--checkpoint", default=str(CHECKPOINT_PADRAO),
                   help="checkpoint de treino (pesos + otimizador) para poder retomar")
    g.add_argument("--retomar", action="store_true", help="continua o treino a partir do --checkpoint")
    g.add_argument("--fracao-validacao", type=float, default=0.1)

    g = p.add_argument_group("tokenizador")
    g.add_argument("--tokenizador", choices=["char", "bpe"], default="char")
    g.add_argument("--vocab-bpe", type=int, default=512, help="tamanho do vocabulário quando --tokenizador bpe")

    g = p.add_argument_group("arquitetura (sobrescreve o preset)")
    g.add_argument("--preset", choices=["auto", *PRESETS], default="auto")
    g.add_argument("--n-layer", type=int)
    g.add_argument("--n-head", type=int)
    g.add_argument("--n-embd", type=int)
    g.add_argument("--block-size", type=int)
    g.add_argument("--dropout", type=float)

    g = p.add_argument_group("otimização (sobrescreve o preset)")
    g.add_argument("--batch-size", type=int)
    g.add_argument("--max-iters", type=int)
    g.add_argument("--epocas", type=float, help="alternativa a --max-iters: quantas passadas pelo dataset")
    g.add_argument("--lr", type=float, help="learning rate máximo")
    g.add_argument("--warmup", type=int, help="iterações de aquecimento do learning rate")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--eval-interval", type=int)
    g.add_argument("--eval-iters", type=int)
    g.add_argument("--log-interval", type=int, default=25)
    g.add_argument("--tempo-max", type=float, help="para (e salva) depois de N minutos")
    g.add_argument("--paciencia", type=int, default=4,
                   help="para cedo se a validação não melhorar por N avaliações seguidas (0 desliga)")
    g.add_argument("--criterio", choices=["validacao", "final"], default="validacao",
                   help="salvar o modelo com a menor loss de validação, ou o do fim do treino")

    g = p.add_argument_group("máquina")
    g.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps...")
    g.add_argument("--compile", action="store_true", help="usa torch.compile (mais rápido em GPU)")
    g.add_argument("--fp16", action="store_true", help="salva os pesos em float16 (arquivo 2x menor)")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--prompt-amostra", default="Pessoa: o que é Touhou?\nXselo:",
                   help="texto usado para mostrar uma amostra a cada avaliação ('' desliga)")
    return p.parse_args()


def escolher_device(pedido: str) -> str:
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ler_dados(fontes: list[str]) -> tuple[str, list[Path]]:
    arquivos: list[Path] = []
    for fonte in fontes:
        caminho = Path(fonte)
        if not caminho.is_absolute():
            caminho = RAIZ / caminho
        if caminho.is_dir():
            arquivos += sorted(caminho.rglob("*.txt"))
        elif caminho.is_file():
            arquivos.append(caminho)
        else:
            print(f"aviso: {fonte} não existe, pulando")
    textos = [normalizar_texto(a.read_text(encoding="utf-8")).strip() for a in arquivos]
    texto = "\n\n".join(t for t in textos if t) + "\n"
    return texto, arquivos


def dividir_treino_validacao(dados: torch.Tensor, fracao: float, bloco: int):
    """Separa pedaços intercalados para validação (em vez de só o final do arquivo),
    assim a validação vê um pouco de cada seção do dataset."""
    tam = max(bloco * 4, 1024)
    pedacos = list(torch.split(dados, tam))
    passo = max(round(1 / fracao), 2) if fracao > 0 else 0
    val = [p for i, p in enumerate(pedacos) if passo and i % passo == passo - 1]
    treino = [p for i, p in enumerate(pedacos) if not (passo and i % passo == passo - 1)]
    if not val or sum(len(p) for p in val) <= bloco + 1:
        # dataset pequeno demais: usa o finalzinho
        corte = int(len(dados) * (1 - max(fracao, 0.1)))
        return dados[:corte], dados[corte:]
    return torch.cat(treino), torch.cat(val)


def lr_no_passo(it: int, lr_max: float, warmup: int, total: int) -> float:
    """Aquecimento linear e depois decaimento cosseno até 10% do lr máximo."""
    lr_min = lr_max / 10
    if it < warmup:
        return lr_max * (it + 1) / warmup
    if it >= total:
        return lr_min
    progresso = (it - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * progresso)) * (lr_max - lr_min)


def formatar_tempo(seg: float) -> str:
    seg = int(seg)
    h, resto = divmod(seg, 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    args = args_cli()
    device = escolher_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device.startswith("mps") else "cpu")
    preset = args.preset if args.preset != "auto" else ("cpu" if device_type == "cpu" else "gpu")
    hp = dict(PRESETS[preset])
    for chave in list(hp):
        valor = getattr(args, chave, None)
        if valor is not None:
            hp[chave] = valor

    torch.manual_seed(args.seed)
    if device_type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"== Ratex :: {NOME_MODELO} ==")
    print(f"device: {device} | preset: {preset} | torch {torch.__version__}")

    # ---------------------------------------------------------------- dados
    texto, arquivos = ler_dados(args.dados)
    if len(texto) < 1000:
        sys.exit("dataset muito pequeno (< 1000 caracteres). Coloque mais texto em dataset.txt!")
    print(f"dados: {len(arquivos)} arquivo(s), {len(texto):,} caracteres")
    for a in arquivos:
        print(f"  - {a.relative_to(RAIZ) if a.is_relative_to(RAIZ) else a}")

    ckpt = None
    caminho_ckpt = Path(args.checkpoint)
    if args.retomar:
        if not caminho_ckpt.exists():
            sys.exit(f"--retomar pedido, mas {caminho_ckpt} não existe")
        ckpt = torch.load(caminho_ckpt, map_location="cpu", weights_only=False)
        tdict = ckpt["tokenizador"]
        tok = TokenizadorChar(tdict["tokens"]) if tdict["tipo"] == "char" else TokenizadorBPE(tdict["tokens"], tdict["merges"])
        cfg = ConfigXselo.de_dict(ckpt["config"])
        faltando = tok.desconhecidos(texto)
        if faltando:
            print(f"aviso: {len(faltando)} caractere(s) novos não existem no vocabulário antigo e serão ignorados: "
                  f"{''.join(sorted(faltando))[:60]!r}")
        print(f"retomando de {caminho_ckpt} (iteração {ckpt['iter']})")
    else:
        t0 = time.time()
        tok = treinar_tokenizador(args.tokenizador, texto, vocab_size=args.vocab_bpe)
        print(f"tokenizador: {tok.tipo}, {tok.vocab_size} tokens ({time.time() - t0:.1f}s)")
        cfg = ConfigXselo(
            vocab_size=tok.vocab_size,
            block_size=hp["block_size"],
            n_layer=hp["n_layer"],
            n_head=hp["n_head"],
            n_embd=hp["n_embd"],
            dropout=hp["dropout"],
        )

    dados = torch.tensor(tok.encode(texto), dtype=torch.long)
    treino, val = dividir_treino_validacao(dados, args.fracao_validacao, cfg.block_size)
    print(f"tokens: {len(dados):,} (treino {len(treino):,} | validação {len(val):,})")

    # ---------------------------------------------------------------- modelo
    modelo = XseloGPT(cfg).to(device)
    if ckpt:
        modelo.load_state_dict(ckpt["modelo"])
    print(f"modelo: {cfg.n_layer} camadas, {cfg.n_head} cabeças, n_embd {cfg.n_embd}, "
          f"contexto {cfg.block_size} -> {modelo.n_parametros() / 1e6:.2f}M parâmetros")

    otim = modelo.otimizador(hp["lr"], args.weight_decay, device_type=device_type)
    if ckpt:
        otim.load_state_dict(ckpt["otimizador"])

    modelo_exec = torch.compile(modelo) if args.compile else modelo

    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    else:
        autocast = nullcontext()
        scaler = None

    B, T = hp["batch_size"], cfg.block_size
    tokens_por_iter = B * T
    max_iters = hp["max_iters"]
    if args.epocas:
        max_iters = math.ceil(args.epocas * len(treino) / tokens_por_iter)
    it_inicio = ckpt["iter"] if ckpt else 0
    if ckpt and it_inicio >= max_iters:
        max_iters = it_inicio + hp["max_iters"]
        print(f"o checkpoint já passou do --max-iters; treinando mais {hp['max_iters']} iterações")
    melhor_val = ckpt["melhor_val"] if ckpt else float("inf")
    print(f"treino: {max_iters} iterações x {B} sequências x {T} tokens "
          f"(~{max_iters * tokens_por_iter / len(treino):.1f} épocas) | lr {hp['lr']}")
    print()

    def pegar_lote(fonte: torch.Tensor):
        ix = torch.randint(len(fonte) - T - 1, (B,))
        x = torch.stack([fonte[i : i + T] for i in ix])
        y = torch.stack([fonte[i + 1 : i + 1 + T] for i in ix])
        if device_type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    @torch.no_grad()
    def avaliar() -> dict:
        modelo_exec.eval()
        saida = {}
        for nome, fonte in (("treino", treino), ("validacao", val)):
            perdas = torch.zeros(hp["eval_iters"])
            for k in range(hp["eval_iters"]):
                x, y = pegar_lote(fonte)
                with autocast:
                    _, loss = modelo_exec(x, y)
                perdas[k] = loss.item()
            saida[nome] = perdas.mean().item()
        modelo_exec.train()
        return saida

    @torch.no_grad()
    def amostra(n: int = 240) -> str:
        if not args.prompt_amostra:
            return ""
        modelo.eval()
        idx = torch.tensor([tok.encode(args.prompt_amostra)], dtype=torch.long, device=device)
        out = modelo.gerar(idx, n, temperatura=0.8, top_k=40, top_p=0.95)
        modelo.train()
        return tok.decode(out[0].tolist())

    info_treino: dict = {}
    salvou = False

    def salvar(it: int, perdas: dict | None, motivo: str) -> None:
        nonlocal salvou
        salvou = True
        info_treino.update({
            "iteracoes": it,
            "loss_treino": round(perdas["treino"], 4) if perdas else None,
            "loss_validacao": round(perdas["validacao"], 4) if perdas else None,
            "melhor_loss_validacao": round(melhor_val, 4) if melhor_val < float("inf") else None,
            "tokens_dataset": len(dados),
            "caracteres_dataset": len(texto),
            "arquivos_dataset": [str(a.relative_to(RAIZ)) if a.is_relative_to(RAIZ) else str(a) for a in arquivos],
            "batch_size": B,
            "lr": hp["lr"],
            "preset": preset,
            "device": device,
            "data": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        salvar_modelo(args.saida, modelo, tok, extra={"treino": info_treino}, fp16=args.fp16)
        print(f"   -> modelo salvo em {args.saida} ({motivo})")

    def salvar_checkpoint(it: int) -> None:
        caminho_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "modelo": modelo.state_dict(),
            "otimizador": otim.state_dict(),
            "config": cfg.para_dict(),
            "tokenizador": tok.para_dict(),
            "iter": it,
            "melhor_val": melhor_val,
        }, caminho_ckpt)

    # ---------------------------------------------------------------- loop
    modelo_exec.train()
    t_inicio = time.time()
    t_ult = t_inicio
    it = it_inicio
    perdas = None
    sem_melhora = 0
    try:
        while True:
            fim = it >= max_iters or (args.tempo_max and time.time() - t_inicio > args.tempo_max * 60)
            if it % hp["eval_interval"] == 0 or fim:
                perdas = avaliar()
                melhorou = perdas["validacao"] < melhor_val
                marca = "  <- melhor até agora" if melhorou else ""
                print(f"== avaliação it {it}: loss treino {perdas['treino']:.4f} | "
                      f"validação {perdas['validacao']:.4f}{marca}")
                if melhorou:
                    melhor_val = perdas["validacao"]
                    sem_melhora = 0
                    if args.criterio == "validacao" and it > it_inicio:
                        salvar(it, perdas, "melhor validação")
                elif it > it_inicio:
                    sem_melhora += 1
                if it > it_inicio:
                    salvar_checkpoint(it)
                if args.paciencia and sem_melhora >= args.paciencia and not fim:
                    print(f"parada antecipada: a validação não melhora há {sem_melhora} avaliações "
                          f"(o modelo começou a decorar o texto em vez de aprender)")
                    break
                if it > it_inicio and it % (hp["eval_interval"] * 2) == 0 and not fim:
                    print("   amostra:", amostra().replace("\n", "\n   | "))
            if fim:
                break

            lr = lr_no_passo(it, hp["lr"], hp["warmup"], max_iters)
            for grupo in otim.param_groups:
                grupo["lr"] = lr
            x, y = pegar_lote(treino)
            with autocast:
                _, loss = modelo_exec(x, y)
            otim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(otim)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), args.grad_clip)
                scaler.step(otim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), args.grad_clip)
                otim.step()
            it += 1

            if it % args.log_interval == 0:
                agora = time.time()
                ms = (agora - t_ult) * 1000 / args.log_interval
                t_ult = agora
                eta = (max_iters - it) * ms / 1000
                epoca = it * tokens_por_iter / len(treino)
                print(f"iter {it:>6}/{max_iters} | época {epoca:6.2f} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {ms:6.1f} ms/it | falta ~{formatar_tempo(eta)}")
    except KeyboardInterrupt:
        print("\ninterrompido (Ctrl+C) - salvando o que já foi aprendido...")
        perdas = avaliar()
        salvar_checkpoint(it)

    if args.criterio == "final" or not salvou:
        salvar(it, perdas, "fim do treino")
    print(f"\ntreino concluído em {formatar_tempo(time.time() - t_inicio)} | "
          f"melhor loss de validação: {melhor_val:.4f}")
    print(f"checkpoint para retomar: {caminho_ckpt}")
    print("\namostra final:\n" + amostra(400))
    print("\nagora é só rodar:  python gerar.py --chat")


if __name__ == "__main__":
    main()
