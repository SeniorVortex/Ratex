#!/usr/bin/env python3
"""
Treina o Xselo híbrido: um adaptador LoRA em cima de um modelo base instruído.

    --versao 0.3 (padrão)  ratex/xselo-0-3/v1: Touhou + assuntos gerais + matemática,
                           base escolhida pelo hardware (Qwen2.5 de 0.5B até 7B)
    --versao 0.2           ratex/xselo-0-2/v1: só Touhou, em cima do Qwen2.5-0.5B-Instruct

Uso rápido:
    python treinar_lora.py                                  # 0.3, base automática
    python treinar_lora.py --base qwen-7b --4bit            # 0.3 no Qwen2.5-7B com QLoRA (GPU com ~16 GB)
    python treinar_lora.py --base qwen-1.5b                 # apelidos: qwen-0.5b/1.5b/3b/7b, smollm2-1.7b...
    python treinar_lora.py --versao 0.2                     # a versão anterior, só Touhou
    python treinar_lora.py --exportar-conversas conv.jsonl  # só mostra os dados de chat

Veja `python treinar_lora.py --help` para todas as opções.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from nucleo.dados_chat import construir_conversas_com_origem, conversas_de_matematica, ler_textos
from nucleo.hibrido import (
    ARQ_RATEX,
    BASES,
    GERACAO_HIBRIDO,
    VERSAO_PADRAO,
    VERSOES,
    carregar_base,
    escolher_device,
    montar_mensagens,
    pasta_da_versao,
    resolver_base,
    responder,
)

RAIZ = Path(__file__).resolve().parent


def args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina o LoRA do Xselo híbrido.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--versao", choices=sorted(VERSOES), default=VERSAO_PADRAO)
    p.add_argument("--base", help="modelo base: 'auto', um apelido (" + ", ".join(BASES) +
                   "), um id do Hugging Face ou uma pasta local. Padrão: o da versão")
    p.add_argument("--4bit", dest="quatro_bits", action="store_true",
                   help="QLoRA: carrega a base em 4 bits (GPU NVIDIA + bitsandbytes). Ligado sozinho no 'auto' quando precisa")
    p.add_argument("--checkpointing", action="store_true",
                   help="gradient checkpointing: bem menos memória, ~30%% mais lento (bom pra bases de 3B/7B)")
    p.add_argument("--dados", nargs="+", help="arquivos/pastas de texto. Padrão: o da versão")
    p.add_argument("--matematica", type=int, help="quantos exercícios de matemática gerados entram. Padrão: o da versão")
    p.add_argument("--saida", help="pasta de saída. Padrão: ratex/xselo-<versão>/v1")
    p.add_argument("--epocas", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--acumular", type=int, default=2, help="passos de acumulação de gradiente")
    p.add_argument("--max-tokens", type=int, default=512, help="tamanho máximo de cada conversa em tokens")
    p.add_argument("--rank", type=int, default=16, help="rank do LoRA")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--repetir-dialogos", type=int, default=2,
                   help="quantas vezes os diálogos Pessoa/Xselo escritos à mão aparecem por época")
    p.add_argument("--fracao-validacao", type=float, default=0.05)
    p.add_argument("--max-conversas", type=int, help="usa só N conversas (para testes rápidos)")
    p.add_argument("--tempo-max", type=float, help="para (e salva) depois de N minutos")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--mesclar", action="store_true",
                   help="também salva o modelo completo já mesclado em <saida>/mesclado (grande, fica fora do git)")
    p.add_argument("--exportar-conversas", metavar="ARQ.jsonl", help="salva as conversas geradas do dataset e sai")
    return p.parse_args()


def tokenizar(tok, conversa: list[dict], max_tokens: int, system_prompt: str):
    """Tokeniza a conversa com o chat template do modelo base. Só as respostas do
    Xselo (e o token de fim de turno) entram na loss; system e usuário ficam -100."""
    texto = tok.apply_chat_template(montar_mensagens(conversa, system_prompt), tokenize=False)
    enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    trechos, pos = [], 0
    for msg in conversa:
        ini = texto.find(msg["content"], pos)
        if ini < 0:
            continue
        fim = ini + len(msg["content"])
        pos = fim
        if msg["role"] == "assistant":
            # inclui o marcador de fim de turno logo depois da resposta, pro modelo aprender a parar
            proximo = texto.find("\n", fim)
            trechos.append((ini, proximo if proximo >= 0 else len(texto)))
    labels = [
        t if any(a <= ini_tok < b for a, b in trechos) and fim_tok > ini_tok else -100
        for t, (ini_tok, fim_tok) in zip(ids, offsets)
    ]
    return ids[:max_tokens], labels[:max_tokens]


def lotes(exemplos, batch_size: int, pad_id: int, embaralhar: bool, rnd: random.Random):
    ordem = list(range(len(exemplos)))
    if embaralhar:
        rnd.shuffle(ordem)
    for i in range(0, len(ordem), batch_size):
        grupo = [exemplos[j] for j in ordem[i : i + batch_size]]
        n = max(len(ids) for ids, _ in grupo)
        ids = torch.full((len(grupo), n), pad_id, dtype=torch.long)
        labels = torch.full((len(grupo), n), -100, dtype=torch.long)
        mask = torch.zeros((len(grupo), n), dtype=torch.long)
        for k, (a, b) in enumerate(grupo):
            ids[k, : len(a)] = torch.tensor(a)
            labels[k, : len(b)] = torch.tensor(b)
            mask[k, : len(a)] = 1
        yield ids, labels, mask


def main() -> None:
    args = args_cli()
    rnd = random.Random(args.seed)
    torch.manual_seed(args.seed)

    v = VERSOES[args.versao]
    system = v["system_prompt"]
    args.dados = args.dados or v["dados"]
    args.saida = args.saida or str(pasta_da_versao(args.versao))
    n_mat = v["matematica"] if args.matematica is None else args.matematica

    texto, arquivos = ler_textos(args.dados, RAIZ)
    conversas = construir_conversas_com_origem(texto, seed=args.seed, nome_modelo=v["nome"], geral=v["geral"])
    conversas += conversas_de_matematica(n_mat, seed=args.seed)
    mistura = dict(sorted(Counter(o for _, o in conversas).items()))
    if args.exportar_conversas:
        with open(args.exportar_conversas, "w", encoding="utf-8") as f:
            for c, origem in conversas:
                f.write(json.dumps({"origem": origem, "messages": montar_mensagens(c, system)}, ensure_ascii=False) + "\n")
        print(f"{len(conversas)} conversas salvas em {args.exportar_conversas} {mistura}")
        return

    device = escolher_device(args.device)
    base, sugere_4bit = resolver_base(args.base or v["base"], device)
    quatro_bits = args.quatro_bits or sugere_4bit
    print(f"== Ratex :: {v['nome']} (LoRA) ==")
    print(f"base: {base}{' (4 bits)' if quatro_bits else ''} | device: {device} | torch {torch.__version__}")
    print(f"dados: {len(arquivos)} arquivo(s), {len(texto):,} caracteres -> {len(conversas)} conversas {mistura}")

    from peft import LoraConfig, get_peft_model

    modelo, tok = carregar_base(base, device, quatro_bits=quatro_bits)
    modelo.config.use_cache = False
    if quatro_bits:
        from peft import prepare_model_for_kbit_training

        modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=args.checkpointing)
    elif args.checkpointing:
        modelo.gradient_checkpointing_enable()
        modelo.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    modelo = get_peft_model(modelo, lora)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros: {total / 1e6:.1f}M no total, {treinaveis / 1e6:.2f}M treináveis no LoRA "
          f"({100 * treinaveis / total:.2f}%)")

    # os diálogos escritos à mão (Touhou e assuntos gerais) são o formato-alvo: ganham peso extra
    rnd.shuffle(conversas)
    if args.max_conversas:
        conversas = conversas[: args.max_conversas]
    n_val = max(1, int(len(conversas) * args.fracao_validacao))
    val_conv, treino_conv = conversas[:n_val], conversas[n_val:]
    treino_conv += [(c, o) for c, o in treino_conv if o == "dialogo"] * (args.repetir_dialogos - 1)

    def preparar(lista):
        exemplos = [tokenizar(tok, c, args.max_tokens, system) for c, _ in lista]
        return [(i, l) for i, l in exemplos if any(x != -100 for x in l[1:])]  # descarta se o corte comeu a resposta

    treino, val = preparar(treino_conv), preparar(val_conv)
    n_tokens = sum(len(i) for i, _ in treino)
    print(f"exemplos: treino {len(treino)} | validação {len(val)} | {n_tokens:,} tokens por época")

    otim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    passos_epoca = math.ceil(len(treino) / args.batch_size / args.acumular)
    total_passos = max(1, int(passos_epoca * args.epocas))
    aquecimento = max(1, total_passos // 20)

    def lr_em(passo: int) -> float:
        if passo < aquecimento:
            return args.lr * (passo + 1) / aquecimento
        prog = (passo - aquecimento) / max(1, total_passos - aquecimento)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    usar_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    @torch.no_grad()
    def avaliar() -> float:
        modelo.eval()
        perdas, pesos = 0.0, 0
        for ids, labels, mask in lotes(val, args.batch_size, tok.pad_token_id, False, rnd):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=usar_bf16):
                out = modelo(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            n = int((labels[:, 1:] != -100).sum())
            perdas += out.loss.item() * n
            pesos += n
        modelo.train()
        return perdas / max(pesos, 1)

    saida = Path(args.saida)
    info: dict = {}

    def salvar(motivo: str) -> None:
        saida.mkdir(parents=True, exist_ok=True)
        modelo.save_pretrained(saida)
        tok.save_pretrained(saida)
        cfg = {
            "nome": v["nome"],
            "versao": args.versao,
            "tipo": "lora",
            "base": base,
            "quatro_bits": quatro_bits,
            "system_prompt": system,
            "geracao": GERACAO_HIBRIDO,
            "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout,
                     "target_modules": "all-linear", "parametros_treinaveis": treinaveis},
            "treino": info,
        }
        (saida / ARQ_RATEX).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"   -> adaptador salvo em {saida} ({motivo})")

    perda_inicial = avaliar()
    print(f"loss de validação antes do LoRA: {perda_inicial:.4f}\n")
    melhor = float("inf")
    passo, t0 = 0, time.time()
    modelo.train()
    epoca = 0
    parar = False
    while passo < total_passos and not parar:
        epoca += 1
        acumulado, n_micro, t_log = 0.0, 0, time.time()
        for k, (ids, labels, mask) in enumerate(lotes(treino, args.batch_size, tok.pad_token_id, True, rnd)):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=usar_bf16):
                out = modelo(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            (out.loss / args.acumular).backward()
            acumulado += out.loss.item()
            n_micro += 1
            if (k + 1) % args.acumular:
                continue
            for g in otim.param_groups:
                g["lr"] = lr_em(passo)
            torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            otim.step()
            otim.zero_grad(set_to_none=True)
            passo += 1
            if passo % 5 == 0 or passo == total_passos:
                seg = (time.time() - t_log) / 5
                t_log = time.time()
                falta = (total_passos - passo) * seg
                print(f"passo {passo:>4}/{total_passos} | época {epoca} | loss {acumulado / n_micro:.4f} | "
                      f"lr {lr_em(passo):.2e} | {seg:.1f} s/passo | falta ~{falta / 60:.0f} min", flush=True)
                acumulado, n_micro = 0.0, 0
            if passo >= total_passos:
                break
            if args.tempo_max and time.time() - t0 > args.tempo_max * 60:
                print("tempo máximo atingido")
                parar = True
                break
        perda = avaliar()
        marca = "  <- melhor até agora" if perda < melhor else ""
        print(f"== fim da época {epoca}: loss de validação {perda:.4f}{marca}")
        if perda < melhor:
            melhor = perda
            info.update({
                "epocas": epoca, "passos": passo, "loss_validacao": round(perda, 4),
                "loss_validacao_antes": round(perda_inicial, 4), "conversas": len(conversas), "mistura": mistura,
                "exemplos_treino": len(treino), "tokens_por_epoca": n_tokens, "lr": args.lr,
                "batch_efetivo": args.batch_size * args.acumular, "max_tokens": args.max_tokens,
                "device": device, "minutos": round((time.time() - t0) / 60, 1),
                "arquivos_dataset": [str(a.relative_to(RAIZ)) if a.is_relative_to(RAIZ) else str(a) for a in arquivos],
                "data": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            salvar(f"época {epoca}")

    print(f"\ntreino concluído em {(time.time() - t0) / 60:.1f} min | loss de validação "
          f"{perda_inicial:.4f} -> {melhor:.4f}")
    if args.mesclar:
        from peft import PeftModel

        modelo_base, _ = carregar_base(base, device)
        mesclado = PeftModel.from_pretrained(modelo_base, saida).merge_and_unload()
        mesclado.save_pretrained(saida / "mesclado")
        tok.save_pretrained(saida / "mesclado")
        print(f"modelo completo mesclado salvo em {saida / 'mesclado'}")

    modelo.eval()
    modelo.config.use_cache = True
    perguntas = ["o que é touhou?", "quem é a Cirno?"]
    if v["geral"]:
        perguntas += ["por que o céu é azul?", "quanto é 15% de 240?"]
    for pergunta in perguntas:
        print(f"\nvocê> {pergunta}\nxselo> ", end="", flush=True)
        responder(modelo, tok, [{"role": "user", "content": pergunta}], system_prompt=system, **GERACAO_HIBRIDO)
    print("\nagora é só rodar:  python gerar.py --chat")


if __name__ == "__main__":
    sys.exit(main())
