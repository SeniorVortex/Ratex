#!/usr/bin/env python3
"""
Polimento por preferência (DPO) de um Xselo já treinado.

Pega o adaptador LoRA de uma versão (ex.: ratex/xselo-0-4/v1) e os pares de
dados_preferencia/ ("nessa conversa, a resposta Boa é melhor que a Ruim") e ensina o
modelo a preferir as respostas boas: parar de inventar, aceitar correção quando a pessoa
está certa, não concordar com informação errada, fugir de texto de robô.

Como funciona, sem matemática: pra cada par, o script mede o quanto o modelo "acha
provável" a resposta boa e a ruim. O treino empurra a boa pra cima e a ruim pra baixo,
sempre comparando com o modelo de antes do polimento, pra ele não se afastar demais do
que já sabia. Um pouco de treino comum na resposta boa (--sft-peso) mantém a escrita
natural.

Uso:
    python treinar_dpo.py --modelo ratex/xselo-0-4/v1            # salva em ratex/xselo-0-4/v1-dpo
    python treinar_dpo.py --modelo /content/saida/xselo-0-4-qwen32b --saida /content/saida/xselo-0-4-qwen32b-dpo
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from nucleo.hibrido import ARQ_RATEX, GERACAO_HIBRIDO, carregar_base, escolher_device, montar_mensagens, responder
from nucleo.preferencias import ler_pares

RAIZ = Path(__file__).resolve().parent


def args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polimento por preferência (DPO) do Xselo.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--modelo", required=True, help="pasta do Xselo já treinado (adaptador LoRA + ratex_config.json)")
    p.add_argument("--dados", nargs="+", default=["dados_preferencia"], help="arquivos/pastas com os pares")
    p.add_argument("--saida", help="pasta do modelo polido. Padrão: <modelo>-dpo")
    p.add_argument("--base", help="modelo base alternativo (ex.: pasta local já baixada)")
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.1,
                   help="força do DPO: maior = se afasta menos do modelo original")
    p.add_argument("--sft-peso", type=float, default=0.2,
                   help="peso do treino comum na resposta boa (mantém a escrita natural)")
    p.add_argument("--acumular", type=int, default=4, help="pares por passo do otimizador")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--checkpointing", action="store_true", help="menos memória, mais lento (bom pro 32B)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def tokenizar_par(tok, system: str, conversa: list[dict], resposta: str, max_tokens: int):
    """(ids, n_prompt): a conversa + a resposta, e onde a resposta começa."""
    prompt = tok.apply_chat_template(montar_mensagens(conversa, system), tokenize=False, add_generation_prompt=True)
    completo = tok.apply_chat_template(montar_mensagens(conversa + [{"role": "assistant", "content": resposta}], system),
                                       tokenize=False)
    if not completo.startswith(prompt):
        raise ValueError("o chat template não produz o prompt como prefixo da conversa completa")
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_resposta = tok(completo[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (ids_prompt + ids_resposta)[-max_tokens:]
    n_prompt = max(1, len(ids) - len(ids_resposta))
    return torch.tensor(ids), n_prompt


def logp_resposta(modelo, ids: torch.Tensor, n_prompt: int, autocast) -> tuple[torch.Tensor, int]:
    """Soma do log da probabilidade de cada token da resposta, e quantos tokens ela tem."""
    with autocast:
        logits = modelo(input_ids=ids[None]).logits[0, :-1]
    alvo = ids[1:]
    lp = -F.cross_entropy(logits.float(), alvo, reduction="none")
    return lp[n_prompt - 1:].sum(), len(alvo) - (n_prompt - 1)


def main() -> None:
    args = args_cli()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pasta = Path(args.modelo)
    if not (pasta / ARQ_RATEX).exists():
        sys.exit(f"{pasta} não tem um Xselo treinado (falta {ARQ_RATEX}).")
    cfg = json.loads((pasta / ARQ_RATEX).read_text(encoding="utf-8"))
    saida = Path(args.saida) if args.saida else pasta.with_name(pasta.name + "-dpo")
    system = cfg["system_prompt"]
    base = args.base or cfg["base"]

    pares, arquivos = ler_pares(args.dados, RAIZ)
    if not pares:
        sys.exit(f"nenhum par Pessoa/Ruim/Boa encontrado em {args.dados}")

    device = escolher_device(args.device)
    tamanho = re.search(r"(\d+(?:\.\d+)?)B", base)
    if device == "cpu" and tamanho and float(tamanho.group(1)) >= 7:
        sys.exit(f"{base} é grande demais pra polir na CPU: use o notebook do Colab.")
    quatro_bits = bool(cfg.get("quatro_bits")) and device == "cuda"
    print(f"== Ratex :: DPO do {cfg['nome']} ==")
    print(f"adaptador: {pasta} | base: {base}{' (4 bits)' if quatro_bits else ''} | device: {device}")
    print(f"pares: {len(pares)} de {len(arquivos)} arquivo(s) | épocas {args.epocas} | lr {args.lr} | beta {args.beta}")

    from peft import PeftModel
    from transformers import AutoTokenizer

    modelo, _ = carregar_base(base, device, quatro_bits=quatro_bits)
    tok = AutoTokenizer.from_pretrained(pasta)
    modelo.config.use_cache = False
    if quatro_bits:
        from peft import prepare_model_for_kbit_training

        modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=args.checkpointing)
    elif args.checkpointing:
        modelo.gradient_checkpointing_enable()
        modelo.enable_input_require_grads()
    modelo = PeftModel.from_pretrained(modelo, pasta, is_trainable=True)
    if not quatro_bits:
        modelo = modelo.to(device)
    usar_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=usar_bf16)

    exemplos = []
    for par in pares:
        boa = tokenizar_par(tok, system, par["conversa"], par["boa"], args.max_tokens)
        ruim = tokenizar_par(tok, system, par["conversa"], par["ruim"], args.max_tokens)
        exemplos.append({"boa": (boa[0].to(device), boa[1]), "ruim": (ruim[0].to(device), ruim[1])})

    # referência: o modelo antes do polimento (calculada uma vez só, antes de mexer nos pesos)
    modelo.eval()
    with torch.no_grad():
        for ex in exemplos:
            ex["ref_boa"] = logp_resposta(modelo, *ex["boa"], autocast)[0].item()
            ex["ref_ruim"] = logp_resposta(modelo, *ex["ruim"], autocast)[0].item()
    prefere_antes = sum(ex["ref_boa"] / ex["boa"][0].numel() > ex["ref_ruim"] / ex["ruim"][0].numel()
                        for ex in exemplos)
    print(f"antes do polimento, o modelo já preferia a resposta boa em {prefere_antes}/{len(exemplos)} pares "
          f"(comparando a probabilidade média por token)\n")

    treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    otim = torch.optim.AdamW(treinaveis, lr=args.lr, weight_decay=0.0)
    total_passos = max(1, math.ceil(len(exemplos) / args.acumular) * args.epocas)
    passo, t0 = 0, time.time()
    modelo.train()
    historico = []
    for epoca in range(1, args.epocas + 1):
        random.shuffle(exemplos)
        soma_loss, acertos, margens = 0.0, 0, []
        otim.zero_grad(set_to_none=True)
        for k, ex in enumerate(exemplos, 1):
            # 1) a margem atual, sem guardar nada pra gradiente
            with torch.no_grad():
                lp_boa0 = logp_resposta(modelo, *ex["boa"], autocast)[0].item()
                lp_ruim0 = logp_resposta(modelo, *ex["ruim"], autocast)[0].item()
            margem = (lp_boa0 - ex["ref_boa"]) - (lp_ruim0 - ex["ref_ruim"])
            # 2) a derivada do DPO, -logsigmoid(beta * margem), só depende da margem por esse fator;
            #    então dá pra fazer o backward de uma resposta de cada vez (metade da memória),
            #    com o mesmo gradiente de fazer as duas juntas
            fator = args.beta * torch.sigmoid(torch.tensor(-args.beta * margem)).item()
            lp_boa, n_boa = logp_resposta(modelo, *ex["boa"], autocast)
            ((-fator * lp_boa + args.sft_peso * (-lp_boa / n_boa)) / args.acumular).backward()
            del lp_boa
            lp_ruim, _ = logp_resposta(modelo, *ex["ruim"], autocast)
            ((fator * lp_ruim) / args.acumular).backward()
            del lp_ruim
            loss = -F.logsigmoid(torch.tensor(args.beta * margem)).item() + args.sft_peso * (-lp_boa0 / n_boa)
            soma_loss += loss
            acertos += int(margem > 0)
            margens.append(margem)
            if k % args.acumular == 0 or k == len(exemplos):
                for g in otim.param_groups:  # aquecimento curto e depois constante
                    g["lr"] = args.lr * min(1.0, (passo + 1) / max(1, total_passos // 10))
                torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
                otim.step()
                otim.zero_grad(set_to_none=True)
                passo += 1
        info = {"epoca": epoca, "loss": round(soma_loss / len(exemplos), 4),
                "prefere_boa": f"{acertos}/{len(exemplos)}", "margem_media": round(sum(margens) / len(margens), 3)}
        historico.append(info)
        print(f"época {epoca}: loss {info['loss']:.4f} | prefere a boa (mais que antes) em {info['prefere_boa']} "
              f"| margem média {info['margem_media']:+.2f} | {(time.time() - t0) / 60:.1f} min")

    saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(saida)
    tok.save_pretrained(saida)
    cfg["dpo"] = {
        "de": str(pasta), "pares": len(pares), "epocas": args.epocas, "lr": args.lr, "beta": args.beta,
        "sft_peso": args.sft_peso, "historico": historico,
        "arquivos": [str(a.relative_to(RAIZ)) if a.is_relative_to(RAIZ) else str(a) for a in arquivos],
        "data": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (saida / ARQ_RATEX).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = pasta / "README.md"
    if card.exists():
        shutil.copy(card, saida / "README.md")
        with open(saida / "README.md", "a", encoding="utf-8") as f:
            f.write(f"\n## Polimento por preferência (DPO)\n\n{len(pares)} pares de `dados_preferencia/`, "
                    f"{args.epocas} épocas, beta {args.beta}. Última época: a resposta boa ficou mais provável "
                    f"que antes em {historico[-1]['prefere_boa']} pares.\n")
    print(f"\n-> modelo polido salvo em {saida}")

    modelo.eval()
    modelo.config.use_cache = True
    for par in pares[:3]:
        print(f"\nvocê> {par['conversa'][-1]['content']}\nxselo> ", end="", flush=True)
        responder(modelo, tok, par["conversa"], system_prompt=system, **dict(GERACAO_HIBRIDO, temperatura=0))


if __name__ == "__main__":
    main()
