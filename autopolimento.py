#!/usr/bin/env python3
"""
Autopolimento: o Xselo treina com as próprias respostas, sem ninguém precisar digitar nada.

Pra cada pedido (dados_autopolimento/pedidos.txt + perguntas do próprio dataset):
    1. o Xselo responde várias vezes (padrão: 4), com um pouco de sorteio, pra sair variado;
    2. um juiz lê todas e escolhe a MELHOR e a PIOR. Faz isso duas vezes, com as respostas em
       ordens diferentes, e só vale quando as duas leituras concordam (juiz não favorece quem
       aparece primeiro);
    3. a melhor vira a resposta "Boa" e a pior, a "Ruim": um par de preferência pro DPO.

Os pares vão pra dados_preferencia/auto.txt (dá pra abrir e revisar), e o treinar_dpo.py usa
junto com os pares escritos à mão e os da arena. É o jeito de ter centenas de pares sem jogar
centenas de partidas na arena.

Juiz padrão: o próprio modelo base com o adaptador do Xselo desligado (não gasta memória a
mais). Com --juiz <modelo do Hugging Face>, o Xselo é descarregado depois de gerar e o juiz
entra no lugar dele, então cabe na mesma GPU.

Dá pra parar e continuar: o progresso fica num .json (--cache), e rodar de novo retoma de onde
parou.

    python autopolimento.py --modelo /content/saida/xselo-0-5-gemma31b
    python autopolimento.py --modelo ratex/xselo-0-3/v1 --pedidos 20 --amostras 4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent

JUIZ_SYSTEM = "Você é um avaliador rigoroso, justo e direto. Avalia respostas em português do Brasil."

JUIZ_LISTA = """Abaixo estão {n} respostas do mesmo assistente para o mesmo pedido. Escolha a MELHOR e a PIOR.

Pedido: {pedido}

{respostas}

Critérios, nesta ordem de importância:
1. fatos certos, nada inventado;
2. cumprir o pedido e todas as restrições dele (tamanho, formato, público, palavras proibidas);
3. escrita viva e natural em português do Brasil: imagens concretas, ritmo variado, sem clichê, sem frase feita, sem enrolação e sem jeito de robô;
4. tamanho adequado ao pedido. Não prefira uma resposta só porque ela é mais longa.

Escreva no máximo duas frases de justificativa e, na última linha, só: MELHOR: <número> | PIOR: <número>"""


def _normal(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\W+", " ", t).strip()


def perguntas_das_provas() -> set[str]:
    """Tudo que cai nas provas: fica de fora do autopolimento, senão a prova mede decoreba."""
    from avaliar import PROSA as PROSA_FIXA
    from avaliar import PROVA
    from prova_dificil import CONTAS, CORRECOES, PEGADINHAS, PROSA

    qs = [q for cat in PROVA.values() for q, _ in cat] + list(PROSA_FIXA) + list(PROSA)
    qs += [q for q, _ in PEGADINHAS] + [q for q, _ in CONTAS] + [c[-1][1] for c, _ in CORRECOES]
    return {_normal(q) for q in qs}


def pedidos_novos(raiz: Path = RAIZ) -> list[str]:
    linhas = []
    for arq in sorted((raiz / "dados_autopolimento").glob("*.txt")):
        linhas += [l.strip() for l in arq.read_text(encoding="utf-8").splitlines()]
    return [l for l in linhas if l and not l.startswith("#")]


def pedidos_do_dataset(raiz: Path = RAIZ) -> list[str]:
    """A primeira fala da pessoa em cada conversa escrita à mão (prosa e assuntos gerais)."""
    pedidos = []
    for pasta in ("dados_prosa", "dados_gerais"):
        for arq in sorted((raiz / pasta).glob("*.txt")):
            texto = arq.read_text(encoding="utf-8")
            pedidos += re.findall(r"(?:^|\n\n)Pessoa:\s*(.+)", texto)
    return pedidos


def escolher_pedidos(n: int, seed: int, raiz: Path = RAIZ) -> list[str]:
    """~2/3 pedidos novos (dados_autopolimento/) e o resto do dataset, sem nada das provas."""
    proibidos = perguntas_das_provas()
    rnd = random.Random(seed)
    vistos: set[str] = set()

    def limpos(lista):
        saida = []
        for p in lista:
            chave = _normal(p)
            if chave and chave not in proibidos and chave not in vistos:
                vistos.add(chave)
                saida.append(p)
        return saida

    novos, antigos = limpos(pedidos_novos(raiz)), limpos(pedidos_do_dataset(raiz))
    rnd.shuffle(novos)
    rnd.shuffle(antigos)
    n_novos = min(len(novos), math.ceil(n * 2 / 3))
    escolhidos = novos[:n_novos] + antigos[: n - n_novos]
    if len(escolhidos) < n:  # faltou do dataset: completa com o resto dos novos
        escolhidos += novos[n_novos: n_novos + n - len(escolhidos)]
    rnd.shuffle(escolhidos)
    return escolhidos


def veredito(texto: str) -> tuple[int, int] | None:
    """(melhor, pior) numerados a partir de 1, ou None se o juiz não respondeu no formato."""
    melhor = re.findall(r"MELHOR\D{0,6}(\d+)", texto, flags=re.I)
    pior = re.findall(r"PIOR\D{0,6}(\d+)", texto, flags=re.I)
    if not melhor or not pior:
        return None
    return int(melhor[-1]), int(pior[-1])


def montar_julgamento(pedido: str, respostas: list[str], ordem: list[int]) -> str:
    blocos = "\n\n".join(f'Resposta {i}:\n"""{respostas[j]}"""' for i, j in enumerate(ordem, 1))
    return JUIZ_LISTA.format(n=len(ordem), pedido=pedido, respostas=blocos)


def decidir(pedido: str, respostas: list[str], julgar, rnd: random.Random) -> dict:
    """Duas leituras do juiz com as respostas em ordens diferentes; só vale se concordarem."""
    ordem1 = list(range(len(respostas)))
    rnd.shuffle(ordem1)
    ordem2 = ordem1[::-1]  # todo mundo muda de lugar
    resultado = {"ordens": [ordem1, ordem2], "vereditos": [], "melhor": None, "pior": None}
    escolhas = []
    for ordem in (ordem1, ordem2):
        texto = julgar(montar_julgamento(pedido, respostas, ordem))
        resultado["vereditos"].append(texto)
        v = veredito(texto)
        if v is None or not all(1 <= x <= len(ordem) for x in v):
            return resultado
        escolhas.append((ordem[v[0] - 1], ordem[v[1] - 1]))
    (m1, p1), (m2, p2) = escolhas
    if m1 == m2 and p1 == p2 and m1 != p1:
        resultado["melhor"], resultado["pior"] = m1, p1
    return resultado


def main() -> None:
    p = argparse.ArgumentParser(description="Autopolimento do Xselo: pares de preferência gerados sozinhos.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--modelo", required=True, help="pasta do Xselo treinado (adaptador LoRA + ratex_config.json)")
    p.add_argument("--pedidos", type=int, default=120, help="quantos pedidos usar")
    p.add_argument("--amostras", type=int, default=4, help="respostas por pedido")
    p.add_argument("--juiz", help="outro modelo do Hugging Face pra julgar (padrão: a base sem o adaptador)")
    p.add_argument("--saida", default=str(RAIZ / "dados_preferencia" / "auto.txt"), help="arquivo dos pares")
    p.add_argument("--cache", help="progresso salvo (pra continuar depois). Padrão: <modelo>/autopolimento.json")
    p.add_argument("--max-tokens", type=int, default=500, help="tamanho máximo de cada resposta")
    p.add_argument("--temperatura", type=float, default=0.9)
    p.add_argument("--memoria", choices=["sim", "nao"], default="sim", help="consulta o dataset, como no chat")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rnd = random.Random(args.seed)

    from nucleo.hibrido import (bitsandbytes_disponivel, carregar_base, carregar_hibrido, escolher_device,
                                responder, responder_varias)
    from nucleo.preferencias import _cabe_no_formato, escrever_pares

    pasta = Path(args.modelo)
    cache_arq = Path(args.cache) if args.cache else pasta / "autopolimento.json"
    cache = json.loads(cache_arq.read_text(encoding="utf-8")) if cache_arq.exists() else {}
    pedidos = cache.get("pedidos") or escolher_pedidos(args.pedidos, args.seed)
    cache.update({"modelo": str(pasta), "pedidos": pedidos})
    amostras: dict = cache.setdefault("amostras", {})
    julgamentos: dict = cache.setdefault("julgamentos", {})

    def salvar():
        cache_arq.parent.mkdir(parents=True, exist_ok=True)
        cache_arq.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    device = escolher_device(args.device)
    faltam = [x for x in pedidos if x not in amostras]
    print(f"== Autopolimento :: {pasta} ==")
    print(f"{len(pedidos)} pedidos x {args.amostras} respostas | já gerados: {len(pedidos) - len(faltam)} "
          f"| já julgados: {len(julgamentos)} | progresso em {cache_arq}")

    cfg = json.loads((pasta / "ratex_config.json").read_text(encoding="utf-8"))
    pendentes = [x for x in pedidos if x not in julgamentos]
    modelo = None
    if faltam or (pendentes and not args.juiz):  # já tudo gerado e juiz de fora: nem carrega o Xselo
        modelo, tok, cfg = carregar_hibrido(pasta, device=device)
    memoria = None
    if faltam and args.memoria == "sim":
        from nucleo.memoria import Memoria

        memoria = Memoria()

    # 1) o Xselo responde cada pedido várias vezes
    t0 = time.time()
    for i, pedido in enumerate(faltam, 1):
        respostas = responder_varias(modelo, tok, [{"role": "user", "content": pedido}], args.amostras,
                                     system_prompt=cfg["system_prompt"], max_novos_tokens=args.max_tokens,
                                     temperatura=args.temperatura, memoria=memoria)
        amostras[pedido] = [[texto, terminou] for texto, terminou in respostas]
        salvar()
        seg = (time.time() - t0) / i
        print(f"gerando {i}/{len(faltam)} | {seg:.0f} s por pedido | falta ~{seg * (len(faltam) - i) / 60:.0f} min",
              flush=True)

    # 2) o juiz escolhe a melhor e a pior
    if args.juiz and pendentes:
        del modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        quatro_bits = device == "cuda" and bitsandbytes_disponivel()
        print(f"\ncarregando o juiz {args.juiz}{' (4 bits)' if quatro_bits else ''}...")
        juiz, tok_juiz = carregar_base(args.juiz, device, quatro_bits=quatro_bits)
        juiz.eval()

        def julgar(texto):
            return responder(juiz, tok_juiz, [{"role": "user", "content": texto}], system_prompt=JUIZ_SYSTEM,
                             stream=False, temperatura=0, max_novos_tokens=200, penalidade_repeticao=1.0)
    else:
        def julgar(texto):
            with modelo.disable_adapter():
                return responder(modelo, tok, [{"role": "user", "content": texto}], system_prompt=JUIZ_SYSTEM,
                                 stream=False, temperatura=0, max_novos_tokens=200, penalidade_repeticao=1.0)

    t0 = time.time()
    for i, pedido in enumerate(pendentes, 1):
        # só entram respostas inteiras (não cortadas no limite) e que cabem no formato do arquivo
        validas = [t for t, terminou in amostras[pedido] if terminou and t.strip() and _cabe_no_formato(t)]
        validas = list(dict.fromkeys(validas))
        if len(validas) < 2:
            julgamentos[pedido] = {"respostas": validas, "melhor": None, "pior": None, "motivo": "poucas respostas"}
        else:
            julgamentos[pedido] = {"respostas": validas, **decidir(pedido, validas, julgar, rnd)}
        salvar()
        if i % 5 == 0 or i == len(pendentes):
            seg = (time.time() - t0) / i
            print(f"julgando {i}/{len(pendentes)} | {seg:.0f} s por pedido | "
                  f"falta ~{seg * (len(pendentes) - i) / 60:.0f} min", flush=True)

    # 3) melhor x pior vira par de preferência
    pares = []
    for pedido in pedidos:
        j = julgamentos.get(pedido, {})
        if j.get("melhor") is None or not _cabe_no_formato(pedido):
            continue
        pares.append({"conversa": [{"role": "user", "content": pedido}],
                      "boa": j["respostas"][j["melhor"]], "ruim": j["respostas"][j["pior"]]})
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(escrever_pares(pares, f"Autopolimento: melhor x pior de {args.amostras} respostas do "
                                           f"{cfg['nome']} ({len(pares)} pares)"), encoding="utf-8")
    sem_acordo = sum(1 for j in julgamentos.values() if j.get("melhor") is None and j.get("motivo") is None)
    print(f"\n{len(pares)} pares gravados em {saida}")
    print(f"(de {len(pedidos)} pedidos: {sem_acordo} sem acordo entre as duas leituras do juiz, "
          f"{sum(1 for j in julgamentos.values() if j.get('motivo'))} com poucas respostas inteiras)")
    if pares:
        tam = sum(len(x["boa"].split()) for x in pares) / len(pares), sum(len(x["ruim"].split()) for x in pares) / len(pares)
        print(f"tamanho médio: boa {tam[0]:.0f} palavras, ruim {tam[1]:.0f} palavras "
              "(se a boa for sempre bem maior, o juiz está premiando tamanho)")


if __name__ == "__main__":
    sys.exit(main())
