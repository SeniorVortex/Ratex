"""
Arena às cegas: você faz uma pergunta, dois competidores respondem como "A" e "B" (sorteados,
sem mostrar quem é quem), você escolhe a melhor, e cada escolha atualiza uma nota Elo,
igual ranking de xadrez. Com o tempo, o placar mostra quem é melhor de verdade no seu gosto.

Os competidores podem dividir o mesmo modelo base na memória: o Xselo (com o LoRA), o
Xselo antes do polimento (outro LoRA) e o Qwen puro (LoRA desligado). Todos usam o mesmo
system prompt, a mesma memória e a mesma busca, então a única diferença é o treino.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ELO_INICIAL = 1000.0
K = 24.0


def esperado(ra: float, rb: float) -> float:
    """Chance esperada de A vencer B pelo Elo."""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


class Arena:
    def __init__(self, arquivo: str | Path | None = None):
        self.arquivo = Path(arquivo) if arquivo else None
        self.notas: dict[str, float] = {}
        self.partidas: list[dict] = []
        if self.arquivo and self.arquivo.exists():
            dados = json.loads(self.arquivo.read_text(encoding="utf-8"))
            self.notas, self.partidas = dados.get("notas", {}), dados.get("partidas", [])

    def registrar(self, a: str, b: str, resultado: str, pergunta: str = "", respostas: tuple = ("", "")) -> None:
        """resultado: 'a', 'b' ou 'empate'."""
        ra, rb = self.notas.setdefault(a, ELO_INICIAL), self.notas.setdefault(b, ELO_INICIAL)
        placar_a = {"a": 1.0, "b": 0.0, "empate": 0.5}[resultado]
        ea = esperado(ra, rb)
        self.notas[a] = ra + K * (placar_a - ea)
        self.notas[b] = rb + K * ((1 - placar_a) - (1 - ea))
        self.partidas.append({"a": a, "b": b, "resultado": resultado, "pergunta": pergunta,
                              "resposta_a": respostas[0], "resposta_b": respostas[1],
                              "data": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        self.salvar()

    def salvar(self) -> None:
        if self.arquivo:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            self.arquivo.write_text(json.dumps({"notas": self.notas, "partidas": self.partidas},
                                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def placar(self) -> str:
        if not self.notas:
            return "(nenhuma partida ainda)"
        vitorias: dict[str, list[int]] = {n: [0, 0] for n in self.notas}  # [vitórias, partidas]
        for p in self.partidas:
            for lado, nome in (("a", p["a"]), ("b", p["b"])):
                vitorias.setdefault(nome, [0, 0])[1] += 1
                if p["resultado"] == lado:
                    vitorias[nome][0] += 1
        linhas = [f"{'#':>2}  {'competidor':<28} {'Elo':>6}  vitórias"]
        for i, (nome, nota) in enumerate(sorted(self.notas.items(), key=lambda x: -x[1]), 1):
            v, n = vitorias.get(nome, [0, 0])
            linhas.append(f"{i:>2}  {nome:<28} {nota:>6.0f}  {v}/{n}")
        return "\n".join(linhas)

    def rodar(self, competidores: dict[str, Callable[[list[dict]], str]], entrada=input) -> None:
        """Loop interativo. competidores: {nome: função(conversa) -> resposta}."""
        nomes = list(competidores)
        if len(nomes) < 2:
            raise ValueError("a arena precisa de pelo menos 2 competidores")
        print(f"🏆 Arena às cegas com {len(nomes)} competidores. Digite uma pergunta ou pedido.")
        print("   Comandos: /placar (ranking), /sair (parar).\n")
        while True:
            pergunta = entrada("você> ").strip()
            if pergunta in ("", "/sair", "sair"):
                break
            if pergunta == "/placar":
                print(self.placar() + "\n")
                continue
            a, b = random.sample(nomes, 2)
            conversa = [{"role": "user", "content": pergunta}]
            resp_a, resp_b = competidores[a](conversa), competidores[b](conversa)
            print(f"\n──── A ────\n{resp_a}\n\n──── B ────\n{resp_b}\n")
            escolha = ""
            while escolha not in ("a", "b", "e"):
                escolha = entrada("qual foi melhor? [a / b / e = empate] ").strip().lower()[:1]
            resultado = {"a": "a", "b": "b", "e": "empate"}[escolha]
            self.registrar(a, b, resultado, pergunta, (resp_a, resp_b))
            vencedor = {"a": a, "b": b}.get(resultado, "empate")
            print(f"   A era {a}, B era {b}. {'Empate!' if vencedor == 'empate' else f'Ponto pro {vencedor}!'}\n")
        print("\n" + self.placar())


def competidores_do_xselo(modelo, tok, cfg: dict, memoria=None, busca=None,
                          outros_adaptadores: dict[str, str] | None = None, max_novos_tokens: int = 500) -> dict:
    """Competidores que dividem o mesmo modelo base: o Xselo carregado, outros adaptadores
    (ex.: {"Xselo sem polimento": "/pasta"}) e o Qwen puro (adaptador desligado)."""
    from .hibrido import responder

    ger = dict(cfg.get("geracao", {}), max_novos_tokens=max_novos_tokens)
    system = cfg["system_prompt"]
    ativo = modelo.active_adapter if isinstance(modelo.active_adapter, str) else "default"
    nome_xselo = f"Xselo {cfg.get('versao', '')}" + (" polido" if cfg.get("dpo") else "")

    def com_adaptador(nome_adaptador):
        def responder_com(conversa):
            modelo.set_adapter(nome_adaptador)
            return responder(modelo, tok, conversa, system_prompt=system, memoria=memoria, busca=busca,
                             stream=False, **ger)
        return responder_com

    def qwen_puro(conversa):
        with modelo.disable_adapter():
            return responder(modelo, tok, conversa, system_prompt=system, memoria=memoria, busca=busca,
                             stream=False, **ger)

    competidores = {nome_xselo: com_adaptador(ativo)}
    for i, (nome, pasta) in enumerate((outros_adaptadores or {}).items()):
        apelido = f"extra{i}"
        if Path(pasta, "adapter_model.safetensors").exists():
            modelo.load_adapter(pasta, adapter_name=apelido)
            competidores[nome] = com_adaptador(apelido)
    modelo.set_adapter(ativo)
    competidores[f"Qwen puro ({Path(cfg['base']).name})"] = qwen_puro
    return competidores
