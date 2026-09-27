"""
Ajudantes pra rodar o Xselo em máquinas emprestadas (Kaggle), onde os arquivos entram como
"inputs" montados em /kaggle/input e ninguém quer ficar digitando caminho.

    achar_xselo("/kaggle/input")               # pasta do adaptador (tem ratex_config.json)
    achar_base("/kaggle/input", "google/gemma-4-31B-it")   # a Gemma já anexada, se tiver
"""

from __future__ import annotations

import json
from pathlib import Path

from .hibrido import ARQ_RATEX


def achar_xselo(raiz: str | Path) -> Path | None:
    """A pasta do Xselo treinado (adaptador + ratex_config.json) mais nova dentro de `raiz`.
    Serve pro .zip que o Colab baixa: o Kaggle descompacta sozinho quando vira dataset."""
    achados = [p.parent for p in Path(raiz).rglob(ARQ_RATEX) if (p.parent / "adapter_config.json").exists()]
    if not achados:
        return None
    # prefere a versão polida com mais pares (o nome tem "-polido-95pares"), depois a mais recente
    def chave(pasta: Path):
        nome = pasta.name
        pares = int(nome.split("-polido-")[1].split("pares")[0]) if "-polido-" in nome else (1 if "dpo" in nome else 0)
        return pares, pasta.stat().st_mtime
    return max(achados, key=chave)


def achar_base(raiz: str | Path, base: str) -> Path | None:
    """A pasta do modelo base já anexada (ex.: Gemma 4 dos "Models" do Kaggle), pra não baixar
    60 GB do Hugging Face. Confere o tipo do modelo e o tamanho pelo nome da pasta."""
    nome = base.split("/")[-1].lower()                 # gemma-4-31b-it
    tamanho = next((t for t in nome.split("-") if t.endswith("b") and t[:-1].replace(".", "").isdigit()), "")
    candidatos = []
    for cfg in Path(raiz).rglob("config.json"):
        pasta = cfg.parent
        if not any(pasta.glob("*.safetensors")):
            continue
        try:
            tipo = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "")
        except (OSError, ValueError):
            continue
        caminho = str(pasta).lower()
        familia = nome.split("-")[0]                   # gemma, qwen3.8...
        if familia.rstrip("0123456789.") in tipo and (not tamanho or tamanho in caminho):
            candidatos.append(pasta)
    return min(candidatos, key=lambda p: len(str(p))) if candidatos else None
