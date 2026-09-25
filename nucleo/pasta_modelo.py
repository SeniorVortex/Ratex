"""
Salvar/carregar o modelo no formato de pasta estilo Hugging Face:

    ratex/xselo-0-1/v1/
      pytorch_model.bin        pesos (state_dict do PyTorch)
      config.json              hiperparâmetros da rede + metadados do treino
      vocab.json               tokenizador (tipo, tokens e merges do BPE)
      generation_config.json   parâmetros padrão de amostragem do gerar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .modelo import ConfigXselo, XseloGPT
from .tokenizador import carregar_tokenizador, salvar_tokenizador

NOME_MODELO = "ratex/xselo-0-1/v1"
PASTA_PADRAO = Path(__file__).resolve().parent.parent / NOME_MODELO

ARQ_PESOS = "pytorch_model.bin"
ARQ_CONFIG = "config.json"
ARQ_VOCAB = "vocab.json"
ARQ_GERACAO = "generation_config.json"

GERACAO_PADRAO = {
    "temperatura": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "penalidade_repeticao": 1.0,
    "janela_repeticao": 64,
    "max_novos_tokens": 500,
}


def _escrever_json(caminho: Path, dados: dict) -> None:
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def salvar_modelo(pasta, modelo: XseloGPT, tokenizador, extra: dict | None = None, fp16: bool = False) -> None:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    estado = {k: v.detach().cpu() for k, v in modelo.state_dict().items()}
    estado.pop("lm_head.weight", None)  # amarrado ao tok_emb; recriado ao carregar
    if fp16:
        estado = {k: v.half() if v.is_floating_point() else v for k, v in estado.items()}
    torch.save(estado, pasta / ARQ_PESOS)

    config = {
        "nome": NOME_MODELO,
        "model_type": "xselo-gpt",
        "architectures": ["XseloGPT"],
        **modelo.cfg.para_dict(),
        "tokenizador": tokenizador.tipo,
        "n_parametros": modelo.n_parametros(),
        "dtype_pesos": "float16" if fp16 else "float32",
    }
    if extra:
        config.update(extra)
    _escrever_json(pasta / ARQ_CONFIG, config)
    salvar_tokenizador(tokenizador, pasta / ARQ_VOCAB)
    if not (pasta / ARQ_GERACAO).exists():
        _escrever_json(pasta / ARQ_GERACAO, GERACAO_PADRAO)


def carregar_modelo(pasta=PASTA_PADRAO, device: str = "cpu"):
    """Retorna (modelo em modo eval, tokenizador, config.json, generation_config)."""
    pasta = Path(pasta)
    faltando = [a for a in (ARQ_PESOS, ARQ_CONFIG, ARQ_VOCAB) if not (pasta / a).exists()]
    if faltando:
        raise FileNotFoundError(
            f"a pasta {pasta} não tem {', '.join(faltando)}. Rode `python treinar.py` primeiro."
        )
    config = json.loads((pasta / ARQ_CONFIG).read_text(encoding="utf-8"))
    tokenizador = carregar_tokenizador(pasta / ARQ_VOCAB)
    modelo = XseloGPT(ConfigXselo.de_dict(config))
    estado = torch.load(pasta / ARQ_PESOS, map_location="cpu", weights_only=True)
    estado = {k: v.float() if v.is_floating_point() else v for k, v in estado.items()}
    estado.setdefault("lm_head.weight", estado["tok_emb.weight"])
    modelo.load_state_dict(estado)
    modelo.to(device).eval()

    geracao = dict(GERACAO_PADRAO)
    if (pasta / ARQ_GERACAO).exists():
        geracao.update(json.loads((pasta / ARQ_GERACAO).read_text(encoding="utf-8")))
    return modelo, tokenizador, config, geracao
