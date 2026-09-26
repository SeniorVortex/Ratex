"""
xselo-0-2: modelo "híbrido" = modelo base instruído do Hugging Face
(Qwen2.5-0.5B-Instruct por padrão, ou SmolLM-135M-Instruct) + um adaptador
LoRA treinado com o dataset e a prosa do Xselo v1.

A pasta ratex/xselo-0-2/v1/ guarda só o que é nosso:

    adapter_model.safetensors   pesos do LoRA (poucos MB)
    adapter_config.json         configuração do LoRA (formato PEFT)
    ratex_config.json           modelo base, system prompt, geração padrão e dados do treino
    tokenizer*.json, ...        tokenizador do modelo base (com o chat template)

O modelo base é baixado do Hugging Face na primeira vez que for usado (e fica
em cache), ou pode ser apontado para uma pasta local com --base.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .dados_chat import SYSTEM_PROMPT

RAIZ = Path(__file__).resolve().parent.parent
NOME_HIBRIDO = "ratex/xselo-0-2/v1"
PASTA_HIBRIDO = RAIZ / NOME_HIBRIDO
BASE_PADRAO = "Qwen/Qwen2.5-0.5B-Instruct"
ARQ_RATEX = "ratex_config.json"

GERACAO_HIBRIDO = {
    "temperatura": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "penalidade_repeticao": 1.1,
    "max_novos_tokens": 320,
}


def eh_hibrido(pasta) -> bool:
    pasta = Path(pasta)
    return (pasta / "adapter_config.json").exists() and (pasta / ARQ_RATEX).exists()


def escolher_device(pedido: str = "auto") -> str:
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def carregar_base(base: str, device: str):
    """Carrega o modelo base + tokenizador (id do Hugging Face ou pasta local)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("o xselo-0-2 precisa de: pip install transformers peft accelerate safetensors") from e
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    try:
        tok = AutoTokenizer.from_pretrained(base)
        modelo = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    except OSError as e:
        raise SystemExit(
            f"não consegui carregar o modelo base {base!r}.\n"
            "Ele é baixado de huggingface.co na primeira execução; confira sua internet, ou baixe\n"
            "a pasta do modelo em outro lugar e passe o caminho com --base /caminho/do/modelo.\n"
            f"Erro original: {e}"
        ) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return modelo.to(device), tok


def carregar_hibrido(pasta=PASTA_HIBRIDO, device: str = "auto", base: str | None = None):
    """Retorna (modelo com LoRA em modo eval, tokenizador, ratex_config)."""
    from peft import PeftModel

    pasta = Path(pasta)
    if not eh_hibrido(pasta):
        raise FileNotFoundError(f"{pasta} não tem um xselo-0-2 treinado. Rode `python treinar_lora.py` primeiro.")
    cfg = json.loads((pasta / ARQ_RATEX).read_text(encoding="utf-8"))
    device = escolher_device(device)
    modelo, tok = carregar_base(base or cfg["base"], device)
    if (pasta / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(pasta)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    modelo = PeftModel.from_pretrained(modelo, pasta).to(device).eval()
    return modelo, tok, cfg


def montar_mensagens(historico: list[dict], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, *historico]


@torch.no_grad()
def responder(modelo, tok, historico: list[dict], system_prompt: str = SYSTEM_PROMPT,
              stream: bool = True, max_novos_tokens: int = 320, temperatura: float = 0.7,
              top_k: int = 40, top_p: float = 0.9, penalidade_repeticao: float = 1.1, **_) -> str:
    """Gera a próxima resposta do Xselo para uma conversa [{role, content}, ...]."""
    from transformers import TextStreamer

    texto = tok.apply_chat_template(montar_mensagens(historico, system_prompt), tokenize=False,
                                    add_generation_prompt=True)
    entrada = tok(texto, return_tensors="pt").to(modelo.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    amostrar = temperatura > 0
    saida = modelo.generate(
        **entrada,
        max_new_tokens=max_novos_tokens,
        do_sample=amostrar,
        temperature=temperatura if amostrar else None,
        top_k=top_k if amostrar else None,
        top_p=top_p if amostrar else None,
        repetition_penalty=penalidade_repeticao,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )
    return tok.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()
