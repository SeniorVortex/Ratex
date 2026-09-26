"""
Xselo "híbrido": modelo base instruído do Hugging Face + um adaptador LoRA
treinado com a prosa do Xselo.

Versões:
    0.2  ratex/xselo-0-2/v1  Qwen2.5-0.5B-Instruct + LoRA, só o dataset de Touhou do v1
    0.3  ratex/xselo-0-3/v1  base escolhida pelo hardware (até Qwen2.5-7B) + LoRA com
                             Touhou + assuntos gerais (dados_gerais/) + matemática gerada

A pasta de cada versão guarda só o que é nosso:

    adapter_model.safetensors   pesos do LoRA (poucas dezenas de MB)
    adapter_config.json         configuração do LoRA (formato PEFT)
    ratex_config.json           versão, modelo base, system prompt, geração padrão e dados do treino
    tokenizer*.json, ...        tokenizador do modelo base (com o chat template)

O modelo base é baixado do Hugging Face na primeira vez que for usado (e fica
em cache), ou pode ser apontado para uma pasta local com --base.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .dados_chat import SYSTEM_PROMPT, SYSTEM_PROMPT_03

RAIZ = Path(__file__).resolve().parent.parent
ARQ_RATEX = "ratex_config.json"

# apelidos curtos para os modelos base mais úteis (todos instruídos e com chat template)
BASES = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "smollm-135m": "HuggingFaceTB/SmolLM-135M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

VERSOES = {
    "0.2": {
        "nome": "ratex/xselo-0-2/v1",
        "base": "qwen-0.5b",
        "dados": ["dataset.txt", "dados_extras"],
        "matematica": 0,
        "geral": False,
        "system_prompt": SYSTEM_PROMPT,
    },
    "0.3": {
        "nome": "ratex/xselo-0-3/v1",
        "base": "auto",
        "dados": ["dataset.txt", "dados_extras", "dados_gerais"],
        "matematica": 300,
        "geral": True,
        "system_prompt": SYSTEM_PROMPT_03,
    },
}
VERSAO_PADRAO = "0.3"

# compatibilidade com o código do 0.2
NOME_HIBRIDO = VERSOES["0.2"]["nome"]
PASTA_HIBRIDO = RAIZ / NOME_HIBRIDO
BASE_PADRAO = BASES["qwen-0.5b"]

GERACAO_HIBRIDO = {
    "temperatura": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "penalidade_repeticao": 1.1,
    "max_novos_tokens": 400,
}


def pasta_da_versao(versao: str) -> Path:
    return RAIZ / VERSOES[versao]["nome"]


def eh_hibrido(pasta) -> bool:
    pasta = Path(pasta)
    return (pasta / "adapter_config.json").exists() and (pasta / ARQ_RATEX).exists()


def pasta_mais_nova() -> Path | None:
    """A versão híbrida treinada mais nova que existir no repositório."""
    for versao in sorted(VERSOES, key=lambda v: tuple(map(int, v.split("."))), reverse=True):
        if eh_hibrido(pasta_da_versao(versao)):
            return pasta_da_versao(versao)
    return None


def escolher_device(pedido: str = "auto") -> str:
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bitsandbytes_disponivel() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolver_base(base: str, device: str) -> tuple[str, bool]:
    """Troca apelidos pelo id do Hugging Face e resolve 'auto' pelo hardware.
    Retorna (id_ou_pasta, usar_4bit_sugerido)."""
    if base != "auto":
        return BASES.get(base, base), False
    if device == "cpu":
        return BASES["qwen-0.5b"], False
    if device == "mps":
        return BASES["qwen-1.5b"], False
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vram >= 40:
        return BASES["qwen-7b"], False
    if vram >= 14 and bitsandbytes_disponivel():
        return BASES["qwen-7b"], True
    if vram >= 12:
        return BASES["qwen-3b"], False
    if vram >= 6:
        return BASES["qwen-1.5b"], False
    return BASES["qwen-0.5b"], False


def carregar_base(base: str, device: str, quatro_bits: bool = False):
    """Carrega o modelo base + tokenizador (id do Hugging Face, apelido ou pasta local)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("o xselo híbrido precisa de: pip install transformers peft accelerate safetensors") from e
    base = BASES.get(base, base)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device in ("cuda", "mps") else torch.float32)
    extra = {}
    if quatro_bits:
        if not bitsandbytes_disponivel():
            raise SystemExit("--4bit precisa de GPU NVIDIA e do pacote bitsandbytes (pip install bitsandbytes)")
        from transformers import BitsAndBytesConfig

        extra = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16),
            "device_map": {"": 0},
        }
    try:
        tok = AutoTokenizer.from_pretrained(base)
        modelo = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, **extra)
    except OSError as e:
        raise SystemExit(
            f"não consegui carregar o modelo base {base!r}.\n"
            "Ele é baixado de huggingface.co na primeira execução; confira sua internet, ou baixe\n"
            "a pasta do modelo em outro lugar e passe o caminho com --base /caminho/do/modelo.\n"
            f"Erro original: {e}"
        ) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not quatro_bits:
        modelo = modelo.to(device)
    return modelo, tok


def carregar_hibrido(pasta=None, device: str = "auto", base: str | None = None):
    """Retorna (modelo com LoRA em modo eval, tokenizador, ratex_config)."""
    from peft import PeftModel

    pasta = Path(pasta) if pasta else pasta_mais_nova()
    if pasta is None or not eh_hibrido(pasta):
        raise FileNotFoundError(f"{pasta} não tem um xselo híbrido treinado. Rode `python treinar_lora.py` primeiro.")
    cfg = json.loads((pasta / ARQ_RATEX).read_text(encoding="utf-8"))
    device = escolher_device(device)
    quatro_bits = bool(cfg.get("quatro_bits")) and device == "cuda" and bitsandbytes_disponivel()
    modelo, tok = carregar_base(base or cfg["base"], device, quatro_bits=quatro_bits)
    if (pasta / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(pasta)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    modelo = PeftModel.from_pretrained(modelo, pasta)
    if not quatro_bits:
        modelo = modelo.to(device)
    return modelo.eval(), tok, cfg


def montar_mensagens(historico: list[dict], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, *historico]


def consulta_da_conversa(historico: list[dict]) -> str:
    """O que buscar na memória: a última pergunta, e a anterior junto se a última for curta
    ("e a irmã dela?")."""
    perguntas = [m["content"] for m in historico if m["role"] == "user"]
    if not perguntas:
        return ""
    if len(perguntas) > 1 and len(perguntas[-1].split()) <= 5:
        return f"{perguntas[-2]} {perguntas[-1]}"
    return perguntas[-1]


@torch.no_grad()
def responder(modelo, tok, historico: list[dict], system_prompt: str = SYSTEM_PROMPT,
              stream: bool = True, max_novos_tokens: int = 400, temperatura: float = 0.7,
              top_k: int = 40, top_p: float = 0.9, penalidade_repeticao: float = 1.1,
              memoria=None, **_) -> str:
    """Gera a próxima resposta do Xselo para uma conversa [{role, content}, ...].
    Com `memoria` (nucleo.memoria.Memoria), os trechos relevantes do dataset entram no prompt."""
    from transformers import TextStreamer

    if memoria is not None:
        from .memoria import prompt_com_memoria

        system_prompt = prompt_com_memoria(system_prompt, memoria.buscar(consulta_da_conversa(historico)))
    texto = tok.apply_chat_template(montar_mensagens(historico, system_prompt), tokenize=False,
                                    add_generation_prompt=True)
    entrada = tok(texto, return_tensors="pt", add_special_tokens=False).to(modelo.device)
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
