"""
Arquitetura do Xselo: um micro-Transformer decoder-only no estilo nanoGPT.

    tokens -> Token Embedding + Positional Embedding
           -> N x [LayerNorm -> Multi-Head Self-Attention causal -> +residual
                   LayerNorm -> Feed-Forward (GELU)             -> +residual]
           -> LayerNorm final -> projeção linear para logits (pesos amarrados
              com o embedding de tokens)

Escolhas (em relação ao Transformer "de livro"):
* Pre-LayerNorm: treino bem mais estável em modelos pequenos.
* Embedding posicional aprendido (como GPT-2/nanoGPT) em vez do senoidal fixo.
* Weight tying entre embedding e cabeça de saída: menos parâmetros, mesmo resultado.
* `scaled_dot_product_attention` do PyTorch: usa kernels otimizados (Flash /
  memória eficiente) quando disponíveis, na CPU ou na GPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ConfigXselo:
    vocab_size: int = 128
    block_size: int = 256  # tamanho máximo do contexto (em tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False  # bias nas Linear/LayerNorm; sem bias é um pouco melhor e mais rápido

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def de_dict(cls, dados: dict) -> "ConfigXselo":
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dados.items() if k in nomes})


class AtencaoCausal(nn.Module):
    """Multi-Head Self-Attention com máscara causal (cada token só olha pro passado)."""

    def __init__(self, cfg: ConfigXselo):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd precisa ser divisível por n_head"
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.drop_resid = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop_resid(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: ConfigXselo):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Bloco(nn.Module):
    def __init__(self, cfg: ConfigXselo):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = AtencaoCausal(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class XseloGPT(nn.Module):
    def __init__(self, cfg: ConfigXselo):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocos = nn.ModuleList([Bloco(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_pesos)
        # projeções que somam no fluxo residual começam menores (truque do GPT-2)
        for nome, p in self.named_parameters():
            if nome.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_pesos(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_parametros(self) -> int:
        # conta o embedding amarrado uma vez só
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, alvos: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequência de {T} tokens > block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for bloco in self.blocos:
            x = bloco(x)
        x = self.ln_f(x)
        if alvos is None:
            # na geração só precisamos do último passo
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), alvos.view(-1))
        return logits, loss

    def otimizador(self, lr: float, weight_decay: float, betas=(0.9, 0.99), device_type: str = "cpu"):
        """AdamW com weight decay só nas matrizes (não em LayerNorm/bias)."""
        params = [p for p in self.parameters() if p.requires_grad]
        com_decay = [p for p in params if p.dim() >= 2]
        sem_decay = [p for p in params if p.dim() < 2]
        grupos = [
            {"params": com_decay, "weight_decay": weight_decay},
            {"params": sem_decay, "weight_decay": 0.0},
        ]
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(grupos, lr=lr, betas=betas, **extra)

    @torch.no_grad()
    def proximo_token(
        self,
        idx: torch.Tensor,
        temperatura: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        penalidade_repeticao: float = 1.0,
        janela_repeticao: int = 64,
    ) -> torch.Tensor:
        """Amostra o próximo token para cada sequência de `idx` (B, T) -> (B, 1)."""
        idx_cond = idx[:, -self.cfg.block_size :]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :].float()

        if penalidade_repeticao != 1.0:
            recentes = idx[:, -janela_repeticao:]
            valores = torch.gather(logits, 1, recentes)
            valores = torch.where(valores > 0, valores / penalidade_repeticao, valores * penalidade_repeticao)
            logits.scatter_(1, recentes, valores)

        if temperatura <= 0:  # guloso
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperatura
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            limite = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < limite, float("-inf"))
        if top_p is not None and 0 < top_p < 1:
            ordenados, indices = torch.sort(logits, descending=True)
            acumulado = torch.cumsum(F.softmax(ordenados, dim=-1), dim=-1)
            remover = acumulado - F.softmax(ordenados, dim=-1) > top_p  # mantém ao menos 1
            ordenados = ordenados.masked_fill(remover, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, indices, ordenados)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def gerar(self, idx: torch.Tensor, max_novos: int, **amostragem) -> torch.Tensor:
        for _ in range(max_novos):
            prox = self.proximo_token(idx, **amostragem)
            idx = torch.cat([idx, prox], dim=1)
        return idx
