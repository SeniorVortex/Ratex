---
language: pt
tags:
  - ratex
  - touhou
  - nanogpt
  - char-level
  - pytorch
---

# ratex/xselo-0-1/v1

**Xselo** é um micro-Transformer decoder-only treinado do zero para falar de **Touhou Project** em português, num tom descontraído, explicando a série para quem nunca ouviu falar dela.

## Arquivos

| arquivo | conteúdo |
|---------|----------|
| `pytorch_model.bin` | pesos (`state_dict` do PyTorch, float32; `lm_head` amarrado ao `tok_emb`) |
| `config.json` | hiperparâmetros da rede + metadados do treino |
| `vocab.json` | tokenizador por caractere (90 símbolos) |
| `generation_config.json` | amostragem padrão do `gerar.py` |

## Arquitetura

| | |
|---|---|
| tipo | Transformer decoder-only (estilo GPT/nanoGPT), pre-LayerNorm |
| camadas (`n_layer`) | 4 |
| cabeças de atenção (`n_head`) | 4 |
| dimensão (`n_embd`) | 256 |
| contexto (`block_size`) | 192 caracteres |
| vocabulário (`vocab_size`) | 90 (caracteres) |
| parâmetros | 3,22 M |

## Treino

- **Dados:** `dataset.txt` (146 mil caracteres): visão geral de Touhou, ZUN, Gensokyo, mecânica de danmaku, todos os jogos principais, dicionário de personagens e memes, músicas, guia para iniciantes, 235 diálogos `Pessoa:`/`Xselo:` e crônicas curtas.
- **Validação:** 10% do texto, em pedaços intercalados ao longo do arquivo.
- **Otimização:** AdamW (lr 1e-3, weight decay 0.1), warmup de 100 iterações e decaimento cosseno, batch de 32 × 192 tokens, dropout 0.2, gradient clipping 1.0.
- **Máquina:** CPU de 4 núcleos, sem GPU; ~42 min para 2500 iterações.
- **Resultado:** melhor loss de validação **1.3319** (cross-entropy por caractere) na iteração 1950, a versão salva aqui. A loss de treino nesse ponto era 0.80.

Reproduzir: `python treinar.py --preset cpu` (seed 1337).

## Uso

```bash
python gerar.py "Touhou é"
python gerar.py --chat
python gerar.py "Reimu" --temperatura 0.7 --amostras 3
```

## Exemplo real (temperatura 0.7)

```
você> quem é a Cirno?
xselo> A Marisa Kirisame é a bibliotecária da Marisa, capitã mais de tudo com somple no
       portão da vilã de loja. Ela é incidente entendertal, mas não tem uma capa e consegue
       estudando por fãs chamam isso que ela é uma mais poderosa.
```

## Limitações (honestas)

É a **v1 preliminar**. Com 3 M de parâmetros e ~150 KB de texto, o modelo aprendeu:

- o **português** e o **tom** do dataset ("os fãs zoam", "é tipo...", "a Reimu é pobre");
- o **vocabulário** de Touhou (nomes, jogos, lugares, spell cards);
- o **formato** de conversa `Pessoa:`/`Xselo:`.

Mas ele **ainda mistura fatos e personagens** e inventa palavras. Isso é esperado para esse tamanho; não use as respostas como fonte. Os caminhos para melhorar, em ordem de impacto:

1. **mais texto** no dataset (o maior gargalo hoje);
2. treinar o preset `gpu` (~11 M de parâmetros, contexto de 256) numa GPU;
3. BPE (`--tokenizador bpe`) quando o dataset passar de ~1 MB.

Touhou Project é obra de ZUN / Team Shanghai Alice. Projeto de fã, sem fins comerciais.
