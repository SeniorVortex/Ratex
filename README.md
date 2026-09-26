# Ratex

Ratex oferece IAs feitas por outras IAs: modelos extremamente pequenos, focados em nichos específicos. No futuro, um modelo poderoso.

## Nosso modelo: `ratex/xselo-0-1/v1`

O **Xselo** é um micro-Transformer (estilo nanoGPT) treinado **do zero absoluto**, em PyTorch puro, e hiperfocado em **Touhou Project**: Gensokyo, danmaku, os jogos do ZUN, as músicas, os incidentes e as personagens (Reimu, Marisa, Sakuya, Remilia, Flandre, Cirno, Yukari, Youmu, Yuyuko, Sanae, Nazrin e muitas outras).

A prosa dele é descontraída, sem academicismo, e feita para explicar Touhou para neandertais, ou seja, gente que nunca ouviu falar disso na vida.

```
$ python gerar.py "Touhou é"
Touhou é um jogo de ritmo de tiros de fãs chamam isso de Touhou. É uma tanuki rata que vive
bebe o jeito mais fofo.
```

### Estado atual da v1

O modelo que está no repositório tem **3,2 M de parâmetros**. Foi treinado **só na CPU** (4 núcleos, ~40 min) com o `dataset.txt` (~146 mil caracteres) e chegou a loss de validação **1,33**. Ele já escreve em português com o tom e o vocabulário de Touhou e respeita o formato de conversa, **mas ainda mistura fatos e personagens**. É a versão preliminar. O caminho para ele ficar bom está no [roadmap](#próximos-passos-roadmap): mais dataset e treino numa GPU.

---

## Estrutura do repositório

```
Ratex/
├── dataset.txt            # o texto de treino (Touhou explicado na linguagem da rua)
├── dados_extras/          # cole aqui mais .txt; eles entram no treino automaticamente
├── treinar.py             # treina o modelo do zero e salva em ratex/xselo-0-1/v1/
├── gerar.py               # gera texto / conversa com o modelo treinado
├── nucleo/
│   ├── modelo.py          # arquitetura Transformer (embeddings, atenção, FFN, LayerNorm, logits)
│   ├── tokenizador.py     # tokenizador por caractere (padrão) e BPE leve (opcional)
│   └── pasta_modelo.py    # salvar/carregar no formato de pasta estilo Hugging Face
├── ratex/xselo-0-1/v1/    # O MODELO
│   ├── pytorch_model.bin      # pesos (state_dict do PyTorch)
│   ├── config.json            # vocab_size, n_embd, n_head, n_layer, block_size, ... + dados do treino
│   ├── vocab.json             # tokenizador (tipo + tokens [+ merges do BPE])
│   ├── generation_config.json # temperatura/top-k/top-p padrão do gerar.py
│   └── README.md              # model card
├── checkpoints/           # (ignorado pelo git) checkpoint para retomar treino
└── requirements.txt
```

## Instalação

Precisa de **Python 3.10+**.

```bash
git clone https://github.com/SeniorVortex/Ratex.git
cd Ratex
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Se você **não tem GPU NVIDIA**, a versão só-CPU do PyTorch é bem menor (~200 MB em vez de ~2 GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

Para GPU NVIDIA (CUDA) ou Mac com chip Apple (MPS), o `pip install -r requirements.txt` normal já serve. O script detecta sozinho o que está disponível.

## Treinando: `python treinar.py`

```bash
python treinar.py
```

Pronto. O script:

1. lê o `dataset.txt` e todos os `.txt` de `dados_extras/`;
2. monta o vocabulário;
3. separa 10% do texto para validação (pedaços espalhados pelo arquivo inteiro, não só o final);
4. treina com `CrossEntropyLoss`, mostrando a loss caindo no terminal, e de tempos em tempos gera uma amostra para você ver o texto melhorando;
5. salva o modelo com a **menor loss de validação** em `ratex/xselo-0-1/v1/`;
6. para sozinho quando a validação para de melhorar (parada antecipada), porque dali em diante o modelo só estaria decorando o texto.

Exemplo de saída:

```
iter   1950/2500 | época  90.69 | loss 1.0466 | lr 2.12e-04 |  964.0 ms/it | falta ~8m50s
== avaliação it 1950: loss treino 0.7996 | validação 1.3319  <- melhor até agora
   -> modelo salvo em ratex/xselo-0-1/v1 (melhor validação)
```

### Presets

O `--preset auto` (padrão) usa `gpu` se encontrar CUDA/MPS e `cpu` se não encontrar.

| preset  | camadas | cabeças | n_embd | contexto | parâmetros | uso |
|---------|---------|---------|--------|----------|-----------:|-----|
| `teste` | 2 | 2 | 64  | 64  | ~0,1M | só ver se tudo funciona (segundos) |
| `cpu`   | 4 | 4 | 256 | 192 | ~3,2M | CPU comum, ~40 min em 4 núcleos |
| `gpu`   | 6 | 6 | 384 | 256 | ~10,8M | GPU, poucos minutos |

Qualquer hiperparâmetro pode ser sobrescrito:

```bash
python treinar.py --preset teste                     # teste relâmpago
python treinar.py --preset gpu --max-iters 8000      # treino mais longo na GPU
python treinar.py --n-layer 6 --n-embd 320 --n-head 5 --block-size 256 --dropout 0.2
python treinar.py --epocas 50                        # define a duração em passadas pelo dataset
python treinar.py --tempo-max 30                     # para sozinho (e salva) em 30 minutos
python treinar.py --paciencia 0                      # desliga a parada antecipada (padrão: 4 avaliações)
python treinar.py --retomar                          # continua do último checkpoint
python treinar.py --tokenizador bpe --vocab-bpe 512  # sub-palavras em vez de caracteres
python treinar.py --criterio final                   # salva o modelo do fim, não o de menor validação
python treinar.py --help                             # todas as opções
```

`Ctrl+C` no meio do treino também salva o que já foi aprendido.

**Caractere ou BPE?** Com o dataset atual (~150 KB), o tokenizador por caractere é o que funciona melhor. Quando o dataset passar de ~1 MB, vale testar o `--tokenizador bpe`: cada token vira um pedaço de palavra, e a mesma janela de contexto enxerga ~3x mais texto.

## Gerando texto: `python gerar.py`

```bash
# continua um texto
python gerar.py "Touhou é"
python gerar.py "Reimu"
python gerar.py "Rato, pera que caralhos rato esta fazendo aqui eee ranego em"

# bate-papo: você pergunta, o Xselo responde (lembra das últimas falas)
python gerar.py --chat

# modo interativo: digita um começo de texto e ele completa
python gerar.py
```

Controles de amostragem:

| opção | padrão | o que faz |
|-------|-------:|-----------|
| `--temperatura` / `-t` | 0.8 | menor = mais conservador; maior = mais criativo e caótico |
| `--top-k` | 40 | só sorteia entre os K tokens mais prováveis |
| `--top-p` | 0.95 | e, entre eles, só os que somam P de probabilidade (nucleus sampling) |
| `--penalidade` | 1.0 | penaliza tokens repetidos recentemente (útil com BPE; 1.0 = desligado) |
| `--tokens` | 500 | máximo de tokens gerados |
| `--amostras` / `-n` | 1 | gera várias versões para o mesmo prompt |
| `--seed` | — | deixa o resultado reproduzível |
| `--parar-em` | — | interrompe quando aparecer um texto (ex.: `--parar-em "Pessoa:"`) |

Além disso, a geração para sozinha se o modelo entrar em loop, repetindo o mesmo trecho várias vezes seguidas. Os padrões ficam em `ratex/xselo-0-1/v1/generation_config.json`, e dá para editar esse arquivo.

## O dataset

O `dataset.txt` é escrito em português, no tom do Xselo, e tem:

- **visão geral de Touhou** na linguagem da rua ("jogo de desviar de mil tiros coloridos feito por um japonês que bebe cerveja");
- **ZUN, Gensokyo e seus lugares**, a mecânica de danmaku (hitbox, foco, bomba, graze, spell cards, 1cc, Lunatic);
- **todos os jogos principais**, do PC-98 ao Touhou 20, com o incidente de cada um;
- **dicionário de personagens** (poderes, manias e memes: Reimu sem dinheiro, Marisa que "pega emprestado", Cirno a fada 9, Yukari "17 anos", o karisma da Remilia...);
- **músicas de ZUN** e a cena de remixes;
- **guia de por onde começar**;
- **centenas de diálogos** `Pessoa:` / `Xselo:` respondendo dúvidas de leigos;
- crônicas curtas de Gensokyo e analogias práticas.

### Adicionando mais texto

- Cole direto no `dataset.txt`, **ou**
- crie qualquer arquivo `.txt` dentro de `dados_extras/` (entra no treino sozinho), **ou**
- aponte outros arquivos/pastas: `python treinar.py --dados dataset.txt meus_textos/ outro.txt`.

Para o modo `--chat` funcionar bem, os diálogos seguem o formato **uma fala por linha**:

```
Pessoa: quem é a Cirno?
Xselo: A Cirno é a fada do gelo, a baixinha de azul que se acha a mais forte de Gensokyo...

Pessoa: próxima conversa...
```

Depois de adicionar texto, rode `python treinar.py` de novo (treino do zero, com vocabulário novo). Se usar `--retomar`, caracteres que não existiam no vocabulário antigo são ignorados.

## A arquitetura

Transformer decoder-only (a mesma família do GPT), escrito do zero em `nucleo/modelo.py`:

```
tokens ─► Token Embedding + Positional Embedding (aprendido)
       ─► N × [ LayerNorm ─► Multi-Head Self-Attention causal ─► + residual
                LayerNorm ─► Feed-Forward 4× (GELU)            ─► + residual ]
       ─► LayerNorm final ─► Linear ─► logits (pesos amarrados ao embedding)
```

Detalhes e otimizações:

- **Pre-LayerNorm**, que deixa o treino bem mais estável em modelos pequenos;
- **weight tying** entre o embedding e a camada de saída: menos parâmetros com a mesma qualidade;
- atenção via `F.scaled_dot_product_attention` (kernels Flash/eficientes quando disponíveis);
- **AdamW** com weight decay só nas matrizes, **warmup + decaimento cosseno** do learning rate e gradient clipping;
- **bf16/fp16 automático** na GPU, `--compile` opcional para `torch.compile`;
- validação por **pedaços intercalados** do dataset, e o modelo salvo é o de melhor validação (evita guardar um modelo que só decorou o texto).

## Próximos passos (roadmap)

- [ ] Mais dataset (é a melhoria com melhor custo-benefício)
- [ ] Treino na GPU com o preset `gpu` e contexto maior (mais "memória" na conversa)
- [ ] Tokenizador BPE como padrão quando o dataset crescer
- [ ] RoPE no lugar do embedding posicional aprendido
- [ ] Ajuste fino em diálogos (instruction tuning) para uma prosa ainda melhor
- [ ] Memória de conversa mais longa no `gerar.py --chat`

---

Touhou Project é uma obra de ZUN / Team Shanghai Alice. Este é um projeto de fã, sem fins comerciais.
