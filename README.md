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

## As versões

| versão | o que é | status |
|---|---|---|
| `ratex/xselo-0-1/v1` | micro-Transformer de 3,2 M feito do zero, só Touhou | treinado (nota **8,3**/100 na prova) |
| `ratex/xselo-0-2/v1` | Qwen2.5-0.5B-Instruct + LoRA, só Touhou | código pronto, falta treinar |
| `ratex/xselo-0-3/v1` | **o grande update**: base de até 7B + LoRA com Touhou, assuntos gerais e matemática | código pronto, falta treinar |

## `ratex/xselo-0-3/v1`: saindo da bolha

A ideia é juntar dois mundos:

- **a inteligência de um modelo pronto.** Matemática, palavras do dia a dia, ciência e conversa vêm do modelo base, que já leu boa parte da internet. O 0.3 escolhe a base pelo hardware:

  | onde roda | base escolhida pelo `auto` |
  |---|---|
  | CPU | `Qwen2.5-0.5B-Instruct` |
  | Mac (MPS) | `Qwen2.5-1.5B-Instruct` |
  | GPU de 6 a 12 GB | `Qwen2.5-1.5B-Instruct` |
  | GPU de 12 a 14 GB | `Qwen2.5-3B-Instruct` |
  | GPU de 14 GB+ com `bitsandbytes` | `Qwen2.5-7B-Instruct` em 4 bits (QLoRA) |
  | GPU de 40 GB+ | `Qwen2.5-7B-Instruct` completo |

  Dá para forçar qualquer uma: `--base qwen-0.5b | qwen-1.5b | qwen-3b | qwen-7b | smollm2-1.7b | smollm-135m`, um id do Hugging Face ou uma pasta local.

- **a personalidade do Xselo em qualquer assunto.** O LoRA é treinado com uma mistura de ~1000 conversas:
  - **Touhou**: todo o `dataset.txt` virando conversa (diálogos escritos à mão + perguntas de leigo geradas a partir de cada parágrafo);
  - **assuntos gerais** (`dados_gerais/`): ~90 conversas escritas à mão sobre ciência, Brasil e mundo, português, matemática, tecnologia e dia a dia, no tom do Xselo, incluindo respostas honestas de "isso eu não sei" (doença rara, notícia recente);
  - **matemática** (`nucleo/matematica.py`): 300 exercícios gerados com passo a passo (soma, porcentagem, desconto, regra de três, equação, área, média, fração, troco, conversão de unidade...). As contas são feitas com frações exatas, então o dataset nunca ensina conta errada.

O LoRA treina só ~1 a 2% dos pesos e deixa a base congelada, então o Xselo ganha o jeito sem esquecer o que a base já sabia.

```bash
pip install -r requirements.txt
python treinar_lora.py                              # 0.3 com a base automática
python treinar_lora.py --base qwen-7b --4bit        # 0.3 no Qwen 7B (GPU NVIDIA ~16 GB + pip install bitsandbytes)
python treinar_lora.py --base qwen-3b --checkpointing  # menos memória, um pouco mais lento
python treinar_lora.py --versao 0.2                 # a 0.2 (só Touhou, Qwen 0.5B)
python gerar.py --chat                              # conversa com o híbrido mais novo treinado
python gerar.py --modelo 0.2 --chat                 # escolhe a versão: 0.1, 0.2 ou 0.3
```

| tempo estimado (3 épocas) | CPU (4 núcleos) | GPU boa (ex.: RTX 3090/4090, A100) |
|---|---|---|
| 0.3 com `qwen-0.5b` | ~2,5 h | ~5 min |
| 0.3 com `qwen-1.5b` | ~7 h | ~10 min |
| 0.3 com `qwen-7b --4bit` | inviável | ~30-60 min |

Opções úteis:

```bash
python treinar_lora.py --epocas 5 --rank 32               # treino mais forte
python treinar_lora.py --matematica 1000                  # mais exercícios de matemática
python treinar_lora.py --tempo-max 60                     # para e salva em 60 minutos
python treinar_lora.py --mesclar                          # também salva o modelo completo (base+LoRA) em .../mesclado
python treinar_lora.py --exportar-conversas conv.jsonl    # só mostra as conversas geradas
python gerar.py --chat --base /pasta/do/Qwen2.5-7B-Instruct   # usa um modelo base já baixado (offline)
```

**Precisa de internet para o Hugging Face** na primeira vez, para baixar o modelo base. Sem acesso ao `huggingface.co`, baixe a pasta do modelo em outro lugar e passe `--base /caminho`.

## A prova: `python avaliar.py`

Para saber se uma versão nova é melhor de verdade, todas passam pela mesma prova de 36 perguntas: 12 de Touhou, 12 de conhecimento geral e 12 contas de matemática que **não** aparecem no treino.

```bash
python avaliar.py --modelo 0.1                      # nota do v1
python avaliar.py --modelo 0.3 --salvar avaliacoes/xselo-0-3-v1.json --mostrar
```

| versão | touhou | geral | matemática | total |
|---|---:|---:|---:|---:|
| 0.1 | 16,7 | 8,3 | 0,0 | **8,3** |
| 0.2 | — | — | — | — |
| 0.3 | — | — | — | — |

A correção é por palavra-chave e número certo, então é uma régua simples: serve para comparar versões, não para medir inteligência de forma absoluta. Os relatórios com todas as respostas ficam em `avaliacoes/`.

Como o dataset vira conversa (`nucleo/dados_chat.py`):

- os diálogos `Pessoa:`/`Xselo:` entram como estão e contam em dobro, por serem o formato-alvo;
- as fichas `Nome: descrição` viram "quem é Nome?", "qual o poder de Nome?", "quais os memes de Nome?"...;
- jogos, lugares, mecânica, músicas, memes, crônicas e analogias viram perguntas de leigo;
- o treino só calcula a loss nas **respostas do Xselo**, e o system prompt fixa a persona.

---

## Estrutura do repositório

```
Ratex/
├── dataset.txt            # o texto de treino (Touhou explicado na linguagem da rua)
├── dados_extras/          # cole aqui mais .txt; eles entram no treino automaticamente
├── dados_gerais/          # 0.3: conversas sobre ciência, português, matemática, tecnologia, dia a dia
├── treinar.py             # v1: treina o micro-Transformer do zero -> ratex/xselo-0-1/v1/
├── treinar_lora.py        # 0.2/0.3: treina o LoRA em cima do modelo base -> ratex/xselo-0-X/v1/
├── gerar.py               # conversa / gera texto (híbrido mais novo se existir, senão v1)
├── avaliar.py             # prova fixa com nota por categoria (touhou, geral, matemática)
├── avaliacoes/            # relatórios da prova de cada versão
├── nucleo/
│   ├── modelo.py          # arquitetura Transformer (embeddings, atenção, FFN, LayerNorm, logits)
│   ├── tokenizador.py     # tokenizador por caractere (padrão) e BPE leve (opcional)
│   ├── pasta_modelo.py    # salvar/carregar no formato de pasta estilo Hugging Face
│   ├── dados_chat.py      # transforma o dataset em conversas + system prompts do Xselo
│   ├── matematica.py      # 0.3: exercícios de matemática com passo a passo e resposta exata
│   └── hibrido.py         # versões, bases, carrega base + LoRA e gera respostas
├── ratex/xselo-0-1/v1/    # O MODELO
│   ├── pytorch_model.bin      # pesos (state_dict do PyTorch)
│   ├── config.json            # vocab_size, n_embd, n_head, n_layer, block_size, ... + dados do treino
│   ├── vocab.json             # tokenizador (tipo + tokens [+ merges do BPE])
│   ├── generation_config.json # temperatura/top-k/top-p padrão do gerar.py
│   └── README.md              # model card
├── ratex/xselo-0-2/v1/    # (depois do treino) adaptador LoRA + tokenizador + ratex_config.json
├── ratex/xselo-0-3/v1/    # (depois do treino) idem, a versão geral
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

Se o `ratex/xselo-0-2/v1/` existir, o `gerar.py` usa ele (veja a seção do xselo-0-2 acima). Os exemplos abaixo são do v1:

```bash
# continua um texto
python gerar.py --modelo ratex/xselo-0-1/v1 "Touhou é"
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

- [x] Ajuste fino em diálogos em cima de um modelo base (0.2, LoRA)
- [x] Sair da bolha: assuntos gerais + matemática com passo a passo (0.3)
- [x] Prova fixa para comparar versões (`avaliar.py`)
- [ ] Treinar a 0.2 e a 0.3 (precisa de acesso ao Hugging Face para baixar a base)
- [ ] Treinar a 0.3 no Qwen 7B numa GPU (Colab ou GPU alugada)
- [ ] Mais dataset: mais conversas gerais e de Touhou no tom do Xselo (a melhoria com melhor custo-benefício)
- [ ] RAG: o Xselo consulta o próprio dataset antes de responder (corta a mistura de fatos)
- [ ] Memória de conversa mais longa no `gerar.py --chat` (resumo das falas antigas)
- [ ] Prova maior e mais difícil conforme as notas subirem

---

Touhou Project é uma obra de ZUN / Team Shanghai Alice. Este é um projeto de fã, sem fins comerciais.
