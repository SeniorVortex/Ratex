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
| `ratex/xselo-0-1/v1` | micro-Transformer de 3,2 M feito do zero, só Touhou | treinado: nota **8,3**/100 na prova |
| `ratex/xselo-0-2/v1` | Qwen2.5-0.5B-Instruct + LoRA, só Touhou | pulada: a 0.3 faz tudo que ela faria (`--versao 0.2` ainda treina) |
| `ratex/xselo-0-3/v1` | **o grande update**: Qwen + LoRA com Touhou, assuntos gerais e matemática, + memória de consulta | **treinado** no Qwen2.5-1.5B: nota **91,7**/100 na prova |
| `ratex/xselo-0-4/v1` | **a prosa como ponto forte**: Qwen2.5-32B + LoRA com escrita caprichada e muitos assuntos; Touhou vira um dos temas | treinado no Colab (A100) |
| `ratex/xselo-0-5/v1` | **base nova de 2026** (Gemma 4 31B, que enxerga imagem) + o triplo de prosa + polimento com a arena; memória longa, bot do Discord e prova com juiz | **treinado** no Colab (A100): **97,2** na prova fixa, 100 nas pegadinhas, correções e contas da difícil |

## `ratex/xselo-0-5/v1`: base nova, olhos, memória longa e Discord

- **Base nova:** [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) (Apache 2.0, sem pedir acesso). Comparei com a [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), lançada em agosto de 2026: a Qwen é mais forte em raciocínio e código (é o foco dela), e a Gemma é mais forte em várias línguas (MMMLU 88,4%), usa atenção comum (treina rápido no Colab sem pacote extra) e tem uma irmã de 12B que cabe numa placa de 16 GB. Pra um modelo cujo ponto forte é prosa em português, a Gemma ganhou. A Qwen3.8-27B continua disponível no notebook (`--base qwen3.8-27b`). As duas **enxergam imagens**.
- **Enxergar imagens:** as bases novas vêm com "olhos". O LoRA mexe só na parte de texto (a visão fica como veio), e o Xselo continua sabendo olhar. No `gerar.py --chat`, use `/imagem foto.jpg o que é isso?`; no Colab, `/foto`; no Discord, é só anexar a imagem.
- **O triplo de prosa:** `dados_prosa/` foi de 105 pra **303 conversas** escritas à mão (crônicas, contos, poesia de vários formatos, cartas, ciência, reescrita, conselho com várias trocas, Touhou, humor e respostas curtas de propósito). Pesos por época: prosa 2×, assuntos gerais 2×, Touhou 0,5×.
- **Memória de conversa longa** (`nucleo/lembrancas.py`): o Xselo lembra de você entre uma conversa e outra. Guarda fatos curtos ("se chama Ana", "mora em Recife", "gosta de Touhou") quando você conta, ou com `/lembrar <fato>`; e quando a conversa fica comprida, ele mesmo resume as mensagens antigas antes de tirá-las do prompt. `/lembrancas` mostra o que ele sabe, `/esquecer` apaga tudo.
- **A arena vira aula:** na arena, quem ganha vira a resposta "Boa" e quem perde, a "Ruim". Se nenhuma prestar, você responde `n` e escreve a certa. `python treinar_dpo.py --arena arena.json` junta essas partidas ao polimento (o notebook faz isso sozinho).
- **Bot do Discord** (`bot_discord.py`): responde quando alguém menciona o bot, responde uma mensagem dele, ou manda DM; lembra de cada pessoa, enxerga imagem anexada e pesquisa na internet com `!buscar`. O passo a passo pra criar o bot (de graça, no site oficial do Discord) está no topo do arquivo e na célula 10 do notebook.
- **Prova difícil com juiz** (`prova_dificil.py`): veja abaixo.

Pra treinar: [notebook no Colab](https://colab.research.google.com/github/SeniorVortex/Ratex/blob/main/notebooks/treinar_no_colab.ipynb), **A100**, "Executar tudo". A versão 0.5 e a Gemma 4 31B já vêm selecionadas.

## `ratex/xselo-0-4/v1`: a prosa como ponto forte

Touhou foi o nicho que fez o Xselo nascer, mas não precisa ser a jaula dele. A 0.4 muda o foco para **escrever bem sobre qualquer coisa**:

- **`dados_prosa/`**: 105 conversas escritas com capricho, incluindo crônica, conto, poema, carta, explicação de ciência com ritmo, conselho sem autoajuda barata, games, anime, futebol, mitologia, reescrita de texto, resumo, mudança de tom e respostas curtas que são curtas de propósito. As respostas agora podem ter **vários parágrafos** (veja `dados_prosa/LEIA-ME.md`).
- **Pesos por fonte:** por época, a prosa entra 3×, os assuntos gerais 2×, e o Touhou 0,6×. Touhou passa a ser ~30% dos exemplos, não mais a maioria.
- **Base Qwen2.5-32B:** o teto que dá pra treinar numa A100 de 40 GB (em 4 bits). É o que traz a inteligência: conhecimento geral, raciocínio e vocabulário.
- **System prompt novo:** escrita como ponto forte, tamanho e tom ajustados ao pedido, imagens concretas, ritmo variado, sem clichê e sem jeito de robô.
- **A prova ganhou uma parte de prosa:** 8 pedidos de escrita que não estão no treino.

Para treinar: abra o [notebook no Colab](https://colab.research.google.com/github/SeniorVortex/Ratex/blob/main/notebooks/treinar_no_colab.ipynb), escolha a **A100** e rode as células (versão 0.4 e Qwen 32B já vêm selecionados). Estimativa: ~45–90 min de treino. Na CPU, o script para com um aviso em vez de tentar.

Pra rodar em casa depois: um 32B em 4 bits ocupa ~20 GB. Numa placa de 16 GB ele roda com parte na RAM, mais devagar. Se ficar lento demais, dá pra treinar a mesma 0.4 no 14B (cabe inteiro em 16 GB), trocando o modelo na primeira célula do notebook.

## `ratex/xselo-0-3/v1`: saindo da bolha

A ideia é juntar dois mundos:

- **a inteligência de um modelo pronto.** Matemática, palavras do dia a dia, ciência e conversa vêm do modelo base, que já leu boa parte da internet. O 0.3 escolhe a base pelo hardware:

  | onde roda | base escolhida pelo `auto` |
  |---|---|
  | CPU com 14 GB+ de RAM | `Qwen2.5-1.5B-Instruct` (o que está no repositório) |
  | CPU com menos RAM | `Qwen2.5-0.5B-Instruct` |
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
python treinar_lora.py --versao 0.3                 # 0.3 com a base automática
python treinar_lora.py --versao 0.3 --base qwen-7b --4bit     # 0.3 no Qwen 7B (GPU NVIDIA ~16 GB + pip install bitsandbytes)
python treinar_lora.py --versao 0.3 --base qwen-3b --checkpointing  # menos memória, um pouco mais lento
python treinar_lora.py --versao 0.2                 # a 0.2 (só Touhou, Qwen 0.5B)
python gerar.py --chat                              # conversa com o híbrido mais novo treinado
python gerar.py --modelo 0.3 --chat                 # escolhe a versão: 0.1, 0.2, 0.3 ou 0.4
```

| tempo medido/estimado (1 época, o padrão) | CPU (4 núcleos) | GPU boa (ex.: RTX 3090/4090, A100) |
|---|---|---|
| 0.3 com `qwen-0.5b` | ~1 h (medido) | ~2 min |
| 0.3 com `qwen-1.5b` | ~2 h 10 min (medido, pico de 12 GB de RAM) | ~5 min |
| 0.3 com `qwen-7b --4bit` | inviável | ~20-30 min |

O tamanho do lote se ajusta sozinho à memória (lote efetivo de 8). Uma época basta com o dataset atual: nas duas bases testadas, a validação piorou a partir da segunda.

Opções úteis:

```bash
python treinar_lora.py --epocas 2 --rank 32               # treino mais forte (vale quando o dataset crescer)
python treinar_lora.py --matematica 1000                  # mais exercícios de matemática
python treinar_lora.py --tempo-max 60                     # para e salva em 60 minutos
python treinar_lora.py --mesclar                          # também salva o modelo completo (base+LoRA) em .../mesclado
python treinar_lora.py --exportar-conversas conv.jsonl    # só mostra as conversas geradas
python gerar.py --chat --base /pasta/do/Qwen2.5-7B-Instruct   # usa um modelo base já baixado (offline)
```

### Treinar no Google Colab (GPU)

A CPU aguenta até o 1.5B. Pra 7B, 14B ou 32B, use o notebook pronto, que roda na GPU do Colab (a A100 do Colab Pro é a ideal):

**[Abrir `notebooks/treinar_no_colab.ipynb` no Colab](https://colab.research.google.com/github/SeniorVortex/Ratex/blob/main/notebooks/treinar_no_colab.ipynb)**

Ele confere a GPU, decide sozinho entre bf16 e 4 bits, treina, roda a prova, deixa você conversar com o modelo e salva no Google Drive (e, se quiser, no Hugging Face). Até onde dá pra ir em cada GPU:

| GPU | maior modelo que dá pra treinar |
|---|---|
| T4 (16 GB) | 7B (4 bits) |
| L4 (24 GB) | 14B (4 bits) |
| A100 (40 GB) | 32B (4 bits) |

70B precisa de GPU de 80 GB (fora do Colab Pro). Modelos de 405B+ só em cluster alugado, ou usados prontos por API (OpenRouter), sem treinar.

**Precisa de internet para o Hugging Face** na primeira vez, para baixar o modelo base. Sem acesso ao `huggingface.co`, baixe a pasta do modelo em outro lugar e passe `--base /caminho`.

## A prova: `python avaliar.py`

Para saber se uma versão nova é melhor de verdade, todas passam pela mesma prova de 36 perguntas: 12 de Touhou, 12 de conhecimento geral e 12 contas de matemática que **não** aparecem no treino.

Nos modelos híbridos, a prova também pede **8 textos** (crônica, poema, conselho, descrição, reescrita...) que não estão no treino. Isso não vira nota, mas mede sintomas de texto ruim: **repetição** de trechos, **vocabulário** pobre, **ritmo** monótono (frases todas do mesmo tamanho) e **clichês** de robô ("é importante ressaltar", "espero ter ajudado"). Essas métricas só valem pra texto coerente, porque até palavras aleatórias "têm vocabulário variado". Quem julga a prosa de verdade é a leitura humana, e os textos ficam salvos no relatório.

```bash
python avaliar.py --modelo 0.1                      # nota do v1
python avaliar.py --modelo 0.3 --salvar avaliacoes/xselo-0-3-v1.json --mostrar
```

| modelo | touhou | geral | matemática | total |
|---|---:|---:|---:|---:|
| xselo 0.1 (feito do zero, 3 M) | 16,7 | 8,3 | 0 | 8,3 |
| Qwen 0.5B puro | 0 | 66,7 | 33,3 | 33,3 |
| Qwen 0.5B puro + memória | 83,3 | 66,7 | 33,3 | 61,1 |
| xselo 0.3 no Qwen 0.5B, sem memória | 25,0 | 50,0 | 83,3 | 52,8 |
| xselo 0.3 no Qwen 0.5B + memória | 83,3 | 83,3 | 83,3 | 83,3 |
| Qwen 1.5B puro + memória | 91,7 | 91,7 | 83,3 | 88,9 |
| xselo 0.3 no Qwen 1.5B, sem memória | 41,7 | 83,3 | 91,7 | 72,2 |
| **xselo 0.3 no Qwen 1.5B + memória** (o atual) | **91,7** | **91,7** | **91,7** | **91,7** |

A correção é por palavra-chave e número certo, então é uma régua simples: serve para comparar versões, não para medir inteligência de forma absoluta. Ela também é **generosa**, porque conta acerto quando a palavra certa aparece, mesmo com erro junto. Lendo resposta por resposta (acertos de 12):

| modelo | Touhou | geral | matemática |
|---|---:|---:|---:|
| xselo 0.3 no Qwen 0.5B + memória | 10 | 6 | 10 |
| **xselo 0.3 no Qwen 1.5B + memória** | **11** | **10** | **11** |

O que a tabela ensina:
- a **memória** resolve Touhou;
- a **base maior** segura o conhecimento geral: o 0.5B com LoRA chegava a responder "Lisboa" para a capital da França;
- o **LoRA** dá a matemática e o jeito do Xselo.

No 1.5B, o Qwen puro com memória já chega perto (88,9), então daqui pra frente o que mais rende é base maior e dataset maior. Os relatórios com todas as respostas ficam em `avaliacoes/`.

## A prova difícil, com juiz: `python prova_dificil.py`

A prova fixa procura palavras-chave: serve pra comparar versões, mas é generosa, e os modelos grandes já tiram quase tudo. A prova difícil mede o que importa pro Xselo, com outro modelo fazendo o papel de corretor:

| parte | o que mede | como corrige |
|---|---|---|
| pegadinhas (18) | premissa falsa, mito, obra que não existe, jogo que ainda não aconteceu | juiz, 0 a 2, com gabarito escrito |
| correções (8) | a pessoa "corrige" o Xselo: às vezes com razão (tem que aceitar), às vezes sem (tem que manter, com educação) | juiz, 0 a 2, com gabarito |
| contas (10) | problemas de vários passos | número exato, sem juiz |
| prosa (20) | pedidos de escrita com restrições (tamanho, palavras proibidas, público) | juiz, 1 a 10, com rubrica |
| pareado (20) | o mesmo pedido pro Xselo e pro modelo base puro (ou outra versão do Xselo, com `--rival`) | juiz escolhe duas vezes, trocando a ordem; só vale vitória quando as duas leituras concordam |

O juiz certo é um modelo de **outra família** (`--juiz Qwen/Qwen3.8-27B` pra julgar um Xselo feito na Gemma): um modelo tende a preferir o próprio jeito de escrever, e aí o pareado contra a base pura fica viciado. Primeiro saem todas as respostas, depois o Xselo sai da GPU e o juiz entra, então os dois cabem numa A100. Sem `--juiz`, julga a própria base com o adaptador desligado (mais rápido, mas puxa pro estilo dela). Nenhuma pergunta das duas provas está nos dados de treino.

**Primeira rodada da 0.5** (Gemma 4 31B + LoRA + 22 pares de DPO, juiz = a própria Gemma, prosa e pareado com 10 pedidos):

| parte | nota |
|---|---|
| pegadinhas | 100 |
| correções | 100 |
| contas | 100 |
| prosa | 9,8 / 10 |
| pareado | 45 (Xselo 4 × 5 Gemma pura, 1 empate) |

Na prova fixa, 97,2 (Touhou 91,7, geral 100, matemática 100). Leitura honesta: **em fatos, pegadinhas e contas, o 31B gabaritou**, e essas partes viraram só conferência de que nada piorou. A nota de prosa (9,8) não diz nada, porque o juiz foi generoso demais. O que sobra de verdade é o **pareado**, e ali deu empate técnico com a Gemma pura, julgado pela própria Gemma, que puxa pro estilo dela. Por isso a prova agora tem 20 pedidos de escrita (10 era pouco pra separar sorte de diferença) e um juiz de fora, e o caminho pra ganhar da base é o autopolimento (abaixo).

```bash
python prova_dificil.py --modelo ratex/xselo-0-5/v1 --juiz Qwen/Qwen3.8-27B --salvar avaliacoes/0.5-dificil.json
python prova_dificil.py --modelo pasta/nova --rival pasta/antiga --partes pareado   # versão nova x antiga
```

No Colab, é a célula **6b**.

## Rodar de graça no Kaggle

O Kaggle (do Google) dá ~30h de GPU por semana de graça: duas T4 que, juntas, seguram o Xselo 31B em 4 bits (o modelo é dividido entre as duas placas sozinho). É mais devagar que a A100 do Colab, mas não gasta unidade. O notebook [`notebooks/xselo_no_kaggle.ipynb`](notebooks/xselo_no_kaggle.ipynb) traz o passo a passo: subir o `.zip` do Xselo como dataset privado, anexar a Gemma 4 pelos *Models* do Kaggle (sem baixar 60 GB), escolher **GPU T4 x2** e dar *Run All*. Tem conversa com memória, busca, `/foto <link>` e o bot do Discord.

Nas T4, a Gemma roda com a parte que não é 4 bits em float32 (`XSELO_DTYPE=float32`), porque em float16 ela pode estourar os números; nelas não existe bf16 de verdade.

## Autopolimento: `python autopolimento.py`

A arena ensina o seu gosto, mas cada partida demora. O autopolimento faz o trabalho pesado sozinho:

1. pra cada pedido (os de `dados_autopolimento/pedidos.txt`, mais perguntas do próprio dataset, nunca as das provas), o Xselo responde **4 vezes**, com um pouco de sorteio;
2. um juiz lê as 4 e escolhe a **melhor** e a **pior**. Faz isso duas vezes, com as respostas em ordens diferentes, e só vale quando as duas leituras concordam;
3. melhor × pior vira um par de preferência em `dados_preferencia/auto.txt`, que o polimento (DPO) usa junto com os pares escritos à mão e os da arena.

```bash
python autopolimento.py --modelo /content/saida/xselo-0-5-gemma31b --pedidos 100
```

O progresso fica salvo num `.json`: se a sessão cair, é só rodar de novo que ele continua de onde parou. No Colab, é a célula **5b**, e o polimento com todos os pares é a **5c**. O modelo polido ganha o número de pares no nome (ex.: `xselo-0-5-gemma31b-polido-95pares`), pra não confundir as rodadas.

Um cuidado: o juiz do autopolimento é a Gemma pura, e o da prova é a Qwen. Assim, se a nota do pareado subir, é porque o Xselo melhorou pra um juiz que não participou do treino, e não porque ele aprendeu a agradar o próprio corretor.

## Polimento por preferência (DPO): `python treinar_dpo.py`

Depois de treinado, o Xselo pode ser **polido com as suas correções**. Cada vez que ele errar numa conversa, anote em `dados_preferencia/` a conversa, a resposta ruim que ele deu e a resposta certa:

```
Pessoa: quem é o ser mais poderoso de touhou?
Ruim: O ser mais poderoso é a Luna Child, a deusa da lua...
Boa: Não existe um ranking oficial, mas a candidata mais citada é a Hecatia...
```

Dá pra incluir o começo da conversa antes do par (linhas `Pessoa:`/`Xselo:`), pra ensinar comportamento no meio do papo, tipo aceitar correção. O treino ensina o modelo a preferir as respostas boas, sempre comparando com o modelo de antes, pra ele não esquecer o que já sabia.

```bash
python treinar_dpo.py --modelo ratex/xselo-0-3/v1          # salva em ratex/xselo-0-3/v1-dpo
```

No Colab, a célula **5c** faz isso sozinha depois do treino e do autopolimento (dá pra desligar na célula 1).

**O que o primeiro teste mostrou** (12 pares, Xselo 0.3 de 1,5B, sem memória, na CPU):
- as probabilidades inverteram como deveriam: antes, o modelo achava a resposta ruim mais provável que a boa (−1,46 contra −2,37 por token); depois, a boa passou na frente (−1,66 contra −2,00);
- em perguntas que não estavam nos pares, ele parou de concordar com erro ("a capital é São Paulo" → "não, eu não errei") e parou de inventar um nome em "qual o personagem mais forte?";
- **mas não passou a dar as respostas certas**: continuou teimando que a Cirno é "deusa do sol". DPO ajusta o **jeito** de responder, não ensina **fatos**, e um modelo de 1,5B não sabe Touhou. Fato se resolve com memória e dataset; o DPO pede algo como 50–200 pares pra fazer efeito firme.

Por isso a versão polida do 0.3 não substituiu a oficial. A ferramenta fica pronta pra quando os pares crescerem, e ela rende mais no 32B, que já sabe muito mais.

**Pares da arena:** `python treinar_dpo.py --modelo ... --arena arena.json` transforma as partidas decididas da arena em pares (gravados em `dados_preferencia/arena.txt`, dá pra abrir e revisar). Respostas que citam fonte da busca (`[1]`, `[2]`) ficam de fora, porque no treino não há busca no prompt e o modelo aprenderia a citar fonte inventada.

## Busca na internet (SearXNG)

O Xselo pode **pesquisar na internet** antes de responder coisas atuais, usando o [SearXNG](https://github.com/searxng/searxng), um buscador aberto que junta Google, Bing, Wikipédia e outros sem rastrear ninguém (`nucleo/busca.py`).

- **Quando ele busca:** se a pergunta tem cara de coisa atual ("agora", "hoje", "notícia", "preço", "quem ganhou", um ano recente...), ou se você começa a mensagem com **`/buscar`**.
- Os melhores resultados entram no prompt com o link, e ele deve citar a fonte como `[1]`, `[2]`.
- `iniciar_searxng()` baixa, configura e liga um SearXNG só pra aquele computador (~1–2 min na primeira vez, num ambiente Python separado pra não bagunçar o resto).

```bash
python gerar.py --chat --busca                           # liga um SearXNG local sozinho
python gerar.py --chat --busca http://localhost:8888     # usa um SearXNG que já está rodando
```

No Colab, a célula **6c** liga a busca, e a conversa (célula 7) já usa. Se preferir rodar o SearXNG pelo Docker no seu PC, lembre de liberar o formato JSON no `settings.yml` (`search: formats: [html, json]`).

## Arena às cegas: o ranking

Pra saber se o Xselo está bom **no seu gosto**, e não só na prova automática: você faz uma pergunta, dois competidores respondem como **A** e **B** sem você saber quem é quem, e você escolhe a melhor. Cada escolha atualiza uma **nota Elo**, igual ranking de xadrez (`nucleo/arena.py`).

- Competidores: o **Xselo polido**, o **Xselo antes do polimento** e o **modelo base puro**. Todos dividem o mesmo modelo base na memória (não gasta GPU a mais) e usam o mesmo prompt, memória e busca, então a única diferença entre eles é o nosso treino.
- `/placar` mostra o ranking; o placar e todas as partidas ficam salvos (no Colab, no seu Drive, em `Ratex/arena.json`), então dá pra continuar em outro dia.
- Se nenhuma das duas respostas prestar, responda `n` e escreva a certa: ela vira a "Boa" contra as duas.
- As partidas viram pares de preferência sozinhas no próximo polimento (`--arena`, célula 5b).

No Colab é a célula **7b**.

## A memória de consulta (RAG)

Um modelo pequeno não decora todos os fatos de Touhou só com o LoRA. Por isso, antes de responder, o Xselo **consulta o próprio dataset** (`nucleo/memoria.py`), procurando em cada parágrafo e cada diálogo de `dataset.txt`, `dados_extras/` e `dados_gerais/`. Os trechos mais relevantes entram no prompt como "anotações do caderno".

A busca junta dois jeitos:

- **por significado:** um modelo pequeno de embeddings multilíngue (`intfloat/multilingual-e5-small`, ~120 MB, baixado na primeira vez e guardado em cache). Ele entende que "o ser mais **poderoso** de Touhou" é a mesma pergunta que "a mais **forte**", e que "quem **pisou** na Lua" é "o primeiro homem a **pisar**". Antes, a busca só por palavra errava esses casos, e o modelo acabava inventando;
- **por palavra** (BM25): boa pra nomes próprios e números ("Nazrin", "Touhou 6").

- Foi a peça que mais subiu a nota de Touhou: de 25 para 83,3.
- A busca **não devolve nada** quando o assunto não está nas anotações (ex.: "capital da França"). Assim ele não é confundido por um trecho parecido e responde com o que o modelo base sabe.
- **Texto novo colado em `dados_extras/` já vale na hora**, sem retreinar.
- Vem ligada no `gerar.py` e no `avaliar.py`; para desligar, use `--sem-memoria` ou `--memoria nao`.
- **Corrigiu o Xselo numa conversa?** Anote a pergunta e a resposta certa num `.txt` em `dados_extras/` ou `dados_gerais/`. A memória passa a usar na hora, e o próximo treino aprende. Foi assim com `dados_gerais/05_correcoes_e_curiosidades.txt`: Luna Child, Hecatia, teorema de Tales, Vantablack, raiva.

Com a memória por significado, a 0.3 tirou 88,9 na prova (antes, 91,7). O ponto perdido é a pergunta da Marisa: a resposta nova continua certa ("ela leva pra casa e diz que devolve quando morrer"), mas não usa as palavras-chave que a correção automática procura.

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
├── dados_prosa/           # 0.4/0.5: escrita caprichada, 303 conversas (crônica, poema, carta, conselho...)
├── treinar.py             # v1: treina o micro-Transformer do zero -> ratex/xselo-0-1/v1/
├── treinar_lora.py        # 0.2/0.3: treina o LoRA em cima do modelo base -> ratex/xselo-0-X/v1/
├── gerar.py               # conversa / gera texto (híbrido mais novo se existir, senão v1)
├── treinar_dpo.py         # polimento por preferência (DPO) com dados_preferencia/
├── dados_preferencia/     # pares "Ruim/Boa" tirados das conversas
├── avaliar.py             # prova fixa com nota por categoria (touhou, geral, matemática)
├── autopolimento.py       # 0.5: o Xselo responde várias vezes, o juiz escolhe, vira par de DPO
├── dados_autopolimento/   # pedidos que o autopolimento usa (nenhum das provas)
├── prova_dificil.py       # 0.5: prova com juiz (pegadinhas, correções, contas, prosa, pareado)
├── bot_discord.py         # 0.5: o Xselo no Discord
├── avaliacoes/            # relatórios da prova de cada versão
├── nucleo/
│   ├── modelo.py          # arquitetura Transformer (embeddings, atenção, FFN, LayerNorm, logits)
│   ├── tokenizador.py     # tokenizador por caractere (padrão) e BPE leve (opcional)
│   ├── pasta_modelo.py    # salvar/carregar no formato de pasta estilo Hugging Face
│   ├── dados_chat.py      # transforma o dataset em conversas + system prompts do Xselo
│   ├── matematica.py      # 0.3: exercícios de matemática com passo a passo e resposta exata
│   ├── memoria.py         # memória de consulta (RAG): significado + palavra nos trechos do dataset
│   ├── busca.py           # busca na internet via SearXNG (e liga um SearXNG local sozinho)
│   ├── arena.py           # arena às cegas com ranking Elo
│   ├── preferencias.py    # pares de dados_preferencia/ e da arena
│   ├── lembrancas.py      # 0.5: memória de conversa longa (fatos por pessoa + resumo)
│   ├── nuvem.py           # acha o Xselo e a base nos arquivos do Kaggle
│   └── hibrido.py         # versões, bases, carrega base + LoRA e gera respostas
├── ratex/xselo-0-1/v1/    # O MODELO
│   ├── pytorch_model.bin      # pesos (state_dict do PyTorch)
│   ├── config.json            # vocab_size, n_embd, n_head, n_layer, block_size, ... + dados do treino
│   ├── vocab.json             # tokenizador (tipo + tokens [+ merges do BPE])
│   ├── generation_config.json # temperatura/top-k/top-p padrão do gerar.py
│   └── README.md              # model card
├── ratex/xselo-0-2/v1/    # (depois do treino) adaptador LoRA + tokenizador + ratex_config.json
├── ratex/xselo-0-3/v1/    # adaptador LoRA (fp16) + tokenizador + ratex_config.json + model card
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
- [x] Treinar a 0.3 (Qwen2.5-0.5B e depois Qwen2.5-1.5B, na CPU)
- [x] RAG: o Xselo consulta o próprio dataset antes de responder
- [x] 0.4: lote de prosa, pesos por fonte, respostas com vários parágrafos, prova de escrita
- [x] Treinar a 0.4 no Qwen 32B no Colab (A100)
- [x] DPO com as correções das conversas (`treinar_dpo.py`, célula 5b do notebook)
- [ ] Juntar 50–200 pares de preferência e polir o 32B
- [x] Busca na internet via SearXNG como ferramenta do Xselo
- [x] Arena às cegas com ranking Elo (Xselo × Xselo sem polimento × Qwen puro)
- [x] Transformar as partidas da arena em pares de preferência automaticamente
- [x] Memória por significado (embeddings) junto com a busca por palavra
- [x] Mais prosa: de 105 pra 303 conversas (0.5)
- [x] Base nova de 2026: Gemma 4 31B (ou Qwen3.8-27B), com visão (0.5)
- [x] Enxergar imagens (`/imagem`, `/foto`, anexo no Discord)
- [x] Bot do Discord (`bot_discord.py`)
- [x] Treinar a 0.5 no Colab e rodar a prova difícil
- [x] Autopolimento: pares de preferência gerados sozinhos, com juiz (`autopolimento.py`)
- [ ] Rodada 2 da 0.5: autopolimento + polimento + prova com juiz de fora (Qwen3.8-27B)
- [ ] Segurar o conhecimento geral da base no LoRA (lr menor, mais conversas gerais)
- [x] Memória de conversa longa: fatos por pessoa + resumo das falas antigas (`nucleo/lembrancas.py`)
- [x] Prova mais difícil, com juiz (`prova_dificil.py`)
- [ ] Rodar no PC (RTX 5060 Ti 16 GB): Gemma 4 12B quantizada (GGUF/Ollama)

---

Touhou Project é uma obra de ZUN / Team Shanghai Alice. Este é um projeto de fã, sem fins comerciais.
