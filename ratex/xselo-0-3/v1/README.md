---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
language: pt
tags: [ratex, touhou, lora]
---

# ratex/xselo-0-3/v1

**Xselo 0.3**, a IA da Ratex. É um adaptador **LoRA** em cima do [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), com **memória de consulta** (RAG) sobre o dataset. É especialista em Touhou, mas também conversa sobre assuntos gerais e faz conta com passo a passo, tudo no tom descontraído do Xselo.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `adapter_model.safetensors` | pesos do LoRA (float16, 37 MB) |
| `adapter_config.json` | configuração do LoRA (PEFT): rank 16, alpha 32, todas as camadas lineares |
| `ratex_config.json` | versão, modelo base, system prompt, geração padrão e dados do treino |
| `tokenizer.json`, `chat_template.jinja`... | tokenizador do Qwen com o chat template |

O modelo base (~3 GB) é baixado do Hugging Face na primeira execução.

## Treino

- **Dados:** 930 conversas.
  - 289 diálogos escritos à mão: 200 de Touhou e 89 de assuntos gerais.
  - 341 perguntas de leigo geradas a partir dos parágrafos do `dataset.txt`.
  - 300 exercícios de matemática com passo a passo e resposta exata.
- **LoRA:** 18,5 M de parâmetros treináveis (1,18% do modelo). A loss só conta nas respostas do Xselo.
- **Máquina:** CPU de 4 núcleos, sem GPU. Foi **1 época**, em 130 min (lote efetivo 8, lr 2e-4). A época 1 foi escolhida por ter sido a melhor na base de 0,5B; depois dela o modelo passa a decorar.
- **Loss de validação:** 2,40 → **1,71**. A mesma receita no Qwen 0.5B foi de 2,85 → 2,00.

## Resultados na prova fixa (`avaliar.py`, 36 perguntas)

Nota por palavra-chave (0 a 100):

| modelo | Touhou | geral | matemática | total |
|---|---:|---:|---:|---:|
| xselo 0.1 (feito do zero, 3 M) | 16,7 | 8,3 | 0 | 8,3 |
| Qwen 0.5B puro | 0 | 66,7 | 33,3 | 33,3 |
| xselo 0.3 no Qwen 0.5B + memória | 83,3 | 83,3 | 83,3 | 83,3 |
| Qwen 1.5B puro + memória | 91,7 | 91,7 | 83,3 | 88,9 |
| xselo 0.3 no Qwen 1.5B, sem memória | 41,7 | 83,3 | 91,7 | 72,2 |
| **xselo 0.3 no Qwen 1.5B + memória** (este) | **91,7** | **91,7** | **91,7** | **91,7** |

A nota por palavra-chave é generosa: ela conta acerto quando a palavra certa aparece, mesmo com erro junto.

Correção rigorosa, lendo resposta por resposta (acertos de 12):

| modelo | Touhou | geral | matemática |
|---|---:|---:|---:|
| xselo 0.3 no Qwen 0.5B + memória | 10 | 6 | 10 |
| **xselo 0.3 no Qwen 1.5B + memória** | **11** | **10** | **11** |

Os erros que sobraram no 1.5B: disse que a névoa vermelha foi da Youmu (foi da Remilia), respondeu "sete" para os lados do hexágono (são seis), errou um sinal numa equação e inventou dois números em detalhes (a massa de Júpiter e o tamanho do Pacífico). Todas as respostas estão em `avaliacoes/`.

O que cada peça faz:

- **A memória (RAG)** entrega os fatos de Touhou e das anotações gerais prontos no contexto.
- **O LoRA** traz a matemática com passo a passo e a personalidade do Xselo.
- **A base maior** (1.5B) segura o conhecimento geral: o 0.5B com LoRA chegava a dizer que a capital da França é Lisboa.

Na honestidade: no 1.5B, o Qwen puro com a mesma memória já chega perto (88,9 contra 91,7). Nessa escala, a maior parte do ganho vem de **base maior + memória**. O LoRA acrescenta a matemática (83 → 92) e respostas mais diretas, no formato e no tom do Xselo.

## Limitações

- **Sem a memória ele ainda inventa** fatos de Touhou com confiança. Deixe a memória ligada (é o padrão).
- Detalhes numéricos fora das anotações (massas, distâncias, populações) podem vir errados.
- Roda na CPU, mas devagar (alguns tokens por segundo); com GPU fica instantâneo.

## Uso

```bash
python gerar.py --chat                              # conversa (usa a 0.3, com memória de consulta)
python gerar.py --chat --sem-memoria                # sem RAG
python avaliar.py --modelo 0.3 --mostrar            # prova fixa
python treinar_lora.py --base qwen-1.5b --epocas 1 --batch-size 1 --acumular 8   # refaz este treino
```

Touhou Project é obra de ZUN / Team Shanghai Alice. Projeto de fã, sem fins comerciais.
