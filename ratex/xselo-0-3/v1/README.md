---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
language: pt
tags: [ratex, touhou, lora]
---

# ratex/xselo-0-3/v1

**Xselo 0.3**, a IA da Ratex. É um adaptador **LoRA** em cima do [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), com **memória de consulta** (RAG) sobre o dataset. É especialista em Touhou, mas também conversa sobre assuntos gerais e faz conta com passo a passo, tudo no tom descontraído do Xselo.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `adapter_model.safetensors` | pesos do LoRA (float16, 18 MB) |
| `adapter_config.json` | configuração do LoRA (PEFT): rank 16, alpha 32, todas as camadas lineares |
| `ratex_config.json` | versão, modelo base, system prompt, geração padrão e dados do treino |
| `tokenizer.json`, `chat_template.jinja`... | tokenizador do Qwen com o chat template |

O modelo base (~1 GB) é baixado do Hugging Face na primeira execução.

## Treino

- **Dados:** 930 conversas.
  - 289 diálogos escritos à mão: 200 de Touhou e 89 de assuntos gerais (`dados_gerais/`).
  - 341 perguntas de leigo geradas a partir dos parágrafos do `dataset.txt`.
  - 300 exercícios de matemática com passo a passo e resposta exata (`nucleo/matematica.py`).
- **LoRA:** 8,8 M de parâmetros treináveis (1,75% do modelo). A loss só conta nas respostas do Xselo.
- **Máquina:** CPU de 4 núcleos, sem GPU. Foram 3 épocas (167 min), e o adaptador salvo é o da **época 1**, que teve a menor loss de validação (2,85 → **2,00**). Nas épocas 2 e 3 a validação piorou (2,05 e 2,15), porque o modelo começou a decorar.

## Resultados na prova fixa (`avaliar.py`, 36 perguntas)

Nota por palavra-chave (0 a 100):

| modelo | Touhou | geral | matemática | total |
|---|---:|---:|---:|---:|
| xselo 0.1 (feito do zero) | 16,7 | 8,3 | 0 | 8,3 |
| Qwen 0.5B puro | 0 | 66,7 | 33,3 | 33,3 |
| Qwen 0.5B puro + memória | 83,3 | 66,7 | 33,3 | 61,1 |
| xselo 0.3 sem memória | 25,0 | 50,0 | 83,3 | 52,8 |
| **xselo 0.3 + memória** (padrão) | **83,3** | **83,3** | **83,3** | **83,3** |

A nota por palavra-chave é **generosa**. Lendo as respostas do 0.3 + memória uma a uma, a nota rigorosa fica em **Touhou 10/12, geral 6/12 e matemática 10/12**. Várias respostas gerais trazem a palavra certa junto com um erro (ex.: "o maior planeta é Marte, seguido de Júpiter"). Todas as respostas estão em `avaliacoes/`.

O que cada peça faz:

- **A memória (RAG) é quem acerta Touhou.** Com ela, até o Qwen puro acerta os fatos.
- **O LoRA trouxe a matemática** (33 → 83) e **a personalidade** do Xselo.
- **O LoRA piorou um pouco o conhecimento geral.** O Qwen puro responde "Paris" para a capital da França; o 0.3 responde "Lisboa". Com 0,5B o modelo é frágil, e o jeito confiante do Xselo às vezes vira invenção confiante.

## Limitações

- É um modelo de 0,5B: **inventa com confiança** quando não sabe, principalmente em conhecimento geral fora das anotações.
- Às vezes conta errado a quantidade de números numa média, ou entra em repetição.
- O próximo passo é uma base maior (Qwen 1.5B em diante), que erra bem menos. Veja o roadmap no README principal.

## Uso

```bash
python gerar.py --chat --modelo 0.3                 # conversa, com memória de consulta
python gerar.py --chat --modelo 0.3 --sem-memoria   # sem RAG
python avaliar.py --modelo 0.3 --mostrar            # prova fixa
```

Touhou Project é obra de ZUN / Team Shanghai Alice. Projeto de fã, sem fins comerciais.
