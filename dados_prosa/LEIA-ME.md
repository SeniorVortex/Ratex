# dados_prosa/

O lote que ensina o Xselo a **escrever bem**. É o coração da versão 0.4: no treino, cada conversa daqui entra **3 vezes por época** (os assuntos gerais entram 2 vezes, e o Touhou 0,6).

| arquivo | o que ensina |
|---|---|
| `01_escrita_criativa.txt` | crônica, conto, poema, carta, descrição, microconto |
| `02_explicar_bonito.txt` | ciência e ideias explicadas com precisão e ritmo |
| `03_vida_e_conversa.txt` | conselho, sentimento, papo sério, sem autoajuda barata |
| `04_cultura_e_mundo.txt` | games, anime, futebol, festa junina, mitologia, história |
| `05_reescrita_e_estilo.txt` | melhorar texto, resumir, mudar o tom, escrever pra públicos diferentes |
| `06_touhou_em_prosa.txt` | Touhou contado com capricho |
| `07_respostas_curtas.txt` | ser breve e bom, papo rápido, conversas com várias trocas |

## Formato

```
== Conversas: título da seção ==

Pessoa: escreve uma crônica sobre fila de banco
Xselo: Primeiro parágrafo da resposta.

Segundo parágrafo, que continua a mesma resposta.

Terceiro parágrafo.

Pessoa: próxima conversa (a linha em branco antes do "Pessoa:" separa as conversas)
Xselo: ...
```

- **Uma resposta pode ter vários parágrafos.** Um parágrafo que vem depois de uma fala do Xselo e não começa com `Pessoa:` continua a resposta.
- **Linhas coladas** (sem linha em branco) também continuam a fala: serve pra versos de poema.
- **Conversa com várias trocas**: coloque as falas seguidas, **sem** linha em branco entre elas:

  ```
  Pessoa: me conta uma piada
  Xselo: ...
  Pessoa: essa foi ruim
  Xselo: ...
  ```

## Como escrever um bom exemplo

O modelo aprende o **jeito**, então cada resposta aqui deve ser o texto que você gostaria que ele escrevesse:

- **concreto em vez de abstrato:** "deixou o café esfriar sem tocar na xícara" em vez de "estava triste";
- **ritmo variado:** frase longa que respira, seguida de uma curta;
- **sem clichê de robô:** nada de "é importante ressaltar", "em resumo", "espero ter ajudado";
- **tamanho certo:** pergunta simples, resposta curta; pedido de texto, resposta caprichada;
- **fatos certos:** o modelo repete o que aprende, inclusive os erros.

Qualquer `.txt` novo nesta pasta entra no treino da 0.4 automaticamente.
