# dados_prosa/

O lote que ensina o Xselo a **escrever bem**: 303 conversas escritas à mão, com os fatos conferidos. Na versão 0.4 eram 105, e cada uma entrava 3 vezes por época; na 0.5, com quase o triplo de texto, cada conversa entra **2 vezes por época** (os assuntos gerais também 2, e o Touhou 0,5).

| arquivo | o que ensina |
|---|---|
| `01_escrita_criativa.txt` | crônica, conto, poema, carta, descrição, microconto |
| `02_explicar_bonito.txt` | ciência e ideias explicadas com precisão e ritmo |
| `03_vida_e_conversa.txt` | conselho, sentimento, papo sério, sem autoajuda barata |
| `04_cultura_e_mundo.txt` | games, anime, futebol, festa junina, mitologia, história |
| `05_reescrita_e_estilo.txt` | melhorar texto, resumir, mudar o tom, escrever pra públicos diferentes |
| `06_touhou_em_prosa.txt` | Touhou contado com capricho |
| `07_respostas_curtas.txt` | ser breve e bom, papo rápido, conversas com várias trocas |
| `08_cronicas_e_contos.txt` | crônicas do cotidiano, contos com final surpreendente, fábula, mistério, ficção científica |
| `09_poesia.txt` | verso livre, haicai, soneto, cordel, poema engraçado |
| `10_cartas_e_mensagens.txt` | e-mail, convite, pêsames, discurso, recado, dedicatória, resposta a cliente |
| `11_explicar_bonito_2.txt` | mais ciência e história: vacina, GPS, DNA, evolução, muro de Berlim... |
| `12_reescrita_e_edicao.txt` | clareza, correção gramatical explicada, mudar público e tom, resumir, encurtar |
| `13_papo_e_conselho.txt` | procrastinação, amizade, carreira, luto, timidez, dinheiro; com várias trocas |
| `14_touhou_em_prosa_2.txt` | Mansão Scarlet Devil, Moriya, Mokou e Kaguya, Satori e Koishi, spell cards... |
| `15_humor.txt` | piada, trocadilho, diálogo absurdo, paródia, horóscopo e crítica de mentira |
| `16_respostas_curtas_2.txt` | respostas de uma ou duas frases, pra ele não enrolar quando não precisa |

As perguntas das provas (`avaliar.py` e `prova_dificil.py`) ficam **de fora** daqui de propósito: senão a
prova mede decoreba, não escrita.

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
