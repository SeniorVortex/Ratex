# dados_preferencia/

Correções pro polimento por preferência (DPO, `treinar_dpo.py`). Cada par diz: "nessa
conversa, a resposta **Boa** é melhor que a **Ruim**".

```
Pessoa: quem é o ser mais poderoso de touhou?
Ruim: O ser mais poderoso é a Luna Child, a deusa da lua...
Boa: Não existe um ranking oficial, mas a candidata mais citada é a Hecatia...
```

- **Ruim:** de preferência, a resposta **de verdade** que o Xselo deu (copiada da conversa).
- **Boa:** a resposta que ele deveria ter dado, no tom dele.
- Pra ensinar comportamento no meio do papo (aceitar correção, não teimar), coloque o
  começo da conversa antes, com `Pessoa:` e `Xselo:` alternando; o par vale pra última
  fala da pessoa.
- Linha em branco + `Pessoa:` começa um par novo; respostas podem ter vários parágrafos.
- Inclua também casos em que **a pessoa está errada** e a resposta boa discorda com
  educação, senão o modelo aprende a concordar com tudo.

DPO ensina **jeito de responder**, não fatos. Pra fato novo, anote também em
`dados_extras/` ou `dados_gerais/`, que a memória de consulta usa na hora.

## Pares que vêm da arena

A arena às cegas (`nucleo/arena.py`, célula 7b do Colab) também vira par: quem ganhou é a
**Boa**, quem perdeu é a **Ruim**, e quando você responde `n` e escreve a resposta certa, ela é
a Boa contra as duas. `python treinar_dpo.py --arena arena.json` grava esses pares em
`arena.txt` (aqui nesta pasta) e usa no polimento. O `arena.txt` é refeito a cada vez a partir
do `arena.json`, então pra corrigir um par, corrija a escolha jogando mais partidas, ou copie o
par pra outro arquivo e edite lá.
