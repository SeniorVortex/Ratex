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
