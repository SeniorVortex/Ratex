# dados_extras/

Cole aqui qualquer arquivo `.txt` com mais texto sobre Touhou (ou sobre o que
você quiser que o Xselo aprenda). Na hora do treino, o `treinar.py` junta
automaticamente o `dataset.txt` da raiz com **todos os `.txt` desta pasta**
(inclusive em subpastas), em ordem alfabética.

Dicas:

- Salve em UTF-8. Aspas curvas, travessões e afins são convertidos sozinhos.
- Para ensinar o formato de conversa, use uma fala por linha:

  ```
  Pessoa: quem é a Reimu?
  Xselo: A Reimu é a sacerdotisa preguiçosa e sem dinheiro do Santuário Hakurei...
  ```

  e deixe uma linha em branco entre uma conversa e outra.
- Uma resposta pode ter **vários parágrafos**: o parágrafo que vem depois de uma fala do
  Xselo e não começa com `Pessoa:` continua a resposta. Veja `dados_prosa/LEIA-ME.md`.
- Este `LEIA-ME.md` não entra no treino (só arquivos `.txt` entram).
