"""
Gerador de exercícios de matemática com resolução passo a passo, no jeito do Xselo.

Tudo é calculado com `fractions.Fraction`, então as respostas são exatas por
construção (nada de "a IA errou a conta no dataset"). Cada exercício termina com
"Resposta: ...", o que facilita a avaliação automática em avaliar.py.

    >>> from nucleo.matematica import gerar_exercicios
    >>> gerar_exercicios(3, seed=1)   # [(pergunta, resposta), ...]
"""

from __future__ import annotations

import random
from fractions import Fraction as F

PERSONAGENS = ["a Reimu", "a Marisa", "a Cirno", "a Sakuya", "a Youmu", "a Sanae", "a Nazrin", "a Meiling",
               "a Patchouli", "a Yuyuko", "a Aya", "a Nitori", "a Chen", "a Ran", "a Koishi"]
ABERTURAS = ["Bora:", "Tranquilo, vamos lá:", "Fácil, olha só:", "Deixa comigo:", "Vamos por partes:", "Simbora:"]


def fmt(x) -> str:
    """Número no formato brasileiro: 1234.5 -> '1.234,5'."""
    x = F(x)
    negativo = x < 0
    x = abs(x)
    if x.denominator == 1:
        inteiro, dec = x.numerator, ""
    else:
        # só chega aqui com decimais exatos de até 4 casas (os geradores garantem isso)
        for casas in range(1, 5):
            if (x * 10**casas).denominator == 1:
                break
        else:
            raise ValueError(f"{x} não tem representação decimal curta")
        n = int(x * 10**casas)
        inteiro, dec = divmod(n, 10**casas)
        dec = "," + str(dec).rjust(casas, "0")
    s = f"{inteiro:,}".replace(",", ".") + dec
    return ("-" if negativo else "") + s


def reais(x) -> str:
    x = F(x)
    s = fmt(x)
    if x.denominator != 1:
        inteiro, _, dec = s.partition(",")
        s = f"{inteiro},{dec.ljust(2, '0')}"
    return f"R$ {s}"


def _abrir(r: random.Random) -> str:
    return r.choice(ABERTURAS)


# ---------------------------------------------------------------- geradores
# cada um recebe um random.Random e devolve (pergunta, resposta)

def soma(r):
    a, b = r.randint(12, 999), r.randint(12, 999)
    q = r.choice([f"quanto é {a} + {b}?", f"quanto dá {a} mais {b}?", f"soma {a} com {b} pra mim"])
    ca, cb = a // 100 * 100, b // 100 * 100
    passos = ""
    if ca and cb:
        passos = (f" Separando as centenas: {ca} + {cb} = {ca + cb}, e o resto: {a - ca} + {b - cb} = "
                  f"{a - ca + b - cb}. Juntando: {ca + cb} + {a - ca + b - cb} = {a + b}.")
    return q, f"{_abrir(r)} {a} + {b} = {fmt(a + b)}.{passos} Resposta: {fmt(a + b)}."


def subtracao(r):
    a = r.randint(50, 2000)
    b = r.randint(10, a - 1)
    q = r.choice([f"quanto é {a} - {b}?", f"quanto dá {a} menos {b}?", f"tira {b} de {a}, quanto sobra?"])
    return q, (f"{_abrir(r)} {a} - {b} = {fmt(a - b)}. Pra conferir, é só somar de volta: "
               f"{fmt(a - b)} + {b} = {a}. Resposta: {fmt(a - b)}.")


def multiplicacao(r):
    a, b = r.randint(11, 99), r.randint(3, 19)
    q = r.choice([f"quanto é {a} x {b}?", f"quanto é {a} vezes {b}?", f"{a} * {b} = ?", f"multiplica {a} por {b}"])
    if b >= 10:
        passos = f"{a} × {b} = {a} × 10 + {a} × {b - 10} = {a * 10} + {a * (b - 10)} = {a * b}"
    else:
        d, u = a // 10 * 10, a % 10
        passos = f"{a} × {b} = {d} × {b} + {u} × {b} = {d * b} + {u * b} = {a * b}"
    return q, f"{_abrir(r)} dá pra quebrar a conta em pedaços: {passos}. Resposta: {fmt(a * b)}."


def divisao(r):
    b = r.randint(2, 25)
    c = r.randint(2, 60)
    if r.random() < 0.7:
        a = b * c
        q = r.choice([f"quanto é {a} dividido por {b}?", f"{a} / {b} = ?", f"divide {a} por {b}"])
        return q, f"{_abrir(r)} {a} ÷ {b} = {c}, porque {c} × {b} = {a}. Resposta: {c}."
    resto = r.randint(1, b - 1)
    a = b * c + resto
    q = f"quanto é {a} dividido por {b}? tem resto?"
    return q, (f"{_abrir(r)} {b} cabe {c} vezes em {a}, porque {c} × {b} = {b * c}. Sobra {a} - {b * c} = {resto}. "
               f"Então dá {c} e sobra {resto}. Resposta: {c}, resto {resto}.")


def porcentagem(r):
    p = r.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    base = r.choice([20, 40, 60, 80, 100, 120, 150, 200, 240, 300, 400, 500, 800, 1000])
    v = F(p, 100) * base
    q = r.choice([f"quanto é {p}% de {base}?", f"calcula {p} por cento de {base}"])
    return q, (f"{_abrir(r)} porcentagem é só uma fração de 100. {p}% = {p}/100. "
               f"Então {p}/100 × {base} = {fmt(v)}. Resposta: {fmt(v)}.")


def desconto(r):
    preco = r.choice([40, 50, 60, 80, 90, 100, 120, 150, 180, 200, 250, 300])
    p = r.choice([10, 15, 20, 25, 30, 40, 50])
    item = r.choice(["uma camiseta", "um tênis", "um jogo", "um fumo da Cirno", "uma mochila", "um fone"])
    d = F(p, 100) * preco
    q = f"{item} custa {reais(preco)} e está com {p}% de desconto. quanto fica?"
    return q, (f"{_abrir(r)} primeiro o desconto: {p}% de {fmt(preco)} = {fmt(d)}. Depois tira do preço: "
               f"{fmt(preco)} - {fmt(d)} = {fmt(preco - d)}. Resposta: {reais(preco - d)}.")


def regra_de_tres(r):
    n1 = r.randint(2, 6)
    unit = F(r.choice([50, 75, 100, 125, 150, 200, 250]), 100)
    n2 = r.randint(n1 + 1, 15)
    coisa = r.choice(["pães", "coxinhas", "latas de refri", "cadernos", "pacotes de figurinha", "bolinhos"])
    q = f"se {n1} {coisa} custam {reais(unit * n1)}, quanto custam {n2}?"
    return q, (f"{_abrir(r)} regra de três. Primeiro o preço de um: {fmt(unit * n1)} ÷ {n1} = {fmt(unit)}. "
               f"Depois multiplica: {fmt(unit)} × {n2} = {fmt(unit * n2)}. Resposta: {reais(unit * n2)}.")


def equacao(r):
    x = r.randint(-9, 15)
    a = r.choice([2, 3, 4, 5, 6, 7, 8, 9])
    b = r.randint(-20, 30)
    c = a * x + b
    sinal = f"+ {b}" if b >= 0 else f"- {-b}"
    q = r.choice([f"resolve {a}x {sinal} = {c}", f"quanto vale x em {a}x {sinal} = {c}?"])
    passo1 = f"passa o {b} pro outro lado subtraindo: {a}x = {c} - {b} = {c - b}" if b >= 0 else \
             f"passa o {-b} pro outro lado somando: {a}x = {c} + {-b} = {c - b}"
    return q, (f"{_abrir(r)} {passo1}. Agora divide por {a}: x = {c - b} ÷ {a} = {x}. "
               f"Conferindo: {a} × {x if x >= 0 else f'({x})'} {sinal} = {c}. Resposta: x = {x}.")


def area(r):
    tipo = r.choice(["retangulo", "quadrado", "triangulo", "circulo"])
    if tipo == "retangulo":
        a, b = r.randint(2, 30), r.randint(2, 30)
        return (f"qual a área de um retângulo de {a} m por {b} m?",
                f"{_abrir(r)} área de retângulo é base vezes altura: {a} × {b} = {a * b}. Resposta: {a * b} m².")
    if tipo == "quadrado":
        a = r.randint(2, 25)
        return (f"qual a área e o perímetro de um quadrado de lado {a} cm?",
                f"{_abrir(r)} a área é lado vezes lado: {a} × {a} = {a * a} cm². O perímetro é a soma dos 4 lados: "
                f"4 × {a} = {4 * a} cm. Resposta: área {a * a} cm², perímetro {4 * a} cm.")
    if tipo == "triangulo":
        b, h = r.randint(2, 20) * 2, r.randint(2, 20)
        return (f"qual a área de um triângulo de base {b} cm e altura {h} cm?",
                f"{_abrir(r)} triângulo é metade do retângulo: base × altura ÷ 2 = {b} × {h} ÷ 2 = "
                f"{b * h} ÷ 2 = {b * h // 2}. Resposta: {b * h // 2} cm².")
    raio = r.randint(1, 10)
    a = F(314, 100) * raio * raio
    return (f"qual a área de um círculo de raio {raio} m? pode usar pi = 3,14",
            f"{_abrir(r)} área do círculo é pi × raio²: 3,14 × {raio}² = 3,14 × {raio * raio} = {fmt(a)}. "
            f"Resposta: {fmt(a)} m².")


def media(r):
    n = r.randint(3, 5)
    nums = [r.randint(2, 10) for _ in range(n)]
    m = F(sum(nums), n)
    while (m * 100).denominator != 1:
        nums[-1] += 1
        m = F(sum(nums), n)
    lista = ", ".join(map(str, nums[:-1])) + f" e {nums[-1]}"
    q = r.choice([f"qual a média de {lista}?", f"tirei as notas {lista}. qual minha média?"])
    return q, (f"{_abrir(r)} soma tudo e divide pela quantidade. Soma: {' + '.join(map(str, nums))} = {sum(nums)}. "
               f"São {n} números: {sum(nums)} ÷ {n} = {fmt(m)}. Resposta: {fmt(m)}.")


def potencia(r):
    if r.random() < 0.5:
        b, e = r.choice([2, 3, 5, 10]), r.randint(2, 6)
        if b == 2:
            e = r.randint(2, 10)
        mults = " × ".join([str(b)] * e)
        return (f"quanto é {b} elevado a {e}?",
                f"{_abrir(r)} potência é multiplicar o número por ele mesmo várias vezes: {mults} = {b ** e}. "
                f"Resposta: {fmt(b ** e)}.")
    n = r.randint(2, 30)
    return (f"qual a raiz quadrada de {n * n}?",
            f"{_abrir(r)} é o número que vezes ele mesmo dá {n * n}. Como {n} × {n} = {n * n}, a raiz é {n}. "
            f"Resposta: {n}.")


def fracoes(r):
    a, b = F(r.randint(1, 5), r.randint(2, 8)), F(r.randint(1, 5), r.randint(2, 8))
    op = r.choice(["+", "-"]) if a != b else "+"
    if op == "-" and a < b:
        a, b = b, a
    res = a + b if op == "+" else a - b
    den = a.denominator * b.denominator
    q = f"quanto é {a.numerator}/{a.denominator} {op} {b.numerator}/{b.denominator}?"
    na, nb = a.numerator * b.denominator, b.numerator * a.denominator
    resp = f"{res.numerator}/{res.denominator}" if res.denominator != 1 else str(res.numerator)
    topo = na + nb if op == "+" else na - nb
    simpl = f", que simplificando fica {resp}" if (topo, den) != (res.numerator, res.denominator) else ""
    return q, (f"{_abrir(r)} pra somar ou subtrair fração, os denominadores têm que ser iguais. Usando {den} como "
               f"denominador: {na}/{den} {op} {nb}/{den} = {topo}/{den}{simpl}. Resposta: {resp}.")


def troco(r):
    quem = r.choice(PERSONAGENS)
    preco = F(r.randint(3, 95) * 5, 10) if r.random() < 0.6 else F(r.randint(5, 180))
    nota = next(n for n in (10, 20, 50, 100, 200) if n > preco)
    item = r.choice(["um bolinho", "um livro", "chá", "um amuleto", "um pepino", "uma vassoura nova", "um lanche"])
    q = f"{quem} comprou {item} de {reais(preco)} e pagou com uma nota de {reais(nota)}. qual o troco?"
    return q, f"{_abrir(r)} troco é o que pagou menos o preço: {fmt(nota)} - {fmt(preco)} = {fmt(nota - preco)}. " \
              f"Resposta: {reais(nota - preco)}."


def conversao(r):
    tipo = r.choice(["km", "horas", "kg", "litros", "dias"])
    if tipo == "km":
        v = F(r.randint(1, 40), 2)
        return (f"quantos metros tem {fmt(v)} km?",
                f"{_abrir(r)} 1 km tem 1.000 metros, então {fmt(v)} × 1.000 = {fmt(v * 1000)}. Resposta: {fmt(v * 1000)} m.")
    if tipo == "horas":
        h, m = r.randint(1, 9), r.choice([0, 15, 30, 45])
        txt = f"{h} horas" + (f" e {m} minutos" if m else "")
        return (f"quantos minutos têm {txt}?",
                f"{_abrir(r)} cada hora tem 60 minutos: {h} × 60 = {h * 60}" + (f", mais {m} = {h * 60 + m}" if m else "") +
                f". Resposta: {h * 60 + m} minutos.")
    if tipo == "kg":
        g = r.randint(1, 40) * 250
        return (f"quantos quilos são {fmt(g)} gramas?",
                f"{_abrir(r)} 1 kg tem 1.000 g, então divide por 1.000: {fmt(g)} ÷ 1.000 = {fmt(F(g, 1000))}. "
                f"Resposta: {fmt(F(g, 1000))} kg.")
    if tipo == "litros":
        l = F(r.randint(1, 20), 2)
        return (f"quantos mililitros tem {fmt(l)} litros?",
                f"{_abrir(r)} 1 litro tem 1.000 mL: {fmt(l)} × 1.000 = {fmt(l * 1000)}. Resposta: {fmt(l * 1000)} mL.")
    s = r.randint(2, 12)
    return (f"quantos dias têm {s} semanas?",
            f"{_abrir(r)} cada semana tem 7 dias: {s} × 7 = {s * 7}. Resposta: {s * 7} dias.")


def problema_historia(r):
    quem = r.choice(PERSONAGENS)
    tipo = r.choice(["doacao", "livros", "sapos", "dividir"])
    if tipo == "doacao":
        ini, por_dia, dias = r.randint(0, 10), r.randint(2, 9), r.randint(3, 12)
        return (f"a Reimu tinha {ini} moedas na caixa de doação e ganhou {por_dia} moedas por dia durante {dias} dias. "
                f"com quantas ela ficou?",
                f"{_abrir(r)} ela ganhou {por_dia} × {dias} = {por_dia * dias} moedas nesses dias. Somando com as "
                f"{ini} que já tinha: {ini} + {por_dia * dias} = {ini + por_dia * dias}. Milagre no santuário! "
                f"Resposta: {ini + por_dia * dias} moedas.")
    if tipo == "livros":
        total, por_visita = r.randint(40, 300), r.randint(2, 9)
        visitas = total // por_visita
        return (f"a biblioteca da Patchouli tem {total} livros e a Marisa pega {por_visita} emprestado por visita. "
                f"quantas visitas completas até sobrar menos de {por_visita}?",
                f"{_abrir(r)} é uma divisão: {total} ÷ {por_visita} = {visitas} com resto {total % por_visita}. "
                f"Então são {visitas} visitas completas e sobram {total % por_visita} livros. Coitada da Patchouli. "
                f"Resposta: {visitas} visitas.")
    if tipo == "sapos":
        por_hora, horas, fugiram = r.randint(2, 9), r.randint(2, 8), r.randint(0, 5)
        total = por_hora * horas
        fugiram = min(fugiram, total)
        return (f"a Cirno congela {por_hora} sapos por hora. em {horas} horas, e com {fugiram} sapos fugindo, "
                f"quantos ela congelou?",
                f"{_abrir(r)} {por_hora} × {horas} = {total} sapos no total, menos os {fugiram} que fugiram: "
                f"{total} - {fugiram} = {total - fugiram}. A mais forte ataca de novo. Resposta: {total - fugiram} sapos.")
    pessoas = r.randint(2, 8)
    fatias = pessoas * r.randint(1, 4)
    return (f"{quem} tem uma pizza com {fatias} fatias pra dividir igualmente entre {pessoas} pessoas. "
            f"quantas fatias cada um come?",
            f"{_abrir(r)} {fatias} ÷ {pessoas} = {fatias // pessoas}. " +
            ("E se a Yuyuko estiver entre essas pessoas, esquece a conta. " if r.random() < 0.3 else "") +
            f"Resposta: {fatias // pessoas} fatias.")


GERADORES = [soma, subtracao, multiplicacao, divisao, porcentagem, desconto, regra_de_tres, equacao,
             area, media, potencia, fracoes, troco, conversao, problema_historia]


def gerar_exercicios(n: int, seed: int = 0) -> list[tuple[str, str]]:
    r = random.Random(seed)
    vistos, saida = set(), []
    tentativas = 0
    while len(saida) < n and tentativas < n * 20:
        tentativas += 1
        pergunta, resposta = r.choice(GERADORES)(r)
        if pergunta in vistos:
            continue
        vistos.add(pergunta)
        saida.append((pergunta, resposta))
    return saida
