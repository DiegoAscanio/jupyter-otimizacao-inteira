<style scoped>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        margin-left: 1in;
        margin-right: 1in;
        margin-top: 0.25in;
        margin-bottom: 0.25in;
    }
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    @media print {
        img {
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
            break-inside: avoid;
            display: block;
            margin: auto; /* centraliza a imagem na página */
        }
        figure, table, pre, blockquote {
            page-break-inside: avoid;
            break-inside: avoid;
        }
        body {
            overflow-wrap: break-word;
            word-wrap: break-word;
        }
    }
</style>

<script async src="https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS_CHTML"></script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
        displayMath: [ ['$$', '$$'], ['\\[', '\\]'] ],
        processEscapes: true
    }
});
</script>

# Terceira Lista de Exercícios de Otimização Inteira

Autor: [Diego Ascânio Santos](mailto:ascanio@cefetmg.br)

![](https://i.imgur.com/aAmyuEt.png)

## 1. a)

**Solução ótima para o problema relaxado:**

$$
\begin{align}
    z^{\*} &= 28.00 \\\\
    x^{\*} &= [ 4.67, \\ 0.00, \ 0.00, \ 0.00, \ 14331.33 ] \\\\
    I^{\*} &= [ 0, \\ 4 ]
    \end{align}
$$

A variável \\(x\_4 = 14331.33\\) é obtida pela adição de um corte big M no método dual simplex para transformar a base inicial como dual factível.

No caso, a solução ótima do problema é dada apenas por $ x^{\*} = [ 4.67, \ 0.00, \ 0.00 ] $, com $ z^{\*} = 28.00$

## 1. b)

Seguindo o procedimento de cortes de chavatal gommory como ensinado no livro do bazaraa, montei um algoritmo de _cutting planes_ e pedindo para o método resolver, foram aplicados dois cortes que removeram soluções fracionárias não factíveis do problema inteiro abordado:

$$
\begin{align}
    & 0.33 x\_1 + 0.67 x\_2 + 0.67 x\_3 \ge 0.33 \\\\
    & 0.50 x\_1 + x\_3 + 0.50 x\_5 \ge 0.50 \\\\
\end{align}
$$

Após a adição destes cortes uma solução ótima inteira foi encontrada:

$$
\begin{align}
    z^{\*} &= 27.00 \\\\
    x^{\*} &= [ 3, \ 1, \ 0  ]
\end{align}
$$

![](https://i.imgur.com/AdkLHpH.png)

## 2. a)

![](https://i.imgur.com/wOEUKZk.png)

**Coberturas mínimas:**

$$
\begin{align}
    & 5 x\_1 + 7 x\_ 3 > 9 \\\\
    & 4 x\_2 + 7 x\_ 3 > 9 \\\\
    & 5 x\_1 + 4 x\_ 2 + 2 x\_4 > 9
\end{align}
$$

**Cortes de cobertura mínimos:**
$$
\begin{align}
    & x\_1 + x\_3 \leq 1 \\\\
    & x\_2 + x\_3 \leq 1 \\\\
    & x\_1 + x\_2 + x\_4 \leq 2
\end{align}
$$

Fazendo o lifting dos cortes propostos:

$$
\begin{align}
& x\_0 + x\_2 \leq 1.00 \\\\
& x\_1 + x\_2 \leq 1.00 \\\\
& x\_0 + x\_1 + x\_3 \leq 2.00
\end{align}
$$

É possível observar que infelizmente não foi possível adicionar aos cortes mínimos encontrados quaisquer outras variáveis para deixar os cortes ainda mais fortes.

O código abaixo mostra as tentativas de _lifting_ realizadas:

```python

extended_cuts = r'''$$
\begin{align}
'''
# primeira cobertura minima: x_0 + x_2 <= 1
minimal_covers = [
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 1]
]
for y in minimal_covers:
    beta = np.sum(y) - 1
    extended_y = _sequential_lifting(
        y,
        beta,
        weights,
        knapsack_capacity
    )
    cut = display_cut(
        np.concatenate(
            (extended_y, [1, beta])
        ),
        type = 'cover'
    )
    extended_cuts += cut
    extended_cuts += r'\\' + '\n'
extended_cuts += r'''\end{align}
$$
'''

```

## 2. b)

**Solução ótima para o problema relaxado:**

$$
\begin{align}
    z^{\*} &= 11.00 \\\\
    x^{\*} &= [ 1.00, \ 0.50, \ 0.00, \ 1.00, \ 0.00, \ 0.00, \ 0.50, \ 1.00, \ 0.00, \ 9213.50 ] \\\\
    I^{\*} &= [ 3, \ 1, \ 6, \ 7, \ 9, \ 0 ]
\end{align}
$$

A variável $x\_9 = 9213.50$ é obtida pela adição de um corte big M no método dual simplex para transformar a base inicial como dual factível.
No caso, a solução ótima do problema é dada apenas por $ x^{\*} = [ 1.00, \ 0.50, \ 0.00, \ 1.00 ] $, com $ z^{\*} = 11.00$.

## 2. c)

Os cortes de cobertura que eliminam a solução fracional do problema relaxado são: 

$$
\begin{align}
    & x\_0 + x\_1 + x\_3 \leq 2.00 \\\\
    & x\_0 + x\_2 \leq 1.00
\end{align}
$$

Adicionando estes cortes e resolvendo o problema, temos por solução:

\begin{align}
    z^{\*} &= 10.00 \\\\
    x^{\*} &= [ 1.00, \ 1.00, \ 0.00, \ 0.00, \ 0.00, \ 0.00, \ 0.00, \ 1.00, \ 1.00, \ 0.00, \ 0.00, \ 9214.00 ] \\\\
    I^{\*} &= [ 5, \ 1, \ 6, \ 7, \ 11, \ 2, \ 8, \ 0 ]
\end{align}

A variável $x_{11} = 9214.00$ é obtida pela adição de um corte big M no método dual simplex para transformar a base inicial como dual factível.
No caso, a solução ótima do problema é dada apenas por $ x^{\*} = [ 1.00, \ 1.00, \ 0.00, \ 0.00 ] $, com $ z^{\*} = 10.00$.
