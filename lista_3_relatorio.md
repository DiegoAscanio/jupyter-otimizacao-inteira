<style scoped>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        margin-left: 1in;
        margin-right: 1in;
        margin-top: 0.25in;
        margin-bottom: 0.25in;
    }
    p {
        text-align: justify;
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

Suporte de cálculos e resoluções disponível em: [https://ascanio.dev/jupyter-otimizacao-inteira/lab/index.html?path=lista-3-otimizacao-inteira.ipynb](https://ascanio.dev/jupyter-otimizacao-inteira/lab/index.html?path=lista-3-otimizacao-inteira.ipynb)

---

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

---

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
& x\_1 + x\_3 \leq 1.00 \\\\
& x\_2 + x\_3 \leq 1.00 \\\\
& x\_1 + x\_2 + x\_3 + x\_4 \leq 2.00
\end{align}
$$

É possível observar que ao menos o último corte pôde ser extendido e fortalecido pela adição da variável $x\_3$.

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
    & x\_1 + x\_2 + x\_3 + x\_4 \leq 2.00 \\\\
\end{align}
$$

Adicionando estes cortes e resolvendo o problema, temos por solução:

\begin{align}
    z^{\*} &= 10.00 \\\\
    x^{\*} &= [ 1.00, \ 1.00, \ 0.00, \ 0.00, \ 0.00, \ 0.00, \ 0.00, \ 1.00, \ 1.00, \ 0.00, \ 9214.00 ] \\\\
    I^{\*} &= [ 3, \ 1, \ 6, \ 7, \ 10, \ 8, \ 0 ]
\end{align}

A variável $x_{10} = 9214.00$ é obtida pela adição de um corte big M no método dual simplex para transformar a base inicial como dual factível.
No caso, a solução ótima do problema é dada apenas por $ x^{\*} = [ 1.00, \ 1.00, \ 0.00, \ 0.00 ] $, com $ z^{\*} = 10.00$.

---

## Exercício 3

![](https://i.imgur.com/qm711y6.png)


a. $ x\_1 + x\_2 \leq 1 $

b. $ x\_2 + x\_3 + x\_4 \leq 2 $

c. $ x\_1 + x\_2 + x\_3 + x\_4 + x\_5 \leq 2 $

O corte mínimo da letra c é $ x\_1 + x\_3 + x\_4 \leq 2 $, entretanto, pelos valores fracionais do vetor x, $ x\_1 + x\_3 + x\_4 \simeq 0.89 $, o que implica que o corte mínimo não elimina a solução fracional do problema relaxado. Aí, fiz o lifting desse corte como ensinado no livro do Wolsey que me deu um corte forte e válido de $ x\_1 + x\_2 + x\_3 + x\_4 + x\_5 \leq 2 $ para o problema inteiro. Como $ x\_2 = x\_5 = 1 $, logo, os outros valores fracionários é que tendem a ser eliminados pelo corte.

---

## Exercício 4

![](https://i.imgur.com/DfGtJj8.png)

Resolvendo pelo método dos planos cortantes com cortes de gommory, apenas a aplicação do corte $ x\_4 + x\_5 \geq 1 $ foi suficiente para eliminar a solução fracional do problema relaxado, resultando na solução ótima inteira $X^{\*} = [5,\ 1,\ 7,\ 0,\ 1]$ com valor ótimo $Z^{\*} = 13$

Como $x\_4 = 6 - x\_1 - x\_2 $ e $ x\_5 = 9 - x\_1 - 3 x\_2 $, substituindo estes valores na restrição do corte, temos que:

$$
\begin{align}
-2 x\_1 - 4 x\_2 + 15 &\geq 1 \therefore \\\\
-2 x\_1 - 4 x\_2 &\geq -14 \therefore \\\\
2 x\_1 + 4 x\_2 &\leq 14 \therefore \\\\
\frac{x\_1}{2} + x\_2 &\leq \frac{7}{2}
\end{align}
$$

Para construir o politopo temos as retas de suas fronteiras:

1. $ y = \frac{x}{2} + 2 $
2. $ y = - x + 6 $
3. $ y = - \frac{x}{3} + 3 $
4. $ y = - \frac{x}{2} + \frac{7}{2} $

A área factível, sombreada em azul no gráfico abaixo, é a região delimitada pelas interseções das retas acima:

![](https://i.imgur.com/LP9f7PS.png)
