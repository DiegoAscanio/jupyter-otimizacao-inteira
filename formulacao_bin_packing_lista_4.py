"""
exercicio 1:

problema bin packing de m caixas, n itens e mn possibilidades 
de transporte dos itens nas caixas, totalizando n + mn variáveis.

vamos considerar n = 5, 5 objetos, m = 5, até 5 caixas, onde cada caixa
tem capacidade C = 10, e os objetos têm pesos:

objetos | 1 | 2 | 3 | 4 | 5 |
w_i     | 3 | 5 | 6 | 4 | 2 |

A formulação genérica é:

    min z = sum(y_j)_(j = 1)^(m)
    subject to:
    sum(x_ij)_(j = 1)^(m) = 1, for all i in {1, ..., n} % que o item i deve estar em apenas uma caixa
    sum(w_i * x_ij)_(i = 1)^(n) <= C y_j , for all j in {1, ..., m} % que a soma dos pesos dos itens na caixa j não pode exceder o peso C da capacidade caixa j se ela for usada
    y_j in {0, 1}
    x_ij in {0, 1}

"""

import numpy as np

w = np.array(
    [3, 5, 6, 4, 2]
) # pesos dos objetos

C = 10 # capacidade das caixas

n = m = len(w) # número de objetos e caixas

# restricao 1: sum(x_ij)_(j = 1)^(m) = 1, for all i in {1, ..., n}
# matriz de coeficientes da restrição 1

A = np.zeros((n, n + n*m)) # variaveis y e x

for i in range(n):
    for j in range(m):
        A[i, n + n*i + j] = 1

b = np.ones(n) # vetor de termos independentes da restrição 1

# restricao 2: -c y_j + sum(w_i * x_ij)_(i = 1)^(n) + s_j = 0 , for all j in {1, ..., m}

# vamos adicionar m colunas para as variáveis de folga s_j na matriz A
A = np.hstack((A, np.zeros((m, m))))
_, numero_variaveis = A.shape

# vamos adicionar m linhas nulas para as m novas restrições na matriz A
A = np.vstack(
    (A, np.zeros((m, numero_variaveis)))
)

# nas m ultimas linhas temos que representar a matriz -Cy nos indices das
# variáveis y_j, que são os primeiros n

A[n: n + m, :n] = -C * np.eye(m)

for j in range(m):
    for i in range(n):
        A[n + j, n + n * i + j] = w[i]

A[n:, -m:] = np.eye(m)  # adicionando as variáveis de folga s_j

# variaveis de excesso

b = np.hstack((b, np.zeros(m)))  # vetor de termos independentes da restrição 2

# agora vamos montar a funcao objetivo

c = np.zeros(numero_variaveis)
# as m primeiras variáveis são as quantidades de caixa y_j que devem ser minimizadas
c[:m] = 1
