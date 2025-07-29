from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import numpy as np

def inteiros_e_indices(
    x : np.ndarray,
    epsilon : float = 1e-4
) -> tuple:
    """
    Identifica os valores inteiros e seus indices em um array
    """
    mask = np.isclose(x, np.round(x), atol=epsilon)
    indices, *_ = np.where(mask)
    valores = x[mask]
    return valores, indices

def fracionarios_e_indices(
    x : np.ndarray,
    epsilon : float = 1e-4
) -> tuple:
    """
    Identifica os valores fracionários e seus indices em um array
    """
    mask = ~np.isclose(x, np.round(x), atol=epsilon)
    indices, *_ = np.where(mask)
    valores = x[mask]
    return valores, indices

def avaliar_resultado(
    res
) -> tuple:
    """
    Avalia o resultado da otimização
    Parâmetros:
        res : OptimizeResult - Resultado da otimização
    Retorna:
        tuple:
            factível : bool - Indica se a solução é factível
            x : np.ndarray - Solução do problema
            z : float - Valor da função objetivo
    """
    if res.success:
        return True, res.x, res.fun
    else:
        return False, None, None

def resolve_PPL(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    all_binary = True
) -> tuple:
    """
    Resolve o problema de Programação Linear (PPL) relaxado
    Parâmetros:
        A : np.ndarray - Matriz dos coeficientes das restrições
        b : np.ndarray - Vetor dos limites das restrições
        c : np.ndarray - Vetor dos coeficientes da função objetivo
    Retorna:
        tuple:
            factível : bool - Indica se a solução é factível
            x : np.ndarray - Solução do problema
            z : float - Valor da função objetivo
    """
    bounds = (0, None)
    if all_binary:
        bounds = (0, 1)
    res = linprog(c, A_eq=A, b_eq=b, bounds = bounds, method='highs')
    return avaliar_resultado(res)

def resolve_subproblema_inteiro(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    indices_para_afixar : np.ndarray,
    valores_afixados : np.ndarray,
    all_binary = True
) -> tuple:
    # preparar o subproblema para resolução
    n = len(c)
    
    for _, i in enumerate(indices_para_afixar):
        # adicionar uma restrição do tipo x_i <= valor_afixado
        A = np.hstack((A, np.zeros((A.shape[0], 1)))) # coluna de zeros
        A = np.vstack((A, np.zeros(A.shape[1]))) # linha de zeros
        A[-1, i] = 1  # afixar a variável i
        A[-1, -1] = 1 # variavel de folga
        b = np.append(b, valores_afixados[_])  # adicionar o valor afixado
        # adicionar uma restrição do tipo x_i >= valor_afixado
        A = np.hstack((A, np.zeros((A.shape[0], 1)))) # coluna de zeros
        A = np.vstack((A, np.zeros(A.shape[1]))) # linha de zeros
        A[-1, i] = 1  # afixar a variável i
        A[-1, -1] = -1 # variavel de excesso
        b = np.append(b, valores_afixados[_])  # adicionar o valor afixado
        # adicionar novas variaveis na função objetivo
        c = np.append(c, np.zeros(2))  # duas variáveis de folga e excesso
    # resolver o subproblema inteiro
    lb = [-np.inf] * len(c)
    ub = [np.inf] * len(c)

    if all_binary:
        lb[0:n] = [0] * n
        ub[0:n] = [1] * n
    
    res = milp(
        c,
        constraints=LinearConstraint(
            A, b, b
        ),
        integrality = np.ones_like(c),
        bounds = Bounds(lb, ub)
    )
    return *avaliar_resultado(res), A, b, c


def rens(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
):
    """
    Heurística construtiva RENS
    
    Recebe um problema de programação linear inteiro (PLI),
    resolve sua versão relaxada e aplica a heurística RENS
    encontrando um subconjunto F contido nas variáveis J do
    problema original tal que F seja as variáveis não-inteiras
    da relaxação linear do problema original.

    Afixa-se os valores das variáveis que ficaram inteiras na
    relaxação linear e resolve-se novamente o subproblema inteiro
    com essas variáveis afixadas.

    Argumentos:
        A : np.ndarray - Matriz dos coeficientes das restrições
        b : np.ndarray - Vetor dos limites das restrições
        c : np.ndarray - Vetor dos coeficientes da função objetivo
    Retorna:
        factivel: bool - Indica se a solução é factível
        z: valor da função objetivo da resolução final
        x: solução do problema
        F: indices das variáveis não inteiras
        x_F: valores intermediários das variáveis não inteiras.
        I: indices das variáveis inteiras
        x_I: valores intermediários das variáveis inteiras.
    """
    factivel, x_barra, z_barra = resolve_PPL(A, b, c)
    # early return se não for factível
    if not factivel:
        return None, None, None, None
    x_F, F = fracionarios_e_indices(x_barra)
    x_I, I = inteiros_e_indices(x_barra)
    # Resolve o subproblema inteiro
    factivel, x, z, A, b, c = resolve_subproblema_inteiro(
        A, b, c, I, x_I
    )
    return factivel, z, x, F, x_F, I, x_I, A, b, c
