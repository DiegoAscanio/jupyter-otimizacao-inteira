from scipy.optimize import linprog, milp, LinearConstraint
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
    c : np.ndarray
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
    res = linprog(c, A_ub=A, b_ub=b, method='highs')
    return avaliar_resultado(res)

def resolve_PPI(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray
) -> tuple:
    """
    Resolve o problema de Programação Linear Inteira (PPI)
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
    res = milp(c, constraints=LinearConstraint(A, b, b), method='highs')
    return avaliar_resultado(res)

def constroi_subproblema(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    indices_para_afixar : np.ndarray,
    valores_afixados : np.ndarray
) -> tuple:
    # preparar o subproblema para resolução
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
    # retorna o subproblema
    return A, b, c

def iguais_e_indices(
    x_tilde : np.ndarray,
    x_barra : np.ndarray,
    epsilon : float = 1e-4
) -> tuple:
    """
    Identifica os valores iguais e seus indices entre dois arrays
    """
    mask = np.isclose(x_tilde, x_barra, atol=epsilon)
    indices, *_ = np.where(mask)
    valores = x_barra[mask]
    return valores, indices

def fix_and_optimize(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    x_tilde : np.ndarray,
    indices_de_otimizacao: np.ndarray
):
    """
    Heurística fix and optimize
    
    Recebe um problema de programação linear inteiro (PLI)
    e sua solução incumbente x_tilde.

    Aplica-se a heurística fix and optimize para resolver o problema.

    Argumentos:
        A : np.ndarray - Matriz dos coeficientes das restrições
        b : np.ndarray - Vetor dos limites das restrições
        c : np.ndarray - Vetor dos coeficientes da função objetivo
        x_tilde : np.ndarray - Solução incumbente do problema original
        indices_de_otimizacao : np.ndarray - Lista de Indices de
        particionamento para otimizacao (e afixamento dos que não
        pertencem) das variáveis
    Retorna:
        factivel: bool - Indica se a solução é factível
        z: valor da função objetivo da resolução final
        x: solução do problema
        snapshots : dict - Dicionário com os snapshots das soluções
        intermediárias
    """
    m, n = A.shape

    factivel,*_  = resolve_PPL(A, b, c)
    # early return se a versão relaxada não for factível
    if not factivel:
        return False, None, None, None

    steps = 0
    z_tilde = c @ x_tilde
    snapshots = {
        steps: {
            'x': x_tilde,
            'z': z_tilde
        }
    }
    for i_T in indices_de_otimizacao:
        indices_para_afixar = np.setdiff1d(
            np.arange(n), i_T, assume_unique=True
        )
        valores_afixados = x_tilde[indices_para_afixar]
        A_tilde, b_tilde, c_tilde = constroi_subproblema(
            A, b, c, indices_para_afixar, valores_afixados
        )
        factivel, x_tilde, z_tilde = resolve_PPI(
            A_tilde, b_tilde, c_tilde
        )
        if not factivel:
            raise Exception(
                'Exceção inesperada durante a resolução do subproblema inteiro'
            )
        steps += 1
        snapshots[steps] = {
            'x': x_tilde,
            'z': z_tilde
        }
    return True, z_tilde, x_tilde, snapshots
