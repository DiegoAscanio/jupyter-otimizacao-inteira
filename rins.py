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
    res = linprog(c, A_eq=A, b_eq=b, method='highs')
    return avaliar_resultado(res)

def resolve_subproblema_inteiro(
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
    # resolver o subproblema inteiro
    res = milp(
        c,
        constraints=LinearConstraint(
            A, b, b
        ),
        integrality = np.ones_like(c)
    )
    return avaliar_resultado(res)

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

def rins(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    x_tilde : np.ndarray
):
    """
    Heurística de refinamento RINS
    
    Recebe um problema de programação linear inteiro (PLI)
    e sua solução incumbente x_tilde.

    resolve sua versão relaxada e aplica a heurística RINS
    encontrando um subconjunto J contido nas variáveis I do
    problema original tal que J seja composto pelas variáveis
    da versão linear relaxada que tenham valor igual aos da
    solução incumbente x_tilde.

    Afixa-se os valores das variáveis pertencentes a J e resolve-se 
    novamente o subproblema inteiro com essas variáveis afixadas.

    Argumentos:
        A : np.ndarray - Matriz dos coeficientes das restrições
        b : np.ndarray - Vetor dos limites das restrições
        c : np.ndarray - Vetor dos coeficientes da função objetivo
        x_tilde : np.ndarray - Solução incumbente do problema original
    Retorna:
        factivel: bool - Indica se a solução é factível
        z: valor da função objetivo da resolução final
        x: solução do problema
        J: indices das variáveis iguais à solução incumbente
        x_barra: solução intermediária do primeiro passo do RINS
    """
    factivel, x_barra, z_barra = resolve_PPL(A, b, c)
    # early return se não for factível
    if not factivel:
        return None, None, None, None
    # identificar os valores iguais e seus indices
    x_J, J = iguais_e_indices(x_tilde, x_barra)
    # resolver o subproblema inteiro com as variáveis afixadas
    factivel, x, z = resolve_subproblema_inteiro(
        A, b, c, J, x_J
    )
    return factivel, z, x, J, x_barra
