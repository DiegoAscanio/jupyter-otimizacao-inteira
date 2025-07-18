from scipy.optimize import linprog
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

def rens(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    epsilon : float = 1e-4,
    max_nodes : int = 1000
):
    snapshots = {
    }

    # 2. Senão a gente tenta encontrar soluções inteiras factíveis
    # para o problema inteiro através da heurística RENS até que 
    # tal solução seja encontada ou o critério de parada seja atingido
    nodes = 0
    stop_condition = False
    while not stop_condition:
        # 1. Resolve o PPL relaxado
        factivel, x, z = resolve_PPL(A, b, c)
        snapshots[nodes] = {
            'A': A.copy(),
            'b': b.copy(),
            'c': c.copy(),
            'm': A.shape[0],
            'n': A.shape[1],
            'x': x.copy() if x is not None else None,
            'z': z if z is not None else None,
            'factivel': factivel
        }
        # 2.1 Identifica os valores inteiros e seus indices
        valores_inteiros, indices_inteiros = inteiros_e_indices(
            x, epsilon
        )
        # 2.3 Identifica os valores não inteiros e seus indices
        valores_fracionarios, indices_fracionarios = fracionarios_e_indices(
            x, epsilon
        )
        # 2.3 Fixa os valores inteiros adicionando restrições do tipo
        # x_i <= valores[i] e x_i >= valores[i] para cada i pertencente
        # aos indices inteiros filtrados
        for i, val in zip(indices_inteiros, valores_inteiros):
            m, n = A.shape
            # Adiciona duas colunas para as variáveis de folga e excesso
            # em A
            A = np.hstack((A, np.zeros((m, 2))))
            # Adiciona duas linhas para as restrições de fixação
            A = np.vstack((A, np.zeros((2, n + 2))))
            # Define as restrições de fixação
            # primeira: x_i <= val
            A[-2,  i] = 1
            A[-2, -2] = 1
            b = np.hstack((b, val))
            # segunda: x_i >= val
            A[-1,  i] = 1
            A[-1, -1] = -1
            b = np.hstack((b, val))
            # Adiciona as duas novas variáveis de folga e excesso
            # na função objetivo
            c = np.hstack((c, 0, 0))
        # 2.4 limita os valores fracionários x_i ao intervalo 
        # floor(x_i) <= x_i <= ceil(x_i) para cada i pertencente
        # aos indices fracionários filtrados
        for i, val in zip(indices_fracionarios, valores_fracionarios):
            m, n = A.shape
            # Adiciona duas colunas para as variáveis de folga e excesso
            # em A correspondentes ao limitante da variavel fracionaria x_i
            A = np.hstack((A, np.zeros((m, 2))))
            # Adiciona duas linhas para as restrições de fixação
            A = np.vstack((A, np.zeros((2, n + 2))))
            # Define as restrições de fixação
            # primeira: x_i <= ceil(val)
            A[-2,  i] = 1
            A[-2, -2] = 1
            b = np.hstack((b, np.ceil(val)))
            # segunda: x_i >= floor(val)
            A[-1,  i] = 1
            A[-1, -1] = -1
            b = np.hstack((b, np.floor(val)))
            # Adiciona as duas novas variáveis de folga e excesso
            # na função objetivo
            c = np.hstack((c, 0, 0))
        nodes += 1
        # Atualiza a condição de parada
        stop_condition = nodes >= max_nodes or len(indices_fracionarios) == 0 or not factivel
    # retorna a solução encontrada ao final da execução
    return z, x, nodes, snapshots
