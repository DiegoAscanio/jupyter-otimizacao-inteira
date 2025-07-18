from scipy.optimize import milp, LinearConstraint
import numpy as np

from rens import  resolve_PPL, avaliar_resultado

def indices_para_afixacao(
    x_relaxado : np.ndarray,
    x_incumbente : np.ndarray,
    epsilon : float = 1e-4
    ) -> np.ndarray:
    """
    Identifica os índices de variáveis a serem fixadas
    com base na solução relaxada e na solução incumbente.
    Parâmetros:
        x_relaxado : np.ndarray - Solução relaxada do PPL
        x_incumbente : np.ndarray - Solução incumbente do PPL
        epsilon : float - Tolerância para comparação de valores
    Retorna:
        indices: np.ndarray - Índices das variáveis a serem fixadas
    """
    mask = np.isclose(x_relaxado, x_incumbente, atol=epsilon)
    indices, *_ = np.where(mask)
    return indices

def rins(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    x_incumbente : np.ndarray,
    epsilon : float = 1e-4
) -> tuple:
    """
    Heurística RINS (Relaxation Induced Neighborhood Search)
    Parâmetros:
        A : np.ndarray - Matriz dos coeficientes das restrições
        b : np.ndarray - Vetor dos limites das restrições
        c : np.ndarray - Vetor dos coeficientes da função objetivo
        x_incumbente : np.ndarray - Solução incumbente do PPL
        epsilon : float - Tolerância para comparação de valores
    Retorna:
        tuple:
            factível : bool - Indica se a solução é factível
            x : np.ndarray - Solução do problema
            z : float - Valor da função objetivo
    """
    # Resolve o PPL relaxado
    res = resolve_PPL(A, b, c)
    factível, x_relaxado, z_relaxado = avaliar_resultado(res)

    if not factível:
        return False, None, None

    # Identifica os índices das variáveis a serem fixadas
    indices = indices_para_afixacao(x_relaxado, x_incumbente, epsilon)

    # Cria restrições no problema para afixar os valores destas variáveis
    for i in indices:
        m, n = A.shape
        # adiciona-se duas restricoes do tipo <= e >= (para igualdade)
        # por meio de duas colunas para as variáveis de folga e excesso
        A = np.hstack(
            (A, np.zeros((m, 2)))
        )
        # Adiciona-se a primeira linha para a variável de folga
        A = np.vstack(
            (A, np.zeros((1, n + 2)))
        )
        A[-1,  i] = 1
        A[-1, -2] = 1
        # Adiciona-se a segunda linha para a variável de excesso
        A = np.vstack(
            (A, np.zeros((1, n + 2)))
        )
        A[-1,  i] = 1
        A[-1, -1] = -1
        # Adiciona-se os valores de afixação no vetor b
        b = np.hstack((b, [x_incumbente[i], x_incumbente[i]]))
        # Adiciona-se duas colunas para as variáveis de folga e excesso
        c = np.hstack((c, [0, 0]))  # Coeficientes para as variáveis de folga e excesso
    # Resolve o PPI com as restrições de afixação
    res = milp(
        c,
        constraints = LinearConstraint(
            A, b, b
        ),
        integrality = [1] * A.shape[1],  # Todas as variáveis são inteiras
    )
    factível, x, z = avaliar_resultado(res)
    return factível, x, z
