import numpy as np
from dual_simplex import dual_simplex, _dual_feasible_basis

def _non_integer_mask(b, epsilon=1e-6):
    """
    Returns a boolean mask indicating which elements of x are not integers.
    """
    return np.abs(b - np.round(b)) > epsilon

def _greatest_non_integer_index(b, epsilon=1e-6):
    """
    Returns the index of the greatest non-integer element in b.
    If all elements are integers, returns None.
    """
    non_integer_mask = _non_integer_mask(b, epsilon)
    if np.all(~non_integer_mask):
        return None
    b_prime = np.abs(b - np.round(b))
    b_prime[~non_integer_mask] = -np.inf # Ignore integer elements
    return np.argmax(b_prime)

def _compute_fractionals(arr : np.ndarray | np.float64) -> np.ndarray | np.float64:
    """
    Computes the fractional parts of the elements in arr.
    """
    return arr - np.floor(arr)

def _prepare_A_and_b_to_apply_gommory_cuts(A, b, I):
    A_I = A[:, I]
    A_I_inv = np.linalg.inv(A_I)
    return A_I_inv @ A, A_I_inv @ b

def _gommorys_dual_fractional_cut(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    I: np.ndarray | list,
    J: np.ndarray | list,
    epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list, bool]:
    """
    Computes a Gommory's dual fractional cut for the given linear program
    and add it to its constraints.
    
    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        c (np.ndarray): Cost vector.
        I (np.ndarray | list): Indices of basic variables.
        J (np.ndarray | list): Indices of non-basic variables.
        epsilon (float): Tolerance for non-integer check.
        
    Returns:
        A_cutted (np.ndarray): A matrix with the cut added.
        b_cutted (np.ndarray): b vector with the cut added.
        c_cutted (np.ndarray): c vector with the cut added.
        I_cutted (np.ndarray | list): Updated indices of basic 
        variables including the excess variable.
        cut_performed (bool): Whether a cut was performed.
    """
    m, n = A.shape
    # Step 1. Create a copy of A, b, and c
    A_cutted = A.copy()
    b_cutted = b.copy()
    c_cutted = c.copy()

    # Step 2. Find the index of the greatest non-integer element in b
    index = _greatest_non_integer_index(b, epsilon)
    if index is None:
        # No non-integer elements found, no cut needs to be performed
        return A_cutted, b_cutted, c_cutted, I, False

    # Step 3. With its index, compute the fractional parts of the row
    rhs_fraction = _compute_fractionals(b[index])
    lhs_fraction = np.zeros((1, n + 1))
    lhs_fraction[0, J] = _compute_fractionals(A[index, J])
    lhs_fraction[0, -1] = -1 # For the excess variable

    # Step 4. Add the new constraint cut to A_cutted, b_cutted, and c_cutted
    # the new constraint cut is in the form of 
    # lhs_fraction @ x >= rhs_fraction
    A_cutted = np.hstack([A_cutted, np.zeros((m, 1))])  # Add a column for the excess variable
    A_cutted = np.vstack([A_cutted, lhs_fraction])  # Add the new cut row
    b_cutted = np.append(b_cutted, rhs_fraction)  # Add the bound for the new cut
    c_cutted = np.hstack([c_cutted, 0])  # Add a cost of 0 for the excess variable
    I_cutted = np.append(I, n)  # Add the index of the excess variable to the basis I
    cut_performed = True

    # Step 5. Return the modified A, b, c, I and the cut_performed flag
    return A_cutted, b_cutted, c_cutted, I_cutted, cut_performed

def cutting_planes(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    I: np.ndarray | list,
    epsilon: float = 1e-6,
    max_iterations: int = int(1e4)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list]:
    """
    Solves an integer linear program using Gommory's cuts and the dual simplex method.
    
    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
        c (np.ndarray): Cost vector.
        I (np.ndarray | list): Indices of basic variables.
        epsilon (float): Tolerance for non-integer check.
        
    Returns:
        z_star (np.float64): Optimal value.
        x_star (np.ndarray): Solution vector.
        I_star (np.ndarray | list): Indices of basic variables in the optimal solution.
        iters: int: Number of iterations performed.
        solution_type (int): Type of solution (1. optimal and unique,
        2. optimal and multiple, 3. dual unbounded).
        steps (dict): steps taken during the solution process.
    """
    # initialize data
    steps = {}
    iters = 0
    continuity = True

    while continuity:
        # Step 0. ensure the problem is dual feasible
        A, b, c, I, _ = _dual_feasible_basis(A, b, c, I)

        # Step 1. Solve the linear program using the dual simplex method
        z_star, x_star, _, I_star, dual_simplex_iters, solution_type, dual_simplex_steps = dual_simplex(A, b, c, I)
        n = len(c)
        J = np.setdiff1d(np.arange(n), I_star) # update the indices of non-basic variables

        if solution_type not in [1, 2]:
            # If the solution is unbounded or infeasible
            continuity = False
            continue

        # Otherwise, we try to perform a Gommory's cut at the result
        # obtained from the dual simplex method
        A, b = _prepare_A_and_b_to_apply_gommory_cuts(A, b, I_star)
        A, b, c, I, cut_performed = _gommorys_dual_fractional_cut(A, b, c, I_star, J, epsilon)

        # update steps with the current iteration data
        steps[iters] = {
            'z_star': z_star,
            'x_star': x_star,
            'I_star': I_star,
            'dual_simplex_iters': dual_simplex_iters,
            'solution_type': solution_type,
            'dual_simplex_steps': dual_simplex_steps,
            'cut_performed': cut_performed
        }

        # Check if an integer solution was found. When no cut is performed,
        # it means that the current solution is integer feasible.
        iters += 1
        continuity = cut_performed and iters < max_iterations

    # Return the final solution
    return z_star, x_star, I_star, iters, solution_type, steps
