from scipy.optimize import milp, LinearConstraint, linprog
from gommory_cuts import _non_integer_mask
import numpy as np

def _solve_separation_problem(x_bar, weights, knapsack_capacity):
    """
    Solve the separation problem to find a minimum cover cut.
    Args: 
        x_bar: The solution vector from the LP relaxation.
        weights: The weights of the items.
        knapsack_capacity: The capacity of the knapsack.
    Returns:
        y: A binary vector indicating which items are included in the cover cut.
        cut: a boolean indicating if a cover cut was found.
    """
    # Step 1. build separation problem of kind
    # zeta = min (1 - x_bar)^T @ y subject to
    # weights @ y >= knapsack_capacity - 1 or
    # - weights @ y + s_variable <= - knapsack_capacity + 1
    n = len(x_bar)
    b = knapsack_capacity
    a = weights
    c = np.concatenate(
        (1 - x_bar, np.zeros(1))
    )
    constraints = LinearConstraint(
        A = np.concatenate(
            (-a, np.ones(1))
        ),
        lb = -b + 1,
        ub = -b + 1
    )
    integrality = np.array(
        [1] * n + [0]
    )  # y is binary, s_variable can be continuous, there's no difference
    # Step 2. solve the separation problem
    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality
    )
    if not res.success:
        raise ValueError("MILP solver failed to find a valid cover cut.")
    y = res.x[:-1]  # Exclude the slack variable
    zeta = res.fun

    return (y, True) if zeta < 1 else (None, False)

def solve_0_1_knapsack_with_cover_cuts(
    weights,
    values,
    knapsack_capacity,
    max_iterations=1000
):
    """
    Solve the 0-1 knapsack problem with cover cuts.
    
    Args:
        weights: List of weights of items.
        values: List of values of items.
        knapsack_capacity: Capacity of the knapsack.
        max_iterations: Maximum number of iterations for the separation algorithm.
    
    Returns:
        optimal_value: Optimal value of the knapsack.
        x_opt: Optimal solution vector.
        solution_type: 1 if the solution is optimal and feasible, 0 otherwise
        iterations: Number of iterations performed.
        steps: Dictionary containing the steps of the cuts performed
    """
    n = len(weights)
    steps = {}
    # transform weights and values into arrays
    weights = np.array(weights)
    values = np.array(values)

    # Step 0. Build the initial LP problem
    # First, we'll make sure that all variables are in [0, 1] interval
    A = np.hstack(
        (np.eye(n), np.eye(n))
    )
    # then we'll add the capacity constraint
    # first, add last column:
    A = np.hstack(
        (A, np.zeros((n, 1)))
    )
    # then, add last row
    A = np.vstack(
        (
            A, np.concatenate(
                (weights, np.zeros(n), np.ones(1))
            )
        )
    )
    # now build rhs b vector
    b = np.concatenate(
        (np.ones(n), [knapsack_capacity])
    )

    # now build the objective function c vector
    c = np.concatenate(
        (-values, np.zeros(n + 1))
    )

    iterations = 0
    stop_criteria = False
    solution_type = np.nan  # nan means no solution found yet
    x_star = None
    optimal_value = -np.inf

    while not stop_criteria:
        # step 1. solve the LP relaxation
        res = linprog(
            c = c,
            A_eq = A,
            b_eq = b
        )

        # step 2. verify if x_star is integer
        x_candidate = res.x[:-1]  # Exclude the slack variable
        if not np.all(_non_integer_mask(x_candidate)):
            # If x_star is integer, we have found the optimal solution
            x_star = x_candidate
            optimal_value = -res.fun
            solution_type = 1
        # otherwise we try to find a minimum cover cut by solving the
        # separation problem
        else:
            y, cut_found = _solve_separation_problem(
                x_candidate,
                weights,
                knapsack_capacity
            )
            if cut_found:
                # Step 3. Add the cover cut to the LP problem
                # The cover cut is of the form y^T @ x <= |y| - 1
                # where |y| is the number of items in the cover cut
                cover_cardinality = np.sum(y) # as y is binary
                # 3.1 add a new column to A
                m, n = A.shape
                A = np.hstack(
                    (A, np.zeros((m, 1)))
                )
                # 3.2 add a new row to A
                new_row = np.zeros((1, n + 1))
                new_row[0, 0:len(y)] = y
                new_row[0, -1] = 1 # slack variable
                A = np.vstack(
                    (A, new_row)
                )
                # 3.3 add a new row to b
                b = np.concatenate(
                    (b, [cover_cardinality - 1])
                )
                # 3.4 add a new slack variable to c
                c = np.concatenate(
                    (c, [0])
                )
                    
            else:
                # there's no valid cover cut to approach us to the optimal
                # integer feasible solution, so we stop with no feasible
                # solution found
                solution_type = 0
        # Step 4. update the steps dictionary
        steps[iterations] = {
            'A': A,
            'b': b,
            'c': c
        }
        # Step 5. increment the iterations counter
        iterations += 1
        # Step 6. update the stop criteria
        stop_criteria = solution_type != np.nan or iterations >= max_iterations
    # Step 7. return the found results
    return (
        optimal_value,
        x_star,
        solution_type,
        iterations,
        steps
    )
