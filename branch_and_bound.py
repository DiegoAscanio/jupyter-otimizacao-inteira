from scipy.optimize import linprog
import numpy as np

try:
    from graphviz import Digraph
except ImportError:
    class Digraph:
        def __init__(self, comment=''):
            self.comment = comment
            self.nodes = []
            self.edges = []

        def node(self, name, label, shape='box'):
            self.nodes.append((name, label, shape))

        def edge(self, from_node, to_node, label=''):
            self.edges.append((from_node, to_node, label))

        def render(self):
            return "Graph rendering not implemented for this mock class."

def generate_dot(node):
    lines = ["digraph G {", "node [shape=box];"]

    def recurse(n):
        if n is None:
            return
        name = f"N{n['name']}"
        z_sup = n.get("z_sup", "None")
        x = n.get("x", None)
        x_str = str(np.round(x, 2)) if x is not None else "N/A"
        label = f"{name}\\nP: {n['name']}\\nz_sup: {z_sup}\\nx: {x_str}"
        lines.append(f'"{name}" [label="{label}"];')

        if 'left' in n and n['left']:
            left_name = f"N{n['left']['name']}"
            var = n.get("branch_variable", "?")
            bound = n.get("lower_bound", "?")
            lines.append(f'"{name}" -> "{left_name}" [label="x_{var} <= {bound}"];')
            recurse(n['left'])

        if 'right' in n and n['right']:
            right_name = f"N{n['right']['name']}"
            var = n.get("branch_variable", "?")
            bound = n.get("upper_bound", "?")
            lines.append(f'"{name}" -> "{right_name}" [label="x_{var} >= {bound}"];')
            recurse(n['right'])

    recurse(node)
    lines.append("}")
    return "\n".join(lines)

def draw_bnb_tree(node, graph=None):
    if graph is None:
        graph = Digraph(comment='Árvore BnB')

    if node is None:
        return graph

    node_name = f"N{node['name']}"

    # Pega informações com verificação de existência
    z_sup = node.get('z_sup')
    z_sup_str = f"{z_sup:.2f}" if z_sup is not None and not np.isnan(z_sup) and not np.isinf(z_sup) else str(z_sup)

    x_vals = node.get('x')
    x_str = np.round(x_vals, 2) if x_vals is not None else "N/A"

    P_name = node['P']['name'] if 'P' in node and 'name' in node['P'] else 'N/A'

    label = f"{node_name}\\nP: {P_name}\\nz_sup: {z_sup_str}\\nx: {x_str}"

    graph.node(node_name, label, shape='box')

    # Processar filhos
    if 'left' in node and node['left']:
        var = node.get('branch_variable', '?')
        bound = node.get('lower_bound', '?')
        edge_label = f"x_{var} ≤ {bound}"
        draw_bnb_tree(node['left'], graph)
        graph.edge(node_name, f"N{node['left']['name']}", label=edge_label)

    if 'right' in node and node['right']:
        var = node.get('branch_variable', '?')
        bound = node.get('upper_bound', '?')
        edge_label = f"x_{var} ≥ {bound}"
        draw_bnb_tree(node['right'], graph)
        graph.edge(node_name, f"N{node['right']['name']}", label=edge_label)

    return graph

def gap(z_sup, z_star):
    """
    Calculate the gap between the upper and lower bounds.
    
    Parameters:
    - z_sup: Upper bound of the objective function.
    - z_star: Lower bound of the objective function.
    
    Returns:
    - The gap as a percentage of the upper bound.
    """
    g = np.clip(
        (z_sup - z_star) / np.abs(z_star),
        0,
        1
    )
    return g if not np.isnan(g) else np.inf

def check_integrality(x : np.ndarray, integrality):
    x_slice = x[integrality == 1]
    return np.all(np.isclose(x_slice, np.round(x_slice)))

def solve_lp(c, A, b):
    """
    Solve the linear programming problem:
        minimize c @ x
        subject to A @ x <= b
    using the simplex method.
    
    Parameters:
    - c: Coefficients for the objective function.
    - A: Coefficients for the inequality constraints.
    - b: Right-hand side of the inequality constraints.
    
    Returns:
    - z_star: Optimal value of the objective function.
    - x_star: Optimal solution vector.
    - status: Status of the optimization (0 for success).
    """
    res = linprog(c, A_eq=A, b_eq=b, method='highs')
    return res.fun, res.x, res.status

def _least_close_to_integer_index(x, integrality):
    """
    Find the index of the variable that is least close to an integer value.
    Arguments:
    x -- current solution vector
    integrality -- list of indices of variables that must take integer values
    Returns:
    The index of the variable that is least close to an integer value.
    """
    x_slice = x[integrality == 1]
    distances = np.abs(x_slice - np.round(x_slice))
    return np.argmax(distances)

def _first_non_integer_index(x, integrality):
    """
    Find the index of the first variable that is not an integer.
    Arguments:
    x -- current solution vector
    integrality -- list of indices of variables that must take integer values
    Returns:
    The index of the first variable that is not an integer.
    """
    x_slice = x[integrality == 1]
    for i, val in enumerate(x_slice):
        if not np.isclose(val, np.round(val)):
            break
    return i

def _random_non_integer_index(x, integrality):
    """
    Find a random index of a variable that is not an integer.
    Arguments:
    x -- current solution vector
    integrality -- list of indices of variables that must take integer values
    Returns:
    A random index of a variable that is not an integer.
    """
    x_slice = x[integrality == 1]
    non_integer_indices = np.where(~np.isclose(x_slice, np.round(x_slice)))[0]
    return np.random.choice(non_integer_indices)

def select_to_branch(x, integrality, strategy = 'lci'):
    """
    Find the index of the variable to branch according to the chosen strategy.
    Arguments:
    x -- current solution vector
    integrality -- list of indices of variables that must take integer values
    strategy -- strategy for selecting the variable to branch ('lci', 'fni', 'random')
    Returns:
    The index of the variable to branch.
    """
    if strategy == 'lci':
        return _least_close_to_integer_index(x, integrality)
    elif strategy == 'fni':
        return _first_non_integer_index(x, integrality)
    elif strategy == 'random':
        return _random_non_integer_index(x, integrality)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

def select_current_problem(L, strategy = 'best'):
    """
    Select the current problem to solve based on the chosen strategy.
    Arguments:
    L -- list of problems to solve
    strategy -- strategy for selecting the current problem ('dfs', 'bfs', 'best')
    Returns:
    The selected problem from the list L.
    """
    if strategy == 'dfs':
        return L.pop()
    elif strategy == 'bfs':
        return L.pop(0)
    elif strategy == 'best':
        # Select the problem with the best upper bound
        # for maximization problems as we're considering
        best_index = np.argmax([p['z_sup'] for p in L])
        return L.pop(best_index)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

def _create_subproblem(P, branch_index, bound, direction):
    """
    Create a subproblem by branching on the variable at branch_index.
    The bound is applied to the variable at branch_index in the specified direction.
    
    Parameters:
    - P: Current problem node.
    - branch_index: Index of the variable to branch on.
    - bound: Value to set for the variable at branch_index.
    - integrality: List of indices of variables that must take integer values.
    - direction: 'up' or 'down' indicating the direction of the branch.
    
    Returns:
    - A new problem dictionary representing the subproblem.
    """
    if direction == 'right':
        # for the right branch, we set the variable to be greater than
        # or equal to the bound, so, we'll add a new constraint with
        # an excess variable. we'll also set the bound to be the ceil
        coefficient = -1
        bound = np.ceil(bound)
    elif direction == 'left':
        # for the left branch, we set the variable to be less than
        # or equal to the bound, so, we'll add a new constraint with
        # a slack variable. we'll also set the bound to be the floor
        coefficient = 1
        bound = np.floor(bound)
    # x_branch_index (le / ge) (floor / ceil) bound
    new_A = np.hstack([
        P['A'],
        np.zeros((P['A'].shape[0], 1))  # add a new column for the new variable
    ])
    new_A = np.vstack([
        new_A,
        np.zeros((1, new_A.shape[1]))  # add a new row for the new constraint
    ])
    new_A[-1, branch_index] = 1
    new_A[-1, -1] = coefficient 
    new_b = np.hstack([
        P['b'],
        bound  # add the new bound
    ])
    new_c = np.hstack([
        P['c'],
        0  # the new variable does not contribute to the objective function
    ])
    subproblem = {
        'name': 2 * P['name'] + (2 if direction == 'right' else 1),
        'branch_variable': None,
        'integrality': np.hstack((P['integrality'], [0])),
        'lower_bound': -np.inf,
        'upper_bound': np.inf,
        'A': new_A,
        'b': new_b,
        'c': new_c,
        'z': -np.inf,  # initial z is set to -inf
        'z_sup': P['z_sup'],  # upper bound remains the same from parent
        'x': None,
        'status': None,
        'parent': P,
        'left': None,
        'right': None,
    }
    return subproblem

problems_to_solve = lambda x, y, z: x and y and z

def _bound(L, z_star):
    """
    Bound tree removing all active problems with superior limits inferior
    than the best incumbent solution. We'll also remove them from the BnB
    tree
    Args:
        L: list of active problems
        z_star: best incumbent solution value
    Returns:
        to_remain: Updated list of active problems
    """
    to_remain = list(
        filter(
            lambda x: x['z_sup'] > z_star,
            L
        )
    )
    to_leave = list(
        filter(
            lambda x: x['z_sup'] <= z_star,
            L
        )
    )
    # remove problems from bnb tree
    '''
    for P in to_leave:
        parent = P['parent']
        if parent['left'] == P:
            parent['left'] = None
        else:
            parent['right'] = None
        P['parent'] = None
    '''
    return to_remain

def branch_and_bound(c, A, b, integrality = None, epsilon = 1e-3, problem_strategy = 'best', branching_strategy = 'lci', max_iters = 1e2):
    """
    Solve the mixed integer linear programming problem using branch and bound.
    Assuming that the objective function is to max c @ x subject 
    to A @ x = b and x >= 0.
    
    Parameters:
    - c: Coefficients for the objective function.
    - A: Coefficients for the inequality constraints.
    - b: Right-hand side of the inequality constraints.
    - integrality: List of indices of variables that must take integer values.
    - epsilon: Tolerance for optimality.
    - problem_strategy: Strategy for selecting the current problem ('dfs', 'bfs', 'best').
    - branching_strategy: Strategy for selecting the variable to branch ('lci', 'fni', 'random').
    
    Returns:
    - z_star: Optimal value of the objective function.
    - x_star: Optimal solution vector.
    - bnb_tree: Dictionary representing the branch and bound tree.
    - iters: number of iterations
    """

    if integrality is None:
        m, n = A.shape
        # we expect at least n - m integer variables
        # and let excess and slack variables be free
        integrality = [1] * (n - m) + [0] * m

    z_star = -np.inf
    biggest_z_sup = np.inf
    x_star = None
    bnb_tree = P = {
        'name': 0,
        'branch_variable': None,
        'integrality': integrality,
        'lower_bound': -np.inf,
        'upper_bound': np.inf,
        'A': A,
        'b': b,
        'c': -c, # as linprog minimizes by default
        'z': z_star,
        'z_sup': biggest_z_sup,
        'x': None,
        'status': None,
        'parent': None,
        'left': None,
        'right': None,
    }
    L = [ P ]
    active_problems = {
    }
    iters = 0

    while problems_to_solve(
        len(L) > 0,
        gap(biggest_z_sup, z_star) > epsilon,
        iters < max_iters
    ):
        active_problems[iters] = [p['name'] for p in L]
        # Step 1. Node selection:
        # Select the current problem based on the chosen strategy
        P_current = select_current_problem(L, problem_strategy)

        # 1.1 Solve the LRP
        current_z, current_x, current_status = solve_lp(
            P_current['c'],
            P_current['A'],
            P_current['b']
        )
        P_current['status'] = current_status

        # Step 2. Elimination test 1
        # If the current problem is infeasible, we skip it
        if current_status != 0:
            continue

        current_z *= -1  # since we minimized the negative of the objective function
        P_current['z_sup'] = current_z
        P_current['z'] = current_z
        P_current['x'] = current_x

        # Step 3. Elimination test 2
        # If the current optimal solution is worse than the best found
        # so far, we skip it
        if current_z <= z_star:
            continue

        # Step 4. Elimination test 3
        # update the best solution found so far if it matches integrality
        # criteria
        if check_integrality(current_x, P_current['integrality']):
            # Update incumbent z_star and x_star if current_z
            # is better than incumbent z_star
            if current_z > z_star:
                z_star = current_z
                x_star = current_x
            # Bounding
            # Prune all active nodes in L with z_sup worse than incumbent
            # z_star
            L = _bound(L, z_star)
            # go back to first step
            continue
        # Step 5. Branching
        # Select the variable to branch on according to the chosen strategy
        branch_index = select_to_branch(current_x, P_current['integrality'], branching_strategy)
        # Get the value of the variable to branch on
        branch_value = current_x[branch_index]

        # update at the current P which variable will branch with its lower and upper bounds
        P_current['branch_variable'] = branch_index
        P_current['lower_bound'] = np.floor(branch_value)
        P_current['upper_bound'] = np.ceil(branch_value)


        # Create two subproblems: one for the left branch (<= floor) and one for the right branch (>= ceil)
        left_subproblem = _create_subproblem(
            P_current, branch_index, branch_value, 'left'
        )
        right_subproblem = _create_subproblem(
            P_current, branch_index, branch_value, 'right'
        )
        # Add them to P_current
        P_current['left'] = left_subproblem
        P_current['right'] = right_subproblem

        # Add them to L
        L += [
            left_subproblem,
            right_subproblem
        ]

        # 7th and last step: update problems_to_solve condition
        biggest_z_sup = np.max([p['z_sup'] for p in L])

        iters += 1

    return z_star, x_star, bnb_tree, active_problems, iters
