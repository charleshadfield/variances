# optimisation problem for finding beta distributions
# two algorithms exist:
# - diagonal (only keep diagonal terms in cost function. Problem is convex)
# - mixed (all influential pairs, not convex, requires HF string)

from scipy.optimize import minimize, LinearConstraint

from var_opt import objective_diagonal, objective_mixed, build_influential_pairs


def find_optimal_beta_scipy(dic_tf, num_qubits, objective, β_initial=None, bitstring_HF=None):
    assert objective in ['diagonal', 'mixed']
    n = num_qubits
    if β_initial is None:
        x0 = [1/3]*(3*n)
    else:
        x0 = beta_to_x(β_initial)

    if objective == 'diagonal':
        def f(x):
            return objective_diagonal(dic_tf, x_to_beta(x))
    else:
        # objective == 'mixed'
        assert len(bitstring_HF) == n
        influential_pairs = build_influential_pairs(dic_tf, n)

        def f(x):
            return objective_mixed(dic_tf, influential_pairs, bitstring_HF, x_to_beta(x))

    result = minimize(f, x0, method='trust-constr', constraints=[constraints(n)])
    β = x_to_beta(result['x'])
    return β


def x_to_beta(x):
    β = {}
    for i, (a, b, c) in enumerate(zip(x[0::3], x[1::3], x[2::3])):
        β[i] = [a, b, c]
    return β


def beta_to_x(β):
    x = []
    for qubit in range(len(β)):
        for probability in β[qubit]:
            x.append(probability)


# Constraints. Identical, irrespective of diagonal_or_mixed


def _lin_con_single(k, n):
    linear_constraint = [0]*(3*n)
    linear_constraint[k] = 1
    return linear_constraint


def _lin_con_triple(i, n):
    linear_constraint = [0]*(3*n)
    for var in [3*i, 3*i+1, 3*i+2]:
        linear_constraint[var] = 1
    return linear_constraint


def linear_constraint_matrix(n):
    # constraints to ensure \beta_{i,P} \ge 1 for all i,P
    mat1 = [_lin_con_single(k, n) for k in range(3*n)]
    # constraints to ensure \sum_P \beta_{i,P} = 1 for all i
    mat2 = [_lin_con_triple(i, n) for i in range(n)]
    return mat1+mat2


def lower_bounds(n):
    bounds_single = [0]*(3*n)
    bounds_triple = [1]*n
    return bounds_single+bounds_triple


def upper_bounds(n):
    return [1]*(4*n)


def constraints(n):
    A = linear_constraint_matrix(n)
    lb = lower_bounds(n)
    ub = upper_bounds(n)
    return LinearConstraint(A, lb, ub)
