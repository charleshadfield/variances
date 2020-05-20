from scipy.optimize import minimize, LinearConstraint


def beta_inverse(Q, alphaQ, β):
    assert len(Q) == len(β.keys())
    prod = 1.0
    for i in range(len(Q)):
        if Q[i] != 'I':
            dic = {'X': 0, 'Y': 1, 'Z': 2}
            b = β[(len(Q)-1)-i][dic[Q[i]]]  # qiskit ordering
            if b == 0.0:
                # this cannot be allowed, as convergence in expectation won't work
                return float('inf')
            else:
                prod *= b
    return prod**(-1)


def objective(pauli_rep, β):
    assert len(β) == pauli_rep.num_qubits
    tally = 0.0
    for Q, alphaQ in pauli_rep.dic.items():
        if Q == 'I'*pauli_rep.num_qubits:
            continue
        tally += alphaQ**2 * beta_inverse(Q, alphaQ, β)
    return tally


def x_to_beta(x):
    β = {}
    for i, (a, b, c) in enumerate(zip(x[0::3], x[1::3], x[2::3])):
        β[i] = [a, b, c]
    return β


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


def optimal_beta(pauli_rep):
    n = pauli_rep.num_qubits
    x0 = [1/3]*(3*n)

    def f(x):
        return objective(pauli_rep, x_to_beta(x))

    result = minimize(f, x0, method='trust-constr', constraints=[constraints(n)])
    β = x_to_beta(result['x'])
    return β
