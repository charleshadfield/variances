# optimisation problem for finding beta distributions
# two algorithms exist:
# - diagonal (only keep diagonal terms in cost function. Problem is convex)
# - mixed (all influential pairs, not convex, requires HF string)

# this file: the objective functions


# Diagonal objective function


def calculate_product_term_diagonal(Q, β):
    # \prod_{i\in\supp(\Qarrow)} \beta_{i,Q_i}^{-1}
    # assert len(Q) == len(β.keys())
    pauli_to_index = {'X': 0, 'Y': 1, 'Z': 2}
    prod = 1.0
    for i in range(len(Q)):
        if Q[i] != 'I':
            qubit = (len(Q)-1)-i  # qiskit ordering
            index = pauli_to_index[Q[i]]
            b = β[qubit][index]
            if b == 0.0:
                # this cannot be allowed, as convergence in expectation won't work
                return float('inf')
            else:
                prod *= b
    return prod**(-1)


def objective_diagonal(dic_tf, β):
    tally = 0.0
    for Q, alphaQ in dic_tf.items():
        tally += alphaQ**2 * calculate_product_term_diagonal(Q, β)
    return tally


# Mixed objective functions


def is_influential_pauli_single_qubit(q, r):
    if q == r:
        return True
    elif q == 'I' and r == 'Z':
        return True
    elif q == 'Z' and r == 'I':
        return True
    else:
        return False


def is_influential_pair(Q, R, n):
    # do not check if Q or R are trace-free. Assume they are!
    for i in range(n):
        if is_influential_pauli_single_qubit(Q[i], R[i]) is False:
            return False
    # if I arrive here, then all qubits i have pauli Q_i R_i acceptable
    return True


def build_influential_pairs(dic_tf, n):
    # do not check if dic_tf has identity term. Assume it is not present!
    pairs = []
    for Q in dic_tf:
        for R in dic_tf:
            if is_influential_pair(Q, R, n) is True:
                pairs.append((Q, R))
    return pairs


def calculate_product_term_mixed(Q, R, bitstring_HF, β):
    # assume Q,R are already admissable!
    # assume β, m are of correct structure!
    # \prod_{i | Q_i=R_i\neq I} \beta_{i,Q_i}^{-1} \prod_{i | Q_i\neq R_i} m_i
    pauli_to_index = {'X': 0, 'Y': 1, 'Z': 2}
    prod = 1.0
    for i in range(len(Q)):
        if Q[i] == R[i] and Q[i] != 'I':
            qubit = (len(Q)-1)-i  # qiskit ordering
            index = pauli_to_index[Q[i]]
            b = β[qubit][index]
            # if b == 0.0:
            # this cannot be allowed, as convergence in expectation won't work
            #    return float('inf')
            # else:
            prod *= b**(-1)
        if Q[i] != R[i]:
            # then Q[i], R[i] are of the form I,Z or Z,I
            bit = int(bitstring_HF[i])
            m = (-1)**bit
            prod *= m
    return prod


def objective_mixed(dic_tf, influential_pairs, bits_HF, β):
    tally = 0.0
    for Q, R in influential_pairs:
        alphaQ = dic_tf[Q]
        alphaR = dic_tf[R]
        prod = calculate_product_term_mixed(Q, R, bits_HF, β)
        tally += alphaQ * alphaR * prod
    return tally
