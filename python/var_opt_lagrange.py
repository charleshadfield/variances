# optimisation problem for finding beta distributions
# two algorithms exist:
# - diagonal (only keep diagonal terms in cost function. Problem is convex)
# and todo:
# - mixed (all influential pairs, not convex, requires HF string)

# see note LagrangeMethodForBetas.md

import numpy as np
from var_opt import calculate_product_term_diagonal  # objective_diagonal,


def find_optimal_beta_lagrange(dic_tf, num_qubits, objective, tol=1.0e-5, iter=1000, β_initial=None, bitstring_HF=None):
    assert objective in ['diagonal']  # TODO 'mixed'

    # ITERATIVE UPDATES TO FIND BETAS
    iterations = 0
    β_old = β_initial
    while True and iterations < iter:
        β_new, error = update_betas(dic_tf, num_qubits, β_old)
        β_old = β_new
        iterations += 1
        if error < tol:
            break
    return β_new


# Diagonal helper functions


def lagrange_restriction_numerator(i, p, dic_tf, β):
    assert p in ['X', 'Y', 'Z']
    tally = 0.0
    for Q in dic_tf:
        if Q[(len(Q)-1)-i] == p:  # qiskit ordering
            tally += (dic_tf[Q]**2) * calculate_product_term_diagonal(Q, β)
    return tally


def lagrange_restriction_denominator(i, dic_tf, β):
    tally = 0.0
    for Q in dic_tf:
        if Q[(len(Q)-1)-i] != "I":  # qiskit ordering
            tally += (dic_tf[Q]**2) * calculate_product_term_diagonal(Q, β)
    return tally


def lagrange_restriction(i, p, dic_tf, β):
    numerator = lagrange_restriction_numerator(i, p, dic_tf, β)
    denominator = lagrange_restriction_denominator(i, dic_tf, β)
    return numerator / denominator


# THIS IS  WHERE WE USE ITERATIVE UPDATES TO FIND THE BEST BETAS

def distance(β_1, β_2):
    two_norm_squared = 0.0
    for qubit in β_1.keys():  # qubit is qubit number (in qiskit ordering)
        two_norm_squared_qubit = np.sum((np.array(β_1[qubit]) - np.array(β_2[qubit]))**2)
        two_norm_squared += two_norm_squared_qubit
    return np.sqrt(two_norm_squared)


def update_betas(dic_tf, num_qubits, β_old=None, weight=0.1):
    """
        iteratively update betas with new values by weight
    """
    if β_old is None:  # initialize with random uniform
        β_old = {}
        for qubit in range(num_qubits):
            β_old[qubit] = [1./3 for _ in range(3)]
    β_new = {}
    for qubit in range(num_qubits):
        β_new[qubit] = []
        for index, pauli in enumerate(("X", "Y", "Z")):
            lagrange_rest = lagrange_restriction(qubit, pauli, dic_tf, β_old)
            update = (1. - weight) * β_old[qubit][index] + weight * lagrange_rest
            β_new[qubit].append(update)
    return β_new, distance(β_new, β_old)
