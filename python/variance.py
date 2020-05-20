import numpy as np

from sparse import to_matrix


def f_individual(q, r, βi):
    assert q in ['I', 'X', 'Y', 'Z']
    assert r in ['I', 'X', 'Y', 'Z']
    assert len(βi) == 3
    if q == 'I' or r == 'I':
        return 1.0
    elif q == r:
        dic = {'X': 0, 'Y': 1, 'Z': 2}
        if βi[dic[q]] == 0.0:
            return 0.0
        return (βi[dic[q]])**(-1)
    else:
        return 0.0


def f_string(Q, R, β):
    assert len(Q) == len(R)
    assert len(Q) == len(β.keys())
    prod = 1.0
    for i in range(len(Q)):
        # qiskit ordering
        prod *= f_individual(Q[i], R[i], β[(len(Q)-1)-i])
    return prod


def pauli_multiply_individual(q, r):
    assert q in ['I', 'X', 'Y', 'Z']
    assert r in ['I', 'X', 'Y', 'Z']
    if q == r:
        # return q*r
        return 'I'
    else:
        # at least one of q or r is the identity, the other one should be returned
        if q != 'I':
            return q
        else:
            return r
    # error if I arrive here


def pauli_multiply_string(Q, R):
    assert len(Q) == len(R)
    string = ''
    for i in range(len(Q)):
        string += pauli_multiply_individual(Q[i], R[i])
    return string


def variance_local(pauli_rep, energy, state, β):
    var = 0.0

    for Q, alphaQ in pauli_rep.dic.items():
        if Q == 'I' * pauli_rep.num_qubits:
            continue
        for R, alphaR in pauli_rep.dic.items():
            if R == 'I' * pauli_rep.num_qubits:
                continue
            f = f_string(Q, R, β)
            if f == 0.0:
                continue
            # else, need to calculate < state | PQ | state >
            QR = pauli_multiply_string(Q, R)
            QRmat = to_matrix(QR)
            tr_rho_QR = np.dot(np.conjugate(state), QRmat * state).real

            var += f * alphaQ * alphaR * tr_rho_QR

    energy_tf = pauli_rep.energy_tf(energy)
    var -= energy_tf**2
    return var


def variance(pauli_rep, energy, state, algo, p_norm=None):
    """
    Calculate variance (precisely) for ground state associated with Hamiltonian
    """

    assert algo in ["ell_1", "uniform", "biased"]
    if algo == "biased":
        assert p_norm in [1, 2, "infinity"]

    elif algo == "ell_1":
        energy_tf = pauli_rep.energy_tf(energy)
        return (pauli_rep.one_norm_tf)**2 - energy_tf**2

    elif algo == "uniform":
        β = pauli_rep.local_dists_uniform()
        return variance_local(pauli_rep, energy, state, β)

    else:
        # algo == "biased"
        β = pauli_rep.local_dists_pnorm(p_norm)
        return variance_local(pauli_rep, energy, state, β)
