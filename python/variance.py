import numpy as np

# from sparse import to_matrix
from sparse import pauli_kron_state_product


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

            # old way:
            # QRmat = to_matrix(QR)
            # tr_rho_QR = np.dot(np.conjugate(state), QRmat * state).real

            # new way:
            tr_rho_QR = np.dot(np.conjugate(state), pauli_kron_state_product(QR, state)).real

            var += f * alphaQ * alphaR * tr_rho_QR

    energy_tf = pauli_rep.energy_tf(energy)
    var -= energy_tf**2
    return var

# Largest Degree first


def variance_ldf_single_group(group, state):
    var = 0.0
    for Q, alphaQ in group.items():
        for R, alphaR in group.items():
            QR = pauli_multiply_string(Q, R)
            tr_rho_QR = np.dot(np.conjugate(state), pauli_kron_state_product(QR, state)).real

            var += alphaQ * alphaR * tr_rho_QR
    return var


def variance_ldf(ldf, state, kappa, energy_tf):
    assert len(ldf.keys()) == len(kappa)
    var = 0.0
    for group_number, group in ldf.items():
        var += (1/kappa[group_number]) * variance_ldf_single_group(group, state)

    var -= energy_tf**2

    return var


def kappa_uniform(ldf):
    return [1/len(ldf)]*len(ldf)


def kappa_1norm(ldf):
    group_1norms = [np.linalg.norm(list(ldf[group].values()), ord=1)
                    for group in range(len(ldf))]
    total_norm = np.linalg.norm(group_1norms, ord=1)
    return group_1norms/total_norm
