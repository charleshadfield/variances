import numpy as np

from sparse import pauli_kron_state_product
from variance import f_string, pauli_multiply_string

from itertools import islice
from multiprocessing import Pool
from functools import reduce


def variance_two_dics(argument):  # dic1, dic2, state, β):
    """
    subroutine for parallel computing
    does not compute trace-free component
    """
    dic1 = argument['dic1']
    dic2 = argument['dic2']
    state = argument['state']
    β = argument['beta']
    n = len(next(iter(dic1.keys())))  # num_qubits
    var = 0.0

    for Q, alphaQ in dic1.items():
        if Q == 'I'*n:
            continue
        for R, alphaR in dic2.items():
            if R == 'I'*n:
                continue
            f = f_string(Q, R, β)
            if f == 0.0:
                continue
            # else, need to calculate < state | PQ | state >
            QR = pauli_multiply_string(Q, R)
            tr_rho_QR = np.dot(np.conjugate(state), pauli_kron_state_product(QR, state)).real

            var += f * alphaQ * alphaR * tr_rho_QR

    return var


def _chunks(dic, size):
    it = iter(dic)
    for i in range(0, len(dic), size):
        yield {k: dic[k] for k in islice(it, size)}


def _size(dic, num_cores):
    return int(len(dic)/num_cores)+1


def variance_local_multithread(pauli_rep, energy, state, β, num_cores=15):
    size = _size(pauli_rep.dic, num_cores)
    chunks = _chunks(pauli_rep.dic, size)

    arguments = []
    for chunk in chunks:
        argument = {'dic1': chunk, 'dic2': pauli_rep.dic, 'state': state, 'beta': β}
        arguments.append(argument)

    p = Pool(processes=num_cores)
    variances = p.map(variance_two_dics, arguments)
    variance = reduce(lambda A, B: A+B, variances)

    energy_tf = pauli_rep.energy_tf(energy)
    variance -= energy_tf**2
    return variance
