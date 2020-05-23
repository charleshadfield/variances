from copy import deepcopy
from kron_vec_product import kron_vec_prod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

row = np.array([0, 1])
col = np.array([0, 1])
data = np.array([1+0j, 1+0j])
id = sparse.csr_matrix((data, (row, col)), shape=(2, 2))
x = sparse.csr_matrix(([1+0j, 1+0j], ([0, 1], [1, 0])), shape=(2, 2))
y = sparse.csr_matrix(([0-1j, 0+1j], ([0, 1], [1, 0])), shape=(2, 2))
z = sparse.csr_matrix(([1+0j, -1+0j], ([0, 1], [0, 1])), shape=(2, 2))

_to_matrix = {'I': id, 'X': x, 'Y': y, 'Z': z}


def to_matrix(pauli_string):
    p0 = pauli_string[0]
    mat = _to_matrix[p0]
    for p in pauli_string[1:]:
        mat = sparse.kron(mat, _to_matrix[p], format="csr")
    return mat


def matrix(pauli_rep):
    n = pauli_rep.num_qubits
    mat = sparse.csr_matrix((2**n, 2**n), dtype=np.complex128)
    for pauli_string, coefficient in pauli_rep.dic.items():
        mat += coefficient * to_matrix(pauli_string)
    return mat


def ground(pauli_rep):
    mat = matrix(pauli_rep)
    evals, evecs = eigsh(mat, which='SA')
    # SA looks for algebraically small evalues
    index = np.argmin(evals)
    return evals[index], evecs[:, index]


def energy(pauli_rep, state, tol=1e-6):
    '''
    Directly calculate energy by tracing state over Pauli kronecker product.
    Return real float.
    '''
    state_dual = np.conjugate(state)
    mat = matrix(pauli_rep)
    trace = np.dot(state_dual, mat * state)
    assert abs(trace.imag) < tol
    return np.real(trace)

# Added by RRHP on May 22, 2020


def get_ij(pauli_string, i, j):
    # we compute the element of the kronecker product on-the-fly
    n = len(pauli_string)
    # N = 2**n
    mij = 1.0

    for k, p in enumerate(pauli_string):
        rk = 0 if i & (1 << (n-k-1)) == 0 else 1  # _i // (N/2**(k+1))
        ck = 0 if j & (1 << (n-k-1)) == 0 else 1  # _j // (N/2**(k+1))
        _v = _to_matrix[p][rk, ck]
        if _v == 0.0:
            return 0.0
        else:
            mij *= _v
    return mij


def energy_kron(pauli_rep, state, tol=1e-6):
    """
    compute energy by taking advantage of kronecker product representation
    """
    state_dual = np.conjugate(state)
    mat_state = np.zeros(state.shape, dtype=complex)
    n = pauli_rep.num_qubits
    N = 2**n
    for pauli_string, coefficient in pauli_rep.dic.items():
        for i in range(N):
            for j in range(N):
                mat_state[i] += get_ij(pauli_string, i, j) * coefficient * state[j]
    trace = np.dot(state_dual, mat_state)
    assert abs(trace.imag) < tol
    return np.real(trace)


def mult_paulis_state(pauli_string, state):
    n = len(pauli_string)
    answer = deepcopy(state)
    for i, p in enumerate(pauli_string[::-1]):
        if p == "I":
            continue
        if 0 < i < n-1:
            if i <= n//2:
                A = sparse.kron(_to_matrix[p], sparse.eye(2**i, format="csr"))
                answer = A * answer.reshape((A.shape[0], -1), order="F")
                answer = answer.reshape((-1, 1), order="F")

            else:
                B = sparse.kron(sparse.eye(2**(n-i-1), format="csr"), _to_matrix[p], format="csr")
                answer = answer.reshape((-1, B.shape[1]), order="F") * B.T
                answer = answer.reshape((-1, 1), order="F")

            # mat = sparse.kron(sparse.eye(2**(n-i-1), format="csr"), _to_matrix[p], format="csr")
            # mat = sparse.kron(mat, sparse.eye(2**i, format="csr"))
            # answer = mat.dot(answer)
        elif i == 0:
            # answer = sparse.kron(sparse.eye(2**(n-1), format="csr"), _to_matrix[p], format="csr").dot(answer)
            answer = _to_matrix[p] * answer.reshape((2, -1), order="F")
            answer = answer.reshape((-1, 1), order="F")
        else:
            # answer = sparse.kron(_to_matrix[p], sparse.eye(2**(n-1), format="csr"), format="csr").dot(answer)
            answer = answer.reshape((-1, 2), order="F") * _to_matrix[p].T
            answer = answer.reshape((-1, 1), order="F")

    return answer


def generalized_kron_state_product(pauli_string, state):
    """
        Using generalized vec-trick for kron-vec multiplication
    """
    paulis = [_to_matrix[p] for p in pauli_string]
    return kron_vec_prod(paulis, state)


def pauli_kron_state_product(pauli_string, vec1):
    """
        THIS MUST BE USED FOR KRONECKER PRODUCT OF PAULI MATRICES
        Using generalized vec-trick optimized for Pauli matrices for kron-vec multiplication
    """
    n = len(pauli_string)
    # N = 2**n
    # halfN = N // 2
    vec = deepcopy(vec1)
    vec = vec.reshape((2, -1), order="F")  # reshape
    for i, p in enumerate(pauli_string[::-1]):
        # apply Pi
        if p == "Z":
            # multiply the second row by -1
            vec[1, :] *= -1
        elif p == "X" or p == "Y":
            # swap the rows
            vec[[0, 1]] = vec[[1, 0]]
            if p == "Y":
                vec *= 1.0j
                vec[0, :] *= -1
        if i < n - 1:
            vec = vec.reshape((-1, 2), order="C").T

    return vec.ravel()
# end of addition
