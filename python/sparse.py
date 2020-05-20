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
