import numpy as np

from qiskit import QuantumCircuit, execute
from qiskit import Aer
simulator = Aer.get_backend('qasm_simulator')


def estimators(trials, samples, pr, β, state):
    return [estimator(samples, pr, β, state) for _ in range(samples)]


def estimator(samples, pr, β, state):
    estimators = [estimator_single(pr, β, state) for _ in range(samples)]
    return 1/samples * sum(estimators)


def estimator_single(pr, β, state):
    num_qubits = pr.num_qubits
    circ = QuantumCircuit(num_qubits, num_qubits)
    circ.initialize(state, range(num_qubits))

    rps = random_pauli_string(num_qubits, β)

    circ += measurement_circuit(rps)

    # run experiment
    result = execute(circ, simulator, shots=1).result()
    counts = result.get_counts(circ)
    # counts is a dictionary with only one entry (since shots=1)
    bit_string = counts.popitem()[0]  # qiskit ordering

    energy_running = 0.0

    dic = pr.dic
    for pauli_string in dic:
        pauli_est = pauli_estimation_given_instance(pauli_string, rps, bit_string, β)
        pauli_energy = pauli_est * dic[pauli_string]

        energy_running += pauli_energy

    energy = energy_running

    return energy


def random_pauli_string(num_qubits, β):
    '''
    produce random pauli string for measurement.
    '''
    pauli_string = ''
    for qubit in range(num_qubits):
        pauli = np.random.choice(['X', 'Y', 'Z'], 1, p=β[qubit])[0]
        pauli_string = pauli + pauli_string
    return pauli_string


def measurement_circuit(pauli_string):
    num_qubits = len(pauli_string)
    circ = QuantumCircuit(num_qubits, num_qubits)
    # qiskit ordering
    for qubit, pauli in enumerate(pauli_string[::-1]):
        circ = measure_pauli(circ, pauli, qubit)
    return circ


def measure_pauli(circ, pauli, qubit):
    '''
    modify circuit by appending measurement.
    return modified circuit
    '''
    if pauli == 'X':
        circ.h(qubit)
    elif pauli == 'Y':
        circ.sdg(qubit)
        circ.h(qubit)
    elif pauli == 'Z':
        pass
    circ.measure(qubit, qubit)
    return circ


def pauli_estimation_given_instance(pauli_string, rps, bit_string, β):
    'given rps P and Pauli operator Q and bitstring of measurements b calculate f(P,Q,β) μ(P, supp(Q))'
    num_qubits = len(bit_string)
    assert num_qubits == len(rps)
    assert num_qubits == len(pauli_string)
    XYZ_to_012 = {'X': 0, 'Y': 1, 'Z': 2}
    m = 1

    # physics ordering
    for k, bit in enumerate(bit_string):
        q = pauli_string[k]
        p = rps[k]
        if q != 'I':
            if q != p:
                return 0
            else:  # q == p
                weight_inv = β[num_qubits-1-k][XYZ_to_012[q]]  # qiskit ordering
                weight = weight_inv ** (-1)
            m = m * (-1)**int(bit) * weight

    return m
