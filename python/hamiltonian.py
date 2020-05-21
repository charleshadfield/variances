import itertools
import numpy as np

from pauli_rep import PauliRep

molecules = ['BeH2', 'LiH']
distances = [round(k, 1) for k in np.arange(start=0.5, stop=5.1, step=0.1)]


class HamiltonianToy():

    molecule = 'toy'
    distance = 'n/a'

    def __init__(self, pauli_rep):

        self.pauli_rep = pauli_rep
        self.num_qubits = self.pauli_rep.num_qubits


class Hamiltonian():

    def __init__(self, molecule, distance):
        self.molecule = molecule
        self.distance = distance
        self.path = self.path()
        self.pauli_rep = PauliRep(self.pauli_rep_dic())
        self.num_qubits = self.pauli_rep.num_qubits

    def path(self):
        # clean up distance to obtain path
        map_number = str(round((self.distance-0.5)*10))
        if self.distance % 1 == 0.0:
            distance_cleaned_up = round(self.distance)
        else:
            distance_cleaned_up = self.distance

        path = 'Hamiltonians/{0}/Potential/PESMap{1}atdistance{2}.txt'.format(self.molecule,
                                                                              map_number,
                                                                              distance_cleaned_up)
        return path

    def pauli_rep_dic(self):
        dic = {}
        with open(self.path, 'r') as f:
            for line1, line2 in itertools.zip_longest(*[f]*2):
                pauli_string = line1[:-1]
                coefficient = float(line2[:-1])
                dic[pauli_string] = coefficient
        return dic


class HamiltonianH2():

    def __init__(self, encoding):
        self.encoding = encoding
        self.path = self.path()
        self.pauli_rep = PauliRep(self.pauli_rep_dic())
        self.num_qubits = self.pauli_rep.num_qubits

    def path(self):
        path = 'Hamiltonians/hydrogen/{}.txt'.format(self.encoding)
        return path

    def pauli_rep_dic(self):
        dic = {}
        path = 'Hamiltonians/hydrogen/{}.txt'.format(self.encoding)
        with open(path, 'r') as f:
            dict1 = eval(f.read())
            list2 = dict1['paulis']
            for item in list2:
                pauli_string = item['label']
                coeff = item['coeff']['real']
                dic[pauli_string] = coeff
        return dic


class HamiltonianWA():

    def __init__(self, molecule, encoding):
        self.molecule = molecule
        self.encoding = encoding
        self.path = self.path()
        self.pauli_rep = PauliRep(self.pauli_rep_dic())
        self.num_qubits = self.pauli_rep.num_qubits

    def path(self):
        path = 'Hamiltonians/{0}/{1}.txt'.format(self.molecule, self.encoding)
        return path

    def pauli_rep_dic(self):
        dic = {}
        with open(self.path, 'r') as f:
            for line1, line2 in itertools.zip_longest(*[f]*2):
                pauli_string = line1[:-1]
                coefficient = float(line2[1:-5])
                dic[pauli_string] = coefficient
        return dic
