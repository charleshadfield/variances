from pauli_rep import PauliRep
from ham_read_file import read_encoding_return_dict, read_ldf_return_dict


class Hamiltonian():

    def __init__(self, folder, encoding):
        self.folder = folder
        self.encoding = encoding
        self.path = self.path()
        self.pauli_rep = PauliRep(self._pauli_rep_dic())
        self.num_qubits = self.pauli_rep.num_qubits

    def path(self):
        path = 'Hamiltonians/{0}/{1}.txt'.format(self.folder, self.encoding)
        return path

    def _pauli_rep_dic(self):
        return read_encoding_return_dict(self.path)

    def ldf(self):
        path = 'Hamiltonians/{0}/{1}_grouped.txt'.format(self.folder, self.encoding)
        return read_ldf_return_dict(path, self.num_qubits)


class HamiltonianToy():
    """used for testing"""

    def __init__(self, pauli_rep):

        self.pauli_rep = pauli_rep
        self.num_qubits = self.pauli_rep.num_qubits


class HamiltonianSmall():
    """LiH and BeH2 in a single encoding. Used for testing"""

    # molecules = ['BeH2', 'LiH']
    # distances = [round(k, 1) for k in np.arange(start=0.5, stop=5.1, step=0.1)]

    def __init__(self, molecule, distance):
        self.molecule = molecule
        self.distance = distance
        self.path = self.path()
        self.pauli_rep = PauliRep(self._pauli_rep_dic())
        self.num_qubits = self.pauli_rep.num_qubits

    def path(self):
        # clean up distance to obtain path
        map_number = str(round((self.distance-0.5)*10))
        if self.distance % 1 == 0.0:
            distance_cleaned_up = round(self.distance)
        else:
            distance_cleaned_up = self.distance

        path = 'Hamiltonians/small/{0}/Potential/PESMap{1}atdistance{2}.txt'.format(self.molecule,
                                                                                    map_number,
                                                                                    distance_cleaned_up)
        return path

    def _pauli_rep_dic(self):
        return read_encoding_return_dict(self.path)
