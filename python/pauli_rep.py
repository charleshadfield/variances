import numpy as np


class PauliRep():

    def __init__(self, dic):
        self.dic = dic
        self.num_qubits = self.num_qubits()
        self.iden_coef = self.iden_coef()
        self.one_norm_tf = sum(abs(self.dic[p]) for p in self.dic) - abs(self.iden_coef)

    def num_qubits(self):
        pauli = next(iter(self.dic.keys()))
        return len(pauli)

    def iden_coef(self):
        iden = 'I' * self.num_qubits
        coef = 0.0
        if iden in self.dic:
            coef = self.dic[iden]
        return coef

    def one_norm_probs_tf(self):
        paulis = []
        probs = []
        for pauli in self.dic:
            # don't include identity
            if pauli == 'I' * self.num_qubits:
                continue
            paulis.append(pauli)
            probs.append(abs(self.dic[pauli]) / self.one_norm_tf)
        return [paulis, probs]

    def _local_pnorm_sum(self, qubit, pauli, norm):
        '''
        For given qubit, given pauli, return p-norm weighted sum over coefficients
        corresponding to pstring for which pstring[qubit]=pauli.

        attn: qiskit ordering
        '''
        assert norm in [1, 2, 'infinity']
        assert pauli in ['X', 'Y', 'Z']
        running_pnorm_sum = 0.0

        for pstring in self.dic:
            # qiskit ordering
            if pstring[-(qubit+1)] == pauli:
                coefficient = self.dic[pstring]
                if norm in [1, 2]:
                    running_pnorm_sum += np.abs(coefficient)**norm
                else:
                    running_pnorm_sum = max(running_pnorm_sum, abs(coefficient))

        if norm == 2:
            running_pnorm_sum = np.sqrt(running_pnorm_sum)

        return running_pnorm_sum

    def local_dists_uniform(self):
        dic = {}
        for i in range(self.num_qubits):
            dic[i] = [1/3, 1/3, 1/3]
        return dic

    def local_dists_pnorm(self, norm):
        assert norm in [1, 2, 'infinity']

        '''
        Return dictionary (over all qubits) of normalized p_norms (over all three paulis)
        attn: qiskit ordering
        '''
        dic = {}
        for qubit in range(self.num_qubits):
            pnorm_sums = [0.0, 0.0, 0.0]  # [X, Y, Z]-sums
            for index, pauli in enumerate(['X', 'Y', 'Z']):
                pnorm_sums[index] = self._local_pnorm_sum(qubit, pauli, norm)
                # this is NOT yet normalized.
            one_norm = sum(pnorm_sums)
            dic[qubit] = [x / one_norm for x in pnorm_sums]
        return dic

    def energy_tf(self, energy):
        '''
        return energy of trace-free hamiltonian
        '''
        return energy - self.iden_coef
