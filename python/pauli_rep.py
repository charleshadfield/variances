import numpy as np

from sparse import ground
from var import variance_local
from var_mt import variance_local_multithread
from var_opt_scipy import find_optimal_beta_scipy
from var_opt_lagrange import find_optimal_beta_lagrange


class PauliRep():

    def __init__(self, dic):
        self.dic = dic
        self.num_qubits = self.num_qubits()
        self.iden_coef = self.iden_coef()
        self.dic_tf = self._build_dic_tf()
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

    def _build_dic_tf(self):
        dic_tf = self.dic.copy()
        iden = 'I' * self.num_qubits
        if iden in dic_tf.keys():
            dic_tf.pop(iden)
        return dic_tf

    def one_norm_probs_tf(self):
        paulis = []
        probs = []
        for pauli in self.dic_tf:
            # don't include identity
            # if pauli == 'I' * self.num_qubits:
            #     continue
            paulis.append(pauli)
            probs.append(abs(self.dic[pauli]) / self.one_norm_tf)
        return [paulis, probs]

    def energy_tf(self, energy):
        return energy - self.iden_coef

    def ground(self, multithread=False, num_cores=15):
        return ground(self, multithread=multithread, num_cores=num_cores)

    def variance_local(self, energy, state, β, multithread=False, num_cores=15):
        if multithread is False:
            return variance_local(self, energy, state, β)
        else:
            return variance_local_multithread(self, energy, state, β, num_cores=num_cores)

    def variance_ell_1(self, energy):
        return (self.one_norm_tf)**2 - (self.energy_tf(energy))**2

    def local_dists_uniform(self):
        dic = {}
        for i in range(self.num_qubits):
            dic[i] = [1/3, 1/3, 1/3]
        return dic

    def local_dists_optimal(self, objective, method, β_initial=None, bitstring_HF=None):
        """Find optimal probabilities beta_{i,P} and return as dictionary
        attn: qiskit ordering"""
        assert objective in ['diagonal', 'mixed']
        assert method in ['scipy', 'lagrange']
        if method == 'scipy':
            return find_optimal_beta_scipy(self.dic_tf, self.num_qubits, objective,
                                           β_initial=β_initial, bitstring_HF=bitstring_HF)
        else:
            # method == 'lagrange'
            return find_optimal_beta_lagrange(self.dic_tf, self.num_qubits, objective,
                                              tol=1.0e-5, iter=10000,
                                              β_initial=β_initial, bitstring_HF=bitstring_HF)

    # Code below is old. And also inefficient! Use np.linalg.norm

    def local_dists_pnorm(self, norm):
        '''
        Return dictionary (over all qubits) of normalized p_norms (over all three paulis)
        attn: qiskit ordering
        '''
        assert norm in [1, 2, 'infinity']

        dic = {}
        for qubit in range(self.num_qubits):
            pnorm_sums = [0.0, 0.0, 0.0]  # [X, Y, Z]-sums
            for index, pauli in enumerate(['X', 'Y', 'Z']):
                pnorm_sums[index] = self._local_pnorm_sum(qubit, pauli, norm)
                # this is NOT yet normalized.
            one_norm = sum(pnorm_sums)
            dic[qubit] = [x / one_norm for x in pnorm_sums]
        return dic

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
