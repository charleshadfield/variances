import itertools


def read_encoding_return_dict(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        first_character = first_line[0]
        if first_character == '{':
            format = 'dictionary'
        else:
            format = 'list'

    if format == 'dictionary':
        dic = {}
        with open(file, 'r') as f:
            dict1 = eval(f.read())
            list2 = dict1['paulis']
            for item in list2:
                pauli_string = item['label']
                coeff = item['coeff']['real']
                dic[pauli_string] = coeff
        return dic

    if format == 'list':
        dic = {}
        with open(file, 'r') as f:
            for line1, line2 in itertools.zip_longest(*[f]*2):
                pauli_string = line1[:-1]
                coefficient = float(line2[1:-5])
                dic[pauli_string] = coefficient
        return dic

    pass


def read_ldf_return_dict(file, num_qubits):
    """
    Build dictionary of PauliRep styled dictionaries over groupings obtained from ldf algorithm.
    LDF algo has already been applied, and file is in same folder as encoding.
    WARNING: idenity string is excluded from this dictionary.
    """
    groups = {}
    group_number = -1
    with open(file, 'r') as f:
        for line_number, line in enumerate(f):
            if line[0:4] == '0.0,':
                group_number += 1
                groups[group_number] = {}
            else:
                # line is a string 'coefficient,pauli'
                # enter this coefficient into PauliRep style dictionary
                index_comma = line.index(',')
                coefficient = float(line[:index_comma])
                pauli = line[index_comma+1:-1]
                # do not include identity string
                if pauli == 'I'*num_qubits:
                    pass
                else:
                    groups[group_number][pauli] = coefficient
    return groups
