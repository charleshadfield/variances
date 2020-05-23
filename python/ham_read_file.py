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
