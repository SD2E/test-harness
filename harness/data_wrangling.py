import numpy as np
import pandas as pd


# This module is for functions that alter the data before passing it into the Test Harness.
# For example, a function for creating a one-hot-encoding of the sequence column


def calculate_max_residues(list_of_dfs, sequence_col='sequence'):
    def is_list_of_dfs(obj):
        if obj and isinstance(obj, list):
            return all(isinstance(elem, pd.DataFrame) for elem in obj)
        else:
            return False

    assert is_list_of_dfs(list_of_dfs), "list_of_dfs must be a list of Pandas Dataframes"

    max_list = []
    for df in list_of_dfs:
        max_list.append(df[sequence_col].map(len).max())
    max_residues = max(max_list)
    return max_residues


def encode_sequences(df, max_residues, col_to_encode='sequence', padding=14):
    amino_dict = dict(
        zip(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'J', 'O'],
            range(23)))  # 'X' means nothing, 'J' means beginning, 'O' means end

    def make_code(sequence):
        extra_padding = hash(sequence) % (max_residues - len(sequence)) #MAX_RESIDUES doesn't include padding or J/O, so this is correct
        sequence = ('X' * (padding + extra_padding) + 'J' + sequence + 'O').ljust(max_residues + 2 + 2*padding, 'X')
        code = np.zeros((23, len(sequence)))
        for i in range(len(sequence)):
            code[amino_dict[sequence[i]], i] = 1.0
        return code

    df['encoded_sequence'] = df[col_to_encode].apply(make_code)
    # shuffle training data, because validation data are selected from end before shuffling:
    df = df.sample(frac=1)

    return df


def encode_dssps(df, max_residues, col_to_encode='dssp', seq_col='sequence', padding=14):
    structure_dict = dict(zip(['L', 'E', 'H', 'X', 'J', 'O'], range(6)))
    def make_struct(dssp, sequence):
        extra_padding = hash(sequence) % (max_residues - len(sequence))
        dssp = ('X' * (padding + extra_padding) + 'J' + dssp + 'O').ljust(max_residues + 2 + 2*padding, 'X')
        code = np.zeros((len(dssp), 6))
        for i in range(len(dssp)):
            code[i, structure_dict[dssp[i]]] = 1.0
        return [code]
    
    df['encoded_dssp'] = df[col_to_encode].apply(lambda x: make_struct(x[col_to_encode], x[seq_col]))
    # shuffle training data, because validation data are selected from end before shuffling:
    df = df.sample(frac=1)
    
    return df
    

def encode_cnn_v2_targets(df,
                          max_residues,
                          stability_col='stabilityscore_cnn_calibrated',
                          stability_c_col='stabilityscore_cnn_calibrated_c',
                          stability_t_col='stabilityscore_cnn_calibrated_t',
                          seq_col='sequence',
                          dssp_col='dssp',
                          padding=14
                         ):
    df = encode_dssps(df, max_residues, col_to_encode=dssp_col, seq_col=seq_col, padding=padding)
    def make_target(row):
        return (
                (
                 row[stability_c_col],
                 row[stability_t_col],
                 row[stability_col]
                ),
                row['encoded_dssp']
               )
    df = df.apply(make_target, axis=1)
    
    return df
    
    