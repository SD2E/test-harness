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
        sequence = 'X' * padding + 'J' + (sequence + 'O').ljust(max_residues + 1 + padding, 'X')
        code = np.zeros((23, len(sequence)))
        for i in range(len(sequence)):
            code[amino_dict[sequence[i]], i] = 1.0
        return code

    df['encoded_sequence'] = df[col_to_encode].apply(make_code)
    # shuffle training data, because validation data are selected from end before shuffling:
    df = df.sample(frac=1)

    return df
