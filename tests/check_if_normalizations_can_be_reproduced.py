import os
import pandas as pd
from pathlib import Path
import joblib
from tests.check_harness_outputs_vs_inputs import compare_cols_of_two_equally_shaped_dataframes


def main():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', -1)

    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')

    training_data_filename = 'perovskite/perovskitedata/0018.perovskitedata.csv'
    # # Reading in data from versioned-datasets repo.
    df = pd.read_csv(os.path.join(VERSIONED_DATA, training_data_filename),
                     comment='#',
                     low_memory=False)

    state_set = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0018.stateset.csv'),
                            comment='#',
                            low_memory=False)

    normalized_train_df = pd.read_csv("normalized_train_df.csv")
    normalized_test_df = pd.read_csv("normalized_test_df.csv")
    feature_cols = [c for c in normalized_train_df if ("_rxn_" in c) or ("_feat_" in c) or ("_calc_" in c)]
    non_numerical_cols = (normalized_train_df.select_dtypes('object').columns.tolist())
    feature_cols = [c for c in feature_cols if c not in non_numerical_cols]
    feature_cols.remove('unchanged__rxn_M_inorganic')
    feature_cols.remove('unchanged__rxn_M_organic')
    feature_cols.remove('_rxn_temperatureC_actual_bulk')

    print(df.shape, state_set.shape)
    print(normalized_train_df.shape, normalized_test_df.shape)
    print()

    index_cols = ['name', 'dataset']

    scaler = joblib.load("normalization_scaler_object.pkl")

    # check training_data:
    input_train = df.loc[df['name'].isin(normalized_train_df['name'])]

    # using loaded scaler to transform input dataframes and see if they match up with the internal test harness transformation
    input_train[feature_cols] = scaler.transform(input_train[feature_cols])

    normalized_train_df = normalized_train_df[index_cols + feature_cols]
    input_train = input_train[index_cols + feature_cols]
    print(input_train.shape, normalized_train_df.shape)
    print()
    compare_cols_of_two_equally_shaped_dataframes(input_train, normalized_train_df, index_cols, feature_cols)
    print()
    print()


if __name__ == '__main__':
    main()
