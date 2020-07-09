import os
import pandas as pd
from pathlib import Path


def main():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', None)

    VERSIONED_DATA = os.path.join(Path(__file__).resolve().parents[2], 'versioned-datasets/data')

    training_data_filename = 'perovskite/perovskitedata/0018.perovskitedata.csv'
    # # Reading in data from versioned-datasets repo.
    df = pd.read_csv(os.path.join(VERSIONED_DATA, training_data_filename),
                     comment='#',
                     low_memory=False)

    state_set = pd.read_csv(os.path.join(VERSIONED_DATA, 'perovskite/stateset/0018.stateset.csv'),
                            comment='#',
                            low_memory=False)

    RUN_ID = "5ZzVXNP2ypqJY"

    out_files_path = "test_harness_results/runs/run_{}".format(RUN_ID)
    output_train = pd.read_csv(os.path.join(out_files_path, "training_data.csv"))
    output_test = pd.read_csv(os.path.join(out_files_path, "testing_data.csv"))
    output_pred = pd.read_csv(os.path.join(out_files_path, "predicted_data.csv"))

    print(df.shape, state_set.shape)
    print(output_train.shape, output_test.shape, output_pred.shape)
    print()

    index_cols = ['name', 'dataset']

    # check training_data:
    input_train = df.loc[df['name'].isin(output_train['name'])]
    mutual_cols = list(set(output_train.columns.tolist()).intersection(set(input_train.columns.tolist())))
    output_train = output_train[mutual_cols]
    input_train = input_train[mutual_cols]
    print(input_train.shape, output_train.shape)
    compare_cols_of_two_equally_shaped_dataframes(input_train, output_train, index_cols)
    print()
    print()

    # check testing_data:
    input_test = df.loc[df['name'].isin(output_test['name'])]
    print(input_test.shape)
    mutual_cols = list(set(output_test.columns.tolist()).intersection(set(input_test.columns.tolist())))
    output_test = output_test[mutual_cols]
    input_test = input_test[mutual_cols]
    print(input_test.shape, output_test.shape)
    compare_cols_of_two_equally_shaped_dataframes(input_test, output_test, index_cols)
    print()
    print()

    # check predicted_data:
    input_pred = state_set.loc[state_set['name'].isin(output_pred['name'])]
    mutual_cols = list(set(output_pred.columns.tolist()).intersection(set(input_pred.columns.tolist())))
    output_pred = output_pred[mutual_cols]
    input_pred = input_pred[mutual_cols]
    print(input_pred.shape, output_pred.shape)
    compare_cols_of_two_equally_shaped_dataframes(input_pred, output_pred, index_cols)


def compare_cols_of_two_equally_shaped_dataframes(df1, df2, index_cols, cols_to_check=None):
    merged = pd.merge(df1, df2, on=index_cols)

    if cols_to_check is None:
        cols_to_check = df1.columns.tolist()
        for i in index_cols:
            cols_to_check.remove(i)

    for col in cols_to_check:
        x = "{}_x".format(col)
        y = "{}_y".format(col)
        merged[x] = merged[x].apply(lambda a: round(a, 10))
        merged[y] = merged[y].apply(lambda a: round(a, 10))
        merged[col] = merged[x] - merged[y]
        print(merged[col].value_counts(dropna=False))
        print()


if __name__ == '__main__':
    main()
