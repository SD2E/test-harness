"""
The purpose of this module is to test out the "run_custom" method with the new "execute" argument.
It also tests out the move of the "_add_sparse_cols" method within the _BaseRun class.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression


def main():
    dir_path = Path(__file__).parent
    project_path = Path(dir_path).parent.parent
    data_path = os.path.join(project_path, "example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv")

    protein_data = pd.read_csv(data_path, comment='#', low_memory=False, nrows=1000)

    # creating a smaller df with less columns to see what's going on easier
    some_features = protein_data.columns.values.tolist()[13:19]
    protein_data = protein_data[["dataset", "name", "topology", "stabilityscore_cnn_calibrated"] + some_features]
    # create fake sparse cols
    fake_sparse_col_1 = "fake_sparse_col_1"
    fake_sparse_col_2 = "fake_sparse_col_2"
    protein_data["fake_sparse_col_1"] = ["A", "B"] * 500
    protein_data["fake_sparse_col_2"] = ["C", "A"] * 500
    fake_sparse_cols = [fake_sparse_col_1, fake_sparse_col_2]
    feature_cols_to_use = some_features + fake_sparse_cols

    regression_prediction_col = "stabilityscore_cnn_calibrated"
    train_df, test_df = train_test_split(protein_data, test_size=0.2, random_state=5, stratify=protein_data['topology'])
    index_cols = ["dataset", "name"]

    th = TestHarness(output_location=dir_path, compress_large_csvs=True)

    # I call run_custom with execute=False so I can see what will be run before it's run
    run_object = th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                               training_data=train_df, testing_data=test_df,
                               description="Demo Run on Rocklin dataset",
                               target_cols=regression_prediction_col,
                               feature_cols_to_use=feature_cols_to_use,
                               index_cols=index_cols,
                               normalize=True, feature_cols_to_normalize=some_features,
                               feature_extraction=False, predict_untested_data=False,
                               sparse_cols_to_use=fake_sparse_cols, execute=False)
    print(run_object.training_data.head(), "\n")
    print("training_data shape: {}".format(run_object.training_data.shape), "\n")
    print("sparse_cols_to_use: {}".format(run_object.sparse_cols_to_use), "\n")
    print("feature_cols_to_use: {}".format(run_object.feature_cols_to_use), "\n")
    print("feature_cols_to_normalize: {}".format(run_object.feature_cols_to_normalize), "\n")

    # Now that I'm satisfied with what's going to be run, I will call run_custom like we usually do
    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                  training_data=train_df, testing_data=test_df,
                  description="Demo Run on Rocklin dataset",
                  target_cols=regression_prediction_col,
                  feature_cols_to_use=feature_cols_to_use,
                  index_cols=index_cols,
                  normalize=True, feature_cols_to_normalize=some_features,
                  feature_extraction=False, predict_untested_data=False,
                  sparse_cols_to_use=fake_sparse_cols)

    last_run = th.list_of_this_instance_run_ids[-1]
    training_data_csv_that_was_output = pd.read_csv(
        os.path.join(dir_path, "test_harness_results/runs/run_{}/training_data.csv.gz".format(last_run)))
    print(training_data_csv_that_was_output.head(), "\n")
    print("training_data_csv_that_was_output shape: {}".format(training_data_csv_that_was_output.shape))


if __name__ == '__main__':
    main()
