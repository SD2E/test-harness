import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from harness.th_model_instances.hamed_models.keras_regression import keras_regression_best
import numpy as np


def main():
    # Set model type to test here (sklearn or keras)
    test_model_type = "sklearn"

    cwd = os.getcwd()
    project_path = os.path.join(cwd.split("/test-harness/")[0], "test-harness")
    data_path = os.path.join(project_path, "example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv")

    protein_data = pd.read_csv(data_path, comment='#', low_memory=False, nrows=1000)
    protein_data_1 = protein_data[0:500]
    protein_data_2 = protein_data[500:1000]

    regression_prediction_col = "stabilityscore_cnn_calibrated"
    feature_columns = protein_data.columns.values.tolist()[13:]

    harness_output_path = os.path.join(project_path, "tests/test_model_saving_and_predict_only")
    th = TestHarness(output_location=harness_output_path)
    train_df, test_df = train_test_split(protein_data_1, test_size=0.2, random_state=5, stratify=protein_data_1['topology'])
    index_cols = ["dataset", "name"]

    if test_model_type == "sklearn":
        th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                      training_data=train_df, testing_data=test_df,
                      description="Demo Run on Rocklin dataset",
                      target_cols=regression_prediction_col,
                      feature_cols_to_use=feature_columns,
                      index_cols=index_cols,
                      normalize=True, feature_cols_to_normalize=feature_columns,
                      feature_extraction=False, predict_untested_data=False)
    elif test_model_type == "keras":
        th.run_custom(function_that_returns_TH_model=keras_regression_best, dict_of_function_parameters={},
                      training_data=train_df, testing_data=test_df,
                      description="Demo Run on Rocklin dataset",
                      target_cols=regression_prediction_col,
                      feature_cols_to_use=feature_columns,
                      index_cols=index_cols,
                      normalize=True, feature_cols_to_normalize=feature_columns,
                      feature_extraction=False, predict_untested_data=False)
    else:
        raise NotImplementedError("{} is not implemented".format(test_model_type))

    last_run = th.list_of_this_instance_run_ids[-1]
    last_run = "run_" + last_run
    print(last_run)

    th.predict_only(run_id_of_saved_model=last_run,
                    data_to_predict=protein_data_2,
                    index_cols=index_cols,
                    target_col=regression_prediction_col,
                    feature_cols_to_use=feature_columns)

    # read in predicted_data.csv file:
    read_in_predicted_data = pd.read_csv(os.path.join(harness_output_path,
                                                      "test_harness_results/runs/{}/predicted_data.csv".format(last_run)))
    print(read_in_predicted_data.head())


if __name__ == '__main__':
    main()
