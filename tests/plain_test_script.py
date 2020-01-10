"""
An example script showing how to invoke the test harness functions to run a random forest regression on protein stability data
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from harness.th_model_instances.hamed_models.rocklin_models import rocklins_linear_regression
import numpy as np

def err(y,y_hat):
    return np.sqrt(np.mean(abs(y_hat-y)**2))

def main():
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    protein_data = pd.read_csv('/Users/meslami/Documents/GitRepos/test-harness/example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv',
                                comment='#', low_memory=False, nrows = 100)


    regression_prediction_col = "stabilityscore_cnn_calibrated"
    classification_prediction_col = "stabilityscore_cnn_calibrated_2classes"

    protein_data.insert(12, classification_prediction_col, protein_data[regression_prediction_col] > 1)
    feature_columns = protein_data.columns.values.tolist()[13:]

    th = TestHarness(output_location='/Users/meslami/Documents/GitRepos/test-harness/tests')

    train_df, test_df = train_test_split(protein_data, test_size=0.2, random_state=5, stratify=protein_data['topology'])

    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                  training_data=train_df, testing_data=test_df,
                  data_and_split_description="Demo Run on Rocklin dataset",
                  cols_to_predict=regression_prediction_col,
                  feature_cols_to_use=feature_columns,
                  index_cols=["dataset", "name"],
                  normalize=True, feature_cols_to_normalize=feature_columns,
                  feature_extraction=False, predict_untested_data=False)

if __name__ == '__main__':
    main()
