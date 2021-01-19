"""
The purpose of this module is to test out automated input_shape determination for Keras models.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.keras_classification import keras_classification_4
from harness.th_model_instances.hamed_models.keras_regression import keras_regression_best


def main():
    dir_path = Path(__file__).parent
    project_path = Path(dir_path).parent.parent
    data_path = os.path.join(project_path, "example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv")
    protein_data = pd.read_csv(data_path, comment='#', low_memory=False, nrows=1000)
    protein_data.insert(12, "stabilityscore_cnn_calibrated_2classes", protein_data['stabilityscore_cnn_calibrated'] > 1)

    # creating a smaller df with less columns to see what's going on easier
    some_features = protein_data.columns.values.tolist()[13:19]
    protein_data = protein_data[["dataset", "name", "topology", "stabilityscore_cnn_calibrated",
                                 "stabilityscore_cnn_calibrated_2classes"] + some_features]
    # create fake sparse cols
    fake_sparse_col_1 = "fake_sparse_col_1"
    fake_sparse_col_2 = "fake_sparse_col_2"
    protein_data["fake_sparse_col_1"] = ["A", "B"] * 500
    protein_data["fake_sparse_col_2"] = ["C", "A"] * 500
    fake_sparse_cols = [fake_sparse_col_1, fake_sparse_col_2]
    feature_cols_to_use = some_features + fake_sparse_cols

    train_df, test_df = train_test_split(protein_data, test_size=0.2, random_state=5, stratify=protein_data['topology'])
    index_cols = ["dataset", "name"]
    classification_prediction_col = "stabilityscore_cnn_calibrated_2classes"
    regression_prediction_col = "stabilityscore_cnn_calibrated"

    th = TestHarness(output_location=dir_path, compress_large_csvs=True)

    th.run_custom(function_that_returns_TH_model=keras_classification_4, dict_of_function_parameters={"input_shape": None},
                  training_data=train_df, testing_data=test_df,
                  description="testing automated input_shape determination",
                  target_cols=classification_prediction_col,
                  feature_cols_to_use=feature_cols_to_use,
                  index_cols=index_cols,
                  normalize=True, feature_cols_to_normalize=some_features,
                  feature_extraction=False, predict_untested_data=False,
                  sparse_cols_to_use=fake_sparse_cols)


if __name__ == '__main__':
    main()
