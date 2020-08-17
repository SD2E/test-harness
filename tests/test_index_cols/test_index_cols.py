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
    protein_data_1 = protein_data[0:500]
    protein_data_2 = protein_data[500:1000]

    regression_prediction_col = "stabilityscore_cnn_calibrated"
    feature_columns = protein_data.columns.values.tolist()[13:]

    th = TestHarness(output_location=dir_path, compress_large_csvs=False)
    train_df, test_df = train_test_split(protein_data_1, test_size=0.2, random_state=5, stratify=protein_data_1['topology'])
    index_cols = ["dataset", "name"]

    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                  training_data=train_df, testing_data=test_df,
                  description="Demo Run on Rocklin dataset",
                  target_cols=regression_prediction_col,
                  feature_cols_to_use=feature_columns,
                  index_cols=None,  # testing what happens if index_cols=None
                  normalize=True, feature_cols_to_normalize=feature_columns,
                  feature_extraction=False, predict_untested_data=protein_data_2)


if __name__ == '__main__':
    main()