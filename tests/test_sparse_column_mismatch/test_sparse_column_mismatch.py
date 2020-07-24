"""
Purpose of this module is to check what happens if there is a mismatch between the sparse column values
in train, test, and predict_untested DataFrames.

For example:
- let's say we have a single sparse_col called "gene".
- the gene column has three possible values: "A", "B", or "C"
- we split our data into train and test, and it just so happens that
  our training data only has rows with genes A and B, but no C. However,
  our testing data has rows with all three genes.
- so when we pass the gene column to sparse_cols_to_use, what will happen?
  My guess is that it will create the sparse cols based on the gene column in the
  training data, which means only 2 columns will be created (gene_A and gene_B).
  So then gene_C is not created, which is a problem

Conclusion:


"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression


def main():
    cwd = os.getcwd()
    project_path = os.path.join(cwd.split("/test-harness/")[0], "test-harness")
    data_path = os.path.join(project_path, "example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv")
    dir_path = Path(__file__).parent
    print(dir_path)

    protein_data = pd.read_csv(data_path, comment='#', low_memory=False, nrows=1000)
    regression_prediction_col = "stabilityscore_cnn_calibrated"
    feature_columns = protein_data.columns.values.tolist()[13:]
    # create fake sparse column
    fake_sparse = ["A", "B"] * int(len(protein_data) / 2)
    protein_data["fake_sparse"] = fake_sparse
    protein_data["fake_sparse"][750:] = "C"

    th = TestHarness(output_location=dir_path, compress_large_csvs=True)
    fake_untested_data = protein_data[100:200]
    train_df = protein_data[200:500]
    test_df = protein_data[500:]

    index_cols = ["dataset", "name", "fake_sparse"]

    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                  training_data=train_df, testing_data=test_df,
                  description="testing sparse columns",
                  target_cols=regression_prediction_col,
                  feature_cols_to_use=feature_columns,
                  index_cols=index_cols,
                  normalize=True, feature_cols_to_normalize=feature_columns,
                  feature_extraction=False, predict_untested_data=fake_untested_data)

    last_run = th.list_of_this_instance_run_ids[-1]
    print(last_run)

    predicted_data = pd.read_csv(os.path.join(dir_path,
                                              "test_harness_results/runs/run_{}/predicted_data.csv.gz".format(last_run)))
    training_data = pd.read_csv(os.path.join(dir_path,
                                             "test_harness_results/runs/run_{}/training_data.csv.gz".format(last_run)))
    testing_data = pd.read_csv(os.path.join(dir_path,
                                            "test_harness_results/runs/run_{}/testing_data.csv.gz".format(last_run)))

    print(predicted_data)
    print()
    print(training_data)
    print()
    print(testing_data)
    print()


if __name__ == '__main__':
    main()
