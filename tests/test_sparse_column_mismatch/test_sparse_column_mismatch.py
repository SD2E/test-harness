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
The _add_sparse_cols method in run_classes was written in a way that accounts for mismatches.
For each sparse_col, it takes a union of all its values in training, testing, and untested data,
and it creates a dummy column for all of those values in all 3 datasets. So if a value didn't
exist in the testing data, but did exist in the training data, then the testing data
will still get a dummy column for that value but it will have all 0s in it. Feel free to
run this script to get a more in depth look at what I'm talking about.
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

    # create fake sparse column
    fake_sparse = ["A", "B"] * int(len(protein_data) / 2)
    protein_data["fake_sparse"] = fake_sparse
    protein_data["fake_sparse"][750:] = "C"

    feature_cols_to_use = some_features + ["fake_sparse"]
    regression_prediction_col = "stabilityscore_cnn_calibrated"

    train_df = protein_data[200:500]
    test_df = protein_data[500:]
    fake_untested_data = protein_data[:200]
    index_cols = ["dataset", "name", "fake_sparse"]

    th = TestHarness(output_location=dir_path, compress_large_csvs=True)

    run_object = th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={},
                               training_data=train_df, testing_data=test_df,
                               description="testing sparse columns",
                               target_cols=regression_prediction_col,
                               feature_cols_to_use=feature_cols_to_use,
                               index_cols=index_cols,
                               normalize=True, feature_cols_to_normalize=some_features,
                               feature_extraction=False, predict_untested_data=fake_untested_data,
                               sparse_cols_to_use=["fake_sparse"],
                               execute=False)

    print(run_object.training_data.head())
    print()
    print(run_object.testing_data.head())
    print()
    print(run_object.predict_untested_data.head())
    print()

    datasets = [run_object.training_data, run_object.testing_data, run_object.predict_untested_data]
    dataset_names = ["training_data", "testing_data", "predict_untested_data"]
    for dataset, name in zip(datasets, dataset_names):
        print(name)
        print("There should be {} As, {} Bs, and {} Cs.".format(
            (dataset["unchanged_fake_sparse"] == "A").sum(),
            (dataset["unchanged_fake_sparse"] == "B").sum(),
            (dataset["unchanged_fake_sparse"] == "C").sum()))
        print("sum of fake_sparse_A: {}".format(dataset["fake_sparse_A"].sum()))
        print("sum of fake_sparse_B: {}".format(dataset["fake_sparse_B"].sum()))
        print("sum of fake_sparse_C: {}".format(dataset["fake_sparse_C"].sum()))
        print()


if __name__ == '__main__':
    main()
