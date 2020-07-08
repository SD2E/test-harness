import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.keras_regression import keras_regression_best
from harness.th_model_instances.hamed_models.keras_classification import keras_classification_4


def main():
    # Set model type to test here (sklearn or keras)
    test_model_type = "sklearn"
    normalize = False
    sparse_cols = None

    cwd = os.getcwd()
    project_path = os.path.join(cwd.split("/test-harness/")[0], "test-harness")
    data_path = os.path.join(project_path, "example_scripts/Data_Sharing_Demo/rocklin_dataset_simplified.csv")

    protein_data = pd.read_csv(data_path, comment='#', low_memory=False, nrows=1000)
    protein_data['stabilityscore_cnn_calibrated_2classes'] = protein_data['stabilityscore_cnn_calibrated'] > 1
    protein_data_1 = protein_data[0:500]
    protein_data_2 = protein_data[500:1000]

    regression_prediction_col = "stabilityscore_cnn_calibrated"
    classification_prediction_col = "stabilityscore_cnn_calibrated_2classes"
    feature_columns = protein_data.columns.values.tolist()[13:]

    harness_output_path = os.path.join(project_path, "tests/test_model_saving_and_predict_only")
    train_df, test_df = train_test_split(protein_data_1, test_size=0.2, random_state=5, stratify=protein_data_1['topology'])
    index_cols = ["dataset", "name"]

    print("\n\n\n************************************** Begin Testing Regression Runs **************************************\n")
    custom_run_preds_tester(harness_output_path=harness_output_path, train_df=train_df, test_df=test_df,
                            target_col=regression_prediction_col, feature_columns=feature_columns, index_cols=index_cols,
                            data_to_predict=protein_data_2, normalize=normalize, sparse_cols=sparse_cols,
                            model_type=test_model_type, run_type="regression")

    print("\n\n\n************************************** Begin Testing Classification Runs **************************************\n")
    custom_run_preds_tester(harness_output_path=harness_output_path, train_df=train_df, test_df=test_df,
                            target_col=classification_prediction_col, feature_columns=feature_columns, index_cols=index_cols,
                            data_to_predict=protein_data_2, normalize=normalize, sparse_cols=sparse_cols,
                            model_type=test_model_type, run_type="classification")

    # TODO: test LOO runs as well...


def custom_run_preds_tester(harness_output_path, train_df, test_df, target_col, feature_columns, index_cols, data_to_predict,
                            normalize=True, sparse_cols=None, model_type="sklearn", run_type="classification"):
    th = TestHarness(output_location=harness_output_path)

    if run_type == "classification":
        if model_type == "sklearn":
            model_to_use = random_forest_classification
        elif model_type == "keras":
            model_to_use = keras_classification_4
        else:
            raise NotImplementedError("{} is not implemented".format(model_type))
    elif run_type == "regression":
        if model_type == "sklearn":
            model_to_use = random_forest_regression
        elif model_type == "keras":
            model_to_use = keras_regression_best
        else:
            raise NotImplementedError("{} is not implemented".format(model_type))
    else:
        raise NotImplementedError("{} is not implemented".format(run_type))

    th.run_custom(function_that_returns_TH_model=model_to_use, dict_of_function_parameters={},
                  training_data=train_df, testing_data=test_df,
                  description="Demo Regression Run on Rocklin dataset",
                  target_cols=target_col,
                  feature_cols_to_use=feature_columns,
                  index_cols=index_cols,
                  normalize=normalize, feature_cols_to_normalize=feature_columns,
                  feature_extraction=False, predict_untested_data=data_to_predict,
                  sparse_cols_to_use=sparse_cols)

    last_run = th.list_of_this_instance_run_ids[-1]
    print("Here is the run_id of the trained model we just saved: {}\n\n\n".format(last_run))

    # read in and print predicted_data.csv generated from setting predict_untested_data=True in run_custom:
    predicted_data_1 = pd.read_csv(os.path.join(harness_output_path,
                                                "test_harness_results/runs/run_{}/predicted_data.csv".format(last_run)))
    # print(predicted_data_1.head(), "\n")

    # now testing out the predict_only method...
    th2 = TestHarness(output_location=harness_output_path)
    th2.predict_only(run_id_of_saved_model=last_run,
                     data_to_predict=data_to_predict,
                     index_cols=index_cols,
                     target_col=target_col,
                     feature_cols_to_use=feature_columns)

    # read in and print predicted_data.csv generated from predict_only method (will overwrite the previous one):
    predicted_data_2 = pd.read_csv(os.path.join(harness_output_path,
                                                "test_harness_results/runs/run_{}/predicted_data.csv".format(last_run)))
    # print(predicted_data_2.head(), "\n")

    # testing if predicted_data_1 has the same or similar predictions to predicted_data_2
    if len(predicted_data_1) != len(predicted_data_2):
        raise Warning("The number of rows in predicted_data_1 is not equal to the number of rows in predicted_data_2!"
                      "\nThere is probably something wrong with one of the Test Harness prediction methods."
                      "\nThis might also break tests below...")
    else:
        print("The number of rows match between predicted_data_1 and predicted_data_2. Good job.")

    if len(predicted_data_1.columns) != len(predicted_data_2.columns):
        raise Warning("The number of columns do not match between predicted_data_1 and predicted_data_2.")
    else:
        print("The number of columns match between predicted_data_1 and predicted_data_2. Good job.")

    preds_col_name = "{}_predictions".format(target_col)
    df_compare_preds = pd.merge(predicted_data_1[["name", preds_col_name]],
                                predicted_data_2[["name", preds_col_name]],
                                how="outer", on="name")
    preds_1 = df_compare_preds["{}_x".format(preds_col_name)]
    preds_2 = df_compare_preds["{}_y".format(preds_col_name)]

    equality_check = np.equal(preds_1, preds_2)
    closeness_check = np.isclose(preds_1, preds_2)
    print("{} out of {} predictions are equal between the two prediction methods.".format(
        np.sum(equality_check), len(equality_check)))
    print("{} out of {} predictions are close between the two prediction methods.".format(
        np.sum(closeness_check), len(closeness_check)))


if __name__ == '__main__':
    main()
