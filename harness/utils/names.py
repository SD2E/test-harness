class Names:
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"

    # general metrics:
    NUM_FEATURES_USED = "Num Features Used"
    NUM_FEATURES_NORMALIZED = "Num Features Normalized"
    SAMPLES_IN_TRAIN = "Samples In Train"
    SAMPLES_IN_TEST = "Samples In Test"

    # classification metrics:
    NUM_CLASSES = "Num Classes"
    AUC_SCORE = "AUC Score"
    ACCURACY = "Accuracy"
    BALANCED_ACCURACY = "Balanced Accuracy"
    F1_SCORE = "F1 Score"
    PRECISION = "Precision"
    RECALL = "Recall"
    AVERAGE_PRECISION = "Average Precision"

    # regression metrics:
    RMSE = "RMSE"
    R_SQUARED = "R-Squared"

    # Leaderboard Columns
    RUN_ID = "Run ID"
    DATE = "Date"
    TIME = "Time"
    MODEL_NAME = "Model Name"
    MODEL_AUTHOR = "Model Author"
    MODEL_DESCRIPTION = "Model Description"
    COLUMN_PREDICTED = "Column Predicted"
    DESCRIPTION = "Description"
    NORMALIZED = "Normalized"
    FEATURE_EXTRACTION = "Feature Extraction"
    WAS_UNTESTED_PREDICTED = "Was Untested Data Predicted"

    LOO_ID = "Leave-One-Out ID"
    TEST_GROUP = "Test Group"
    DATA_DESCRIPTION = "Data Description"
    GROUPING_DESCRIPTION = "Grouping Description"
    GROUP_INDEX = "group_index"

    # Leaderboards
    CUSTOM_CLASS_LBOARD = "custom_classification_leaderboard"
    CUSTOM_REG_LBOARD = "custom_regression_leaderboard"
    LOO_SUMM_CLASS_LBOARD = "loo_summarized_classification_leaderboard"
    LOO_SUMM_REG_LBOARD = "loo_summarized_regression_leaderboard"
    LOO_FULL_CLASS_LBOARD = "loo_detailed_classification_leaderboard"
    LOO_FULL_REG_LBOARD = "loo_detailed_regression_leaderboard"

    # feature extraction types
    ELI5_PERMUTATION = "eli5_permutation"
    RFPIMP_PERMUTATION = "rfpimp_permutation"
    SHAP_AUDIT = "shap_audit"
    BBA_AUDIT = "bba_audit"

    # settings
    NORMAL_OUTPUT = "normal"
    VERBOSE_OUTPUT = "verbose"

    # output file types
    TRAINING_DATA = 'training_data'
    TESTING_DATA = 'testing_data'
    PREDICTED_DATA = 'predicted_data'
    FEATURE_IMPORTANCES = 'feature_importances'
    OUTPUT_FILES = {TRAINING_DATA: TRAINING_DATA + '.csv',
                    TESTING_DATA: TESTING_DATA + '.csv',
                    PREDICTED_DATA: PREDICTED_DATA + '.csv',
                    FEATURE_IMPORTANCES: FEATURE_IMPORTANCES + '.csv'}

    TEST_HARNESS_RESULTS_DIR = "test_harness_results"
    RUNS_DIR = "runs"
    PREDICT_ONLY = "predict_only"
