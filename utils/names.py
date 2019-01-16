class Names:
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"

    # general metrics:
    NUM_FEATURES_USED = "Num Features Used"
    NUM_FEATURES_NORMALIZED = "Num Features Normalized"
    NUM_SAMPLES_IN_TEST = "Num Samples In Test Set"

    # classification metrics:
    AUC_SCORE = "AUC Score"
    ACCURACY = "Accuracy"
    F1_SCORE = "F1 Score"
    PRECISION = "Precision"
    RECALL = "Recall"

    # regression metrics:
    RMSE = "RMSE"
    R_SQUARED = "R-Squared"

    # Leaderboard Columns
    RUN_ID = "Run ID"
    DATE = "Date"
    TIME = "Time"
    MODEL_DESCRIPTION = "Model Description"
    COLUMN_PREDICTED = "Column Predicted"
    DATA_AND_SPLIT_DESCRIPTION = "Data and Split Description"
    NORMALIZED = "Normalized"
    FEATURE_EXTRACTION = "Feature Extraction"
    WAS_UNTESTED_PREDICTED = "Was Untested Data Predicted"

    LOO_ID = "Leave-One-Out ID"
    TEST_GROUP = "Test Group"
    DATA_DESCRIPTION = "Data Description"
    GROUPING_DESCRIPTION = "Grouping Description"

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
