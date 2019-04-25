from sklearn.ensemble import GradientBoostingClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gradient_boosted_tree():
    # Gradient Boosted tree with hyperparams configured on stateset 0021
    xgb = GradientBoostingClassifier(
        learning_rate= 1,
        loss='deviance',
        max_depth= 10,
        max_features='auto',
        n_estimators=100,
    )
    
    th_model = SklearnClassification(model=xgb,
                                     model_author="Scott Novotney",
                                     model_description="gradient boosted tree with n_trees=10,max_depth=5")
    return th_model
