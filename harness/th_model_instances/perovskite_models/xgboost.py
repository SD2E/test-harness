from sklearn.ensemble import GradientBoostingClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def gradient_boosted_tree(learning_rate=1, loss='deviance', max_depth=10, max_features='auto', n_estimators=100):
    # Gradient Boosted tree with hyperparams configured on stateset 0021
    xgb = GradientBoostingClassifier(
        learning_rate= learning_rate,
        loss=loss,
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
    )
    
    th_model = SklearnClassification(model=xgb,
                                     model_author="Scott Novotney",
                                     model_description=f"gradient boosted tree with n_trees={n_estimators},max_depth={max_depth}")
    return th_model
