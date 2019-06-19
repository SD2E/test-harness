from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def rxn_only_svm():
    model = SVC(kernel='rbf',
                decision_function_shape='ovr',
                probability=True,
                C=0.1,
                )
    return SklearnClassification(model=model,
                                 model_author="Scott Novotney",
                                 model_description="RBF svm w/only rxn features",
                                 )


def rxn_only_xgb():
    model = GradientBoostingClassifier(learning_rate=2,
                                       max_depth=3,
                                       n_estimators=10,
                                       )
    return SklearnClassification(model=model,
                                 model_author="Scott Novotney",
                                 model_description="RBF xgb using only 3 rxn features",
                                 )


def rxn_intuition_svm():
    model = SVC(kernel='rbf',
                decision_function_shape='ovr',
                probability=True,
                C=10,
                )
    return SklearnClassification(model=model,
                                 model_author="Scott Novotney, Joshua Schrier",
                                 model_description="RBF xgb w/org/inorg ratio, inorganic and acid amounts ",
                                 )
