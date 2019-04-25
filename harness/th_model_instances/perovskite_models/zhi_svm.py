from sklearn.svm import SVC
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def support_vector_radial_basis_classifier():
    # RBF SVM hyperparameters optimized by Zhi Li for perovskite data classification
    support_vector_model = SVC(C=100000,
                               gamma=0.1,
                               cache_size=5000,
                               max_iter=-1,
                               kernel='rbf',
                               decision_function_shape='ovr',
                               probability=True
                               )
    th_model = SklearnClassification(model=support_vector_model,
                                     model_author="Zhi",
                                     model_description="svm with radial basis function")
    return th_model
