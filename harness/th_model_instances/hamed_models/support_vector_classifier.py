from sklearn.svm import SVC
from harness.th_model_classes.class_sklearn_classification import SklearnClassification


def support_vector_classifier(C=1, kernel='rbf', gamma=1):
    svc = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=svc, model_author='Hamed', model_description=f"SVC with C={C}, gamma={gamma} and kernel={kernel}")

    return th_model


def linear_SVC():
    lin_svc = SVC(kernel='linear', probability=True)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    th_model = SklearnClassification(model=lin_svc, model_author='Hamed', model_description="Linear SVC")

    return th_model
