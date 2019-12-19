from harness.th_model_classes.class_sklearn_classification import SklearnClassification
from .maml_utils import MAML


def maml():
    th_model = SklearnClassification(model=MAML(),
                                     model_author="Dylan Slack",
                                     model_description="Meta learning model")
    return th_model
