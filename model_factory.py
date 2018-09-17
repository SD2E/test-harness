import pandas as pd
from BlackBoxAuditing.model_factories.AbstractModelFactory import AbstractModelFactory
from BlackBoxAuditing.model_factories.AbstractModelVisitor import AbstractModelVisitor

class ModelFactory(AbstractModelFactory):
    def __init__(self, *args, **kwargs):
        if "options" in kwargs:
            options = kwargs["options"]
            if "model_runner" in options:
                self.model_runner = options.pop("model_runner")
        super(ModelFactory, self).__init__(all_data=pd.concat([self.model_runner.training_data, self.model_runner.testing_data]).values.tolist(),
                                           headers=[self.model_runner.col_to_predict] + self.model_runner.feature_cols_to_use,
                                           response_header=self.model_runner.col_to_predict,
                                           features_to_ignore=[])

    def build(self, train_set):
        return ModelVisitor(model_runner=self.model_runner)

class ModelVisitor(AbstractModelVisitor):
    def __init__(self, *args, **kwargs):
        self.model_runner = kwargs["model_runner"]
        super(ModelVisitor, self).__init__(model_name='')

    def test(self, test_set, test_name=""):
        preds = self.model_runner._predict(test_set)
        actual = self.model_runner.testing_data[self.model_runner.col_to_predict].values.tolist()
        return list(zip(actual, preds))
