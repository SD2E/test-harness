from harness.test_harness_models_abstract_classes import ClassificationModel

'''
This class takes in a set of binary inputs and computes a logical OR across those binary inputs
'''
class LogicalORClassification(ClassificationModel):
    def __init__(self, model_author, model_description, verbose=0,model=None):
        super(LogicalORClassification, self).__init__(model, model_author, model_description)
        self.verbose = verbose

    def _fit(self, X, y):
        pass

    def _predict(self, X):
        for i,col in enumerate(X.columns):
            if i ==0:
                sum = X[col].apply(bool)
                continue
            sum = sum | X[col].apply(bool)
        return sum

    def _predict_proba(self, X):
        probs = []
        for i,col in enumerate(X.columns):
            if i ==0:
                sum = X[col].apply(bool)
                continue
            sum = sum | X[col].apply(bool)
        for element in sum:
            if element:
                probs.append([1])
            else:
                probs.append([0])
        return probs

def or_model():
    th_model = LogicalORClassification(model_author="Mohammed", model_description='Logical OR Model')
    return th_model