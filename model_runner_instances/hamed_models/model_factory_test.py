import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
# from test_harness.model_runner_subclasses.mr_sklearn_regression import SklearnRegression
from test_harness.model_runner_subclasses.mr_sklearn_classification import \
    SklearnClassification


def logreg(training_data, testing_data):
    rocklins_logistic_model = LogisticRegression(penalty='l1', C=0.1)

    mr = SklearnClassification(model=rocklins_logistic_model,
                               model_description="Rocklin Logistic: sklearn LogisticRegression with penalty='l1' and C=0.1",
                               topology_specific_or_general='general',
                               training_data=training_data, testing_data=testing_data, data_set_description='15k',
                               train_test_split_description='12k-3k', col_to_predict='stable?',
                               predict_untested=False)

    print(mr.run_model())
    # return mr


training_data = pd.read_csv(
    '/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v0-15k/normalized_data_v0_train.csv')
testing_data = pd.read_csv(
    '/Users/he/PycharmProjects/SD2/protein-design/data_so_far_7-20-18/v0-15k/normalized_data_v0_test.csv')

training_data = training_data.sample(frac=0.0005, random_state=5)
testing_data = testing_data.sample(frac=0.0005, random_state=5)

logreg(training_data, testing_data)
