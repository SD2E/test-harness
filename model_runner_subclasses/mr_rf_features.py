from test_harness.test_harness_models_abc import RegressionModel
import pandas as pd
import rfpimp


class RFRegression(RegressionModel):
    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _feature_importances(self, X):
        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(X.columns, self.model.feature_importances_):
            feats[feature] = importance  # add the name/value pair

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
        importances['Feature'] = importances.index
        importances.reset_index(inplace=True, drop=True)
        importances = importances[['Feature', 'Importance']]
        importances.sort_values(by='Importance', inplace=True, ascending=False)
        return importances

    def _permutation_importances(self, X, y):
        pimportances = rfpimp.importances(self.model, X, y)
        pimportances['Feature'] = pimportances.index
        pimportances.reset_index(inplace=True, drop=True)
        pimportances = pimportances[['Feature', 'Importance']]
        pimportances.sort_values(by='Importance', inplace=True, ascending=False)
        print(pimportances)
        print()
        return pimportances
