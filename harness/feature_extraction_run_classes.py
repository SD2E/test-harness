from math import fabs
from operator import itemgetter
import time

from harness.utils.names import Names

import BlackBoxAuditing as BBA
from BlackBoxAuditing.model_factories.SKLearnModelVisitor import SKLearnModelVisitor
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import pandas as pd
import rfpimp
import shap


plt.switch_backend('agg')


class FeatureExtractionRun:

    def __init__(self, base_run_instance):
        """

        :param base_run_instance: an instantiated _BaseRun class that has trained models
        """
        self.base_run_instance = base_run_instance
        self.shap_values = None
        self.shap_plots_dict = None
        self.feature_importances = None

    # TODO: add different options for eli5.sklearn.permutation_importance (current usage) and eli5.permutation_importance
    def feature_extraction_method(self, method=Names.ELI5_PERMUTATION):
        print("Starting Feature Extraction...")
        start_time = time.time()

        if method == True:
            method = Names.ELI5_PERMUTATION

        if method == Names.ELI5_PERMUTATION:
            pi_object = PermutationImportance(self.base_run_instance.test_harness_model.model)
            pi_object.fit(self.base_run_instance.testing_data[self.base_run_instance.feature_cols_to_use],
                          self.base_run_instance.testing_data[self.base_run_instance.col_to_predict]
                          )
            feature_importances_df = pd.DataFrame()
            feature_importances_df["Feature"] = self.base_run_instance.feature_cols_to_use
            feature_importances_df["Importance"] = pi_object.feature_importances_
            feature_importances_df["Importance_Std"] = pi_object.feature_importances_std_
            feature_importances_df.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = feature_importances_df.copy()
        elif method == Names.RFPIMP_PERMUTATION:
            pis = rfpimp.importances(self.base_run_instance.test_harness_model.model,
                                     self.base_run_instance.testing_data[self.base_run_instance.feature_cols_to_use],
                                     self.base_run_instance.testing_data[self.base_run_instance.col_to_predict])
            pis['Feature'] = pis.index
            pis.reset_index(inplace=True, drop=True)
            pis = pis[['Feature', 'Importance']]
            pis.sort_values(by='Importance', inplace=True, ascending=False)
            self.feature_importances = pis.copy()
        elif method == "sklearn_rf_default":
            pass  # TODO

        elif method == Names.BBA_AUDIT:
            data = self.perform_bba_audit(training_data=self.base_run_instance.training_data.copy(),
                                          testing_data=self.base_run_instance.testing_data.copy(),
                                          features=self.base_run_instance.feature_cols_to_use,
                                          classifier=self.base_run_instance.test_harness_model.model,
                                          col_to_predict=self.base_run_instance.col_to_predict)
            feature_importances_df = pd.DataFrame(data, columns=["Feature", "Importance"])
            self.feature_importances = feature_importances_df.copy()

        elif method == Names.SHAP_AUDIT:
            self.shap_plots_dict = {}
            data = self.perform_shap_audit()
            feature_importances_df = pd.DataFrame(data, columns=["Feature", "Importance"])
            self.feature_importances = feature_importances_df.copy()

        print(("Feature Extraction time with method {0} was: {1:.2f} seconds".format(method, time.time() - start_time)))

    def perform_shap_audit(self):
        """
        From:
        https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20scikit-learn.html
        """
        import warnings
        warnings.filterwarnings('ignore')

        features = self.base_run_instance.feature_cols_to_use
        classifier = self.base_run_instance.test_harness_model
        train_X = self.base_run_instance.training_data[features]
        test_X = self.base_run_instance.testing_data[features]

        shap.initjs()
        # f = lambda x: classifier._predict_proba(x)[:,1]
        f = classifier._predict_proba
        train_X_df = pd.DataFrame(data=train_X, columns=features)
        # train_X_df = self.training_data.copy()
        med = train_X_df.median().values.reshape((1, train_X_df.shape[1]))
        explainer = shap.KernelExplainer(f, med)
        test_X_df = pd.DataFrame(data=test_X, columns=features)
        # test_X_df = self.testing_data.copy()
        shap_values = explainer.shap_values(test_X_df)

        # store shap_values so they can be accessed and output by TestHarness class
        self.shap_values = self.base_run_instance.training_data[self.base_run_instance.index_cols + features].copy()
        self.shap_values[features] = shap_values.copy()

        means = []
        totals_list = [0.0 for f in features]
        for val_list in shap_values:
            for i in range(0, len(val_list)):
                totals_list[i] += fabs(val_list[i])
        means = [(feat, total / len(shap_values)) for feat, total in zip(features, totals_list)]
        # mean_shaps are returned for use as feature_importances
        mean_shaps = sorted(means, key=itemgetter(1), reverse=True)

        # generating plots
        print("Generating SHAP plots!")

        plt.close("all")  # close all previous pyplot figures

        # Base Values vs. Model Output for first prediction
        shap.force_plot(explainer.expected_value, shap_values[0, :], test_X_df.iloc[0, :], matplotlib=True, show=False)
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["example_single_sample_force_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        # TODO: figure out how to save multiple sample force plots as variable
        # # Base Values vs. Model Output for full predictions
        # shap.force_plot(explainer.expected_value, shap_values, train_X_df, matplotlib=True, show=False)
        # fig = plt.gcf()      # get current figure
        # self.shap_plots_dict["force_plot_all_samples"] = fig
        # plt.close("all")     # close all previous pyplot figures

        # Dependencies plots
        for feature in self.base_run_instance.feature_cols_to_use:
            shap.dependence_plot(feature, shap_values, test_X_df, interaction_index='auto', show=False)
            fig = plt.gcf()  # get current figure
            self.shap_plots_dict["dependence_plot_{}".format(feature.upper())] = fig
            plt.close("all")  # close all previous pyplot figures

        # Feature importance summaries in 3 plots: dot, violin, and bar
        shap.summary_plot(shap_values, test_X_df, show=False, plot_type="dot")
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_dot_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        shap.summary_plot(shap_values, test_X_df, show=False, plot_type="violin")
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_violin_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        shap.summary_plot(shap_values, test_X_df, plot_type="bar", show=False)
        fig = plt.gcf()  # get current figure
        self.shap_plots_dict["summary_bar_plot"] = fig
        plt.close("all")  # close all previous pyplot figures

        return mean_shaps

    def perform_bba_audit(self, training_data,
                          testing_data,
                          features,
                          classifier,
                          col_to_predict):
        combined_df = training_data.append(testing_data)
        X = combined_df[features]
        y = pd.DataFrame(combined_df[col_to_predict], columns=[col_to_predict])

        data = BBA.data.load_testdf_only(X, y)
        response_index = len(data[0]) - 1
        auditor = BBA.Auditor()
        auditor.trained_model = SKLearnModelVisitor(classifier, response_index)
        auditor(data)
        print("BBA AUDITOR RESULTS:\n")
        print(auditor._audits_data["ranks"])
        return auditor._audits_data["ranks"]