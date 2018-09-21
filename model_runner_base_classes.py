import os
import time
import inspect
import pandas as pd
from math import sqrt
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
import itertools
import rfpimp
# from test_harness.model_factory import ModelFactory, ModelVisitor
# import BlackBoxAuditing as BBA

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

PWD = os.getcwd()
HERE = os.path.realpath(__file__)
PARENT = os.path.dirname(HERE)
DEFAULT_DATA_PATH = os.path.join(PWD, 'versioned_data/asap/')


# TODO: add checks for instance parameter types to see if they're of the correct type
class ModelRunner(metaclass=ABCMeta):
    def __init__(self, model=LinearRegression(),
                 model_description='Default sklearn Linear Regression', training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stabilityscore',
                 feature_cols_to_use=None, id_col='name', topology_col='topology',
                 topology_specific_or_general='general', predict_untested=None):
        self.model = model
        self.model_description = model_description
        self.training_data = training_data
        self.testing_data = testing_data
        self.data_set_description = data_set_description
        self.train_test_split_description = train_test_split_description
        self.col_to_predict = col_to_predict
        self.feature_cols_to_use = feature_cols_to_use
        self.id_col = id_col
        self.topology_col = topology_col
        self.topology_specific_or_general = topology_specific_or_general
        self.predict_untested = predict_untested
        self.stack_trace = inspect.stack()
        self.feature_importances = None
        self.permutation_importances = None

        if (self.training_data is None and self.testing_data is None and self.data_set_description is None and
                self.train_test_split_description is None):
            train_path = os.path.join(DEFAULT_DATA_PATH, 'consistent_training_data_v1.asap.csv')
            test_path = os.path.join(DEFAULT_DATA_PATH, 'consistent_testing_data_v1.asap.csv')
            if (not os.path.isfile(train_path) and not os.path.isfile(test_path)):
                raise IOError("Training or Testing default data does not exist in the default data path. Perhaps you forgot to download the data?")
            self.training_data = pd.read_csv(train_path, comment='#', sep=',')
            self.testing_data = pd.read_csv(test_path, comment='#', sep=',')
            self.data_set_description = 'Default: All V1 Data'
            self.train_test_split_description = """Default: Random split of 80% train and 20% test. Stratified on "stable?" and "library". random_state=5."""
            self.default_data_set_used = True
        elif (self.training_data is not None and self.testing_data is not None and
              self.data_set_description is not None and self.train_test_split_description is not None):
            self.default_data_set_used = False
        # TODO: add check for isinstance pandas dataframe
        else:
            raise ValueError(""""training_data", "testing_data", "data_set_description", and 
                                "train_test_split_description" must all be None or all be Pandas Dataframes.""")

        if self.predict_untested is None:
            untested_path = os.path.join(DEFAULT_DATA_PATH, 'normalized_and_cleaned_untested_designs_v1_with_lib.asap.csv')
            if not os.path.isfile(untested_path):
                raise IOError("Untested default data does not exist in the default data path. Perhaps you forgot to download the data?")
            self.predict_untested = pd.read_csv(untested_path, comment='#', sep=',')
        elif (not self.predict_untested is False and not isinstance(self.predict_untested, pd.DataFrame)):
            raise ValueError("'predict_untested must be None, False, or a Pandas Dataframe.")

        if self.feature_cols_to_use is None:
            self.feature_cols_to_use = ['AlaCount', 'T1_absq', 'T1_netq', 'Tend_absq', 'Tend_netq', 'Tminus1_absq',
                                        'Tminus1_netq', 'abego_res_profile', 'abego_res_profile_penalty',
                                        'avg_all_frags', 'avg_best_frag', 'bb', 'buns_bb_heavy', 'buns_nonheavy',
                                        'buns_sc_heavy', 'buried_minus_exposed', 'buried_np', 'buried_np_AFILMVWY',
                                        'buried_np_AFILMVWY_per_res', 'buried_np_per_res', 'buried_over_exposed',
                                        'chymo_cut_sites', 'chymo_with_LM_cut_sites', 'contact_all',
                                        'contact_core_SASA', 'contact_core_SCN', 'contig_not_hp_avg',
                                        'contig_not_hp_avg_norm', 'contig_not_hp_internal_max', 'contig_not_hp_max',
                                        'degree', 'dslf_fa13', 'entropy', 'exposed_hydrophobics',
                                        'exposed_np_AFILMVWY', 'exposed_polars', 'exposed_total', 'fa_atr',
                                        'fa_atr_per_res', 'fa_dun_dev', 'fa_dun_rot', 'fa_dun_semi', 'fa_elec',
                                        'fa_intra_atr_xover4', 'fa_intra_elec', 'fa_intra_rep_xover4',
                                        'fa_intra_sol_xover4', 'fa_rep', 'fa_rep_per_res', 'fa_sol', 'frac_helix',
                                        'frac_loop', 'frac_sheet', 'fxn_exposed_is_np', 'hbond_bb_sc', 'hbond_lr_bb',
                                        'hbond_lr_bb_per_sheet', 'hbond_sc', 'hbond_sr_bb', 'hbond_sr_bb_per_helix',
                                        'helix_sc', 'holes', 'hphob_sc_contacts', 'hphob_sc_degree', 'hxl_tors',
                                        'hydrophobicity', 'largest_hphob_cluster', 'lk_ball', 'lk_ball_bridge',
                                        'lk_ball_bridge_uncpl', 'lk_ball_iso', 'loop_sc', 'mismatch_probability',
                                        'n_charged', 'n_hphob_clusters', 'n_hydrophobic', 'n_hydrophobic_noA',
                                        'n_polar_core', 'n_res', 'nearest_chymo_cut_to_Cterm',
                                        'nearest_chymo_cut_to_Nterm', 'nearest_chymo_cut_to_term',
                                        'nearest_tryp_cut_to_Cterm', 'nearest_tryp_cut_to_Nterm',
                                        'nearest_tryp_cut_to_term', 'net_atr_net_sol_per_res', 'net_atr_per_res',
                                        'net_sol_per_res', 'netcharge', 'nres', 'nres_helix', 'nres_loop', 'nres_sheet',
                                        'omega', 'one_core_each', 'p_aa_pp', 'pack', 'percent_core_SASA',
                                        'percent_core_SCN', 'pro_close', 'rama_prepro', 'ref', 'res_count_core_SASA',
                                        'res_count_core_SCN', 'score_per_res', 'ss_contributes_core',
                                        'ss_sc', 'sum_best_frags', 'total_score', 'tryp_cut_sites', 'two_core_each',
                                        'worst6frags', 'worstfrag']

        elif not isinstance(self.feature_cols_to_use, list) and \
                not isinstance(self.feature_cols_to_use, dict):
            raise ValueError(
                'feature_cols_to_use must be a list, a dict, or None. None defaults to all available Rosetta features.')

        if self.topology_specific_or_general != 'general' and self.topology_specific_or_general != 'specific':
            raise ValueError("'topology_specific_or_general' must be a string equal to 'general' or 'specific'")

    @abstractmethod
    def _fit(self, X, y):
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def _topology_general_predictions(self, train, test, predict):
        pass

    @abstractmethod
    def _topology_specific_predictions(self, train, test, predict):
        pass

    @abstractmethod
    def run_model(self, train, test, predict):
        pass

    def splits_by_columns(self, cols=['topology', 'library']):
        all_data = pd.concat([self.training_data.copy(), self.testing_data.copy()])
        unique_combos = (all_data[cols].drop_duplicates())

        combinations = []
        for index, row in unique_combos.iterrows():
            combinations.append(dict(row))
        print(combinations)

        print("Number of train/test split combinations to iterate through =", len(combinations))

        splits_results = pd.DataFrame()
        splits_features = None
        for combo in combinations:
            test_split = all_data.copy()
            train_split = all_data.copy()
            print("Test split based on:", combo)
            train_split = train_split.loc[~(train_split[list(combo)] == pd.Series(combo)).all(axis=1)]
            test_split = test_split.loc[(test_split[list(combo)] == pd.Series(combo)).all(axis=1)]
            print("Number of samples in train split:", train_split.shape)
            print("Number of samples in test split:", test_split.shape)
            this_run_results = self.run_model(train_split, test_split, None)
            this_run_results['test_split'] = str(combo)
            splits_results = pd.concat([splits_results, this_run_results])

            # this_run_features = self.feature_importances
            # this_run_features.rename(columns={'Importance': combo['library'] + '_' + combo['topology']}, inplace=True)
            # if isinstance(this_run_features, pd.DataFrame):
            #     if splits_features is None:
            #         splits_features = this_run_features
            #     else:
            #         splits_features = pd.merge(splits_features, this_run_features, on='Feature')
            # print()

            this_run_perms = self.permutation_importances
            this_run_perms.rename(columns={'Importance': combo['library'] + '_' + combo['topology']}, inplace=True)
            if isinstance(this_run_perms, pd.DataFrame):
                if splits_features is None:
                    splits_features = this_run_perms
                else:
                    splits_features = pd.merge(splits_features, this_run_perms, on='Feature')
            print()

        return splits_results, splits_features

    def custom_splits(self, grouping_df):
        all_data = pd.concat([self.training_data.copy(), self.testing_data.copy()])

        relevant_groupings = grouping_df.copy()
        relevant_groupings = relevant_groupings.loc[(relevant_groupings['library'].isin(all_data['library'])) &
                                                    (relevant_groupings['topology'].isin(all_data['topology']))]
        print(relevant_groupings)
        print()

        splits_results = pd.DataFrame()
        splits_features = None
        for group in list(set(relevant_groupings['group_index'])):
            train_split = all_data.copy()
            test_split = all_data.copy()
            print("Test split based on group {}:".format(group))
            group_df = relevant_groupings.loc[relevant_groupings['group_index'] == group]
            print(group_df)
            train_split = train_split.loc[~((train_split['library'].isin(group_df['library'])) &
                                            (train_split['topology'].isin(group_df['topology'])))]
            test_split = test_split.loc[(test_split['library'].isin(group_df['library'])) &
                                        (test_split['topology'].isin(group_df['topology']))]
            print(train_split.shape)
            print(test_split.shape)

            print("Number of samples in train split:", train_split.shape)
            print("Number of samples in test split:", test_split.shape)
            this_run_results = self.run_model(train_split, test_split, None)
            this_run_results['test_split'] = str(list(set(group_df['library'])) + list(set(group_df['topology'])))
            splits_results = pd.concat([splits_results, this_run_results])

            this_run_perms = self.permutation_importances
            this_run_perms.rename(
                columns={'Importance': str(list(set(group_df['library'])) + list(set(group_df['topology'])))},
                inplace=True)
            if isinstance(this_run_perms, pd.DataFrame):
                if splits_features is None:
                    splits_features = this_run_perms
                else:
                    splits_features = pd.merge(splits_features, this_run_perms, on='Feature')
            print()

        return splits_results, splits_features


class ClassificationModelRunner(ModelRunner, metaclass=ABCMeta):
    def __init__(self, model=LogisticRegression(),
                 model_description='Default sklearn Logistic Regression Classifier', training_data=None,
                 testing_data=None, data_set_description=None, train_test_split_description=None,
                 col_to_predict='stable?', feature_cols_to_use=None, id_col='name', topology_col='topology',
                 topology_specific_or_general='general', predict_untested=None):
        super(ClassificationModelRunner, self).__init__(model, model_description, training_data,
                                                        testing_data, data_set_description,
                                                        train_test_split_description, col_to_predict,
                                                        feature_cols_to_use, id_col, topology_col,
                                                        topology_specific_or_general, predict_untested)
        self.type = 'classification'

    # in subclasses, this method should return probability values for being in the positive (1) class
    @abstractmethod
    def _predict_proba(self, X):
        pass

    # TODO: fix predict dataframe interactions
    def _topology_general_predictions(self, train, test, predict):
        train_df = train.copy()
        test_df = test.copy()

        training_start_time = time.time()
        self._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("classifier training time was: {}".format(time.time() - training_start_time)))

        testing_start_time = time.time()
        test_df.loc[:, 'mr_class_predictions'] = self._predict(test_df[self.feature_cols_to_use])
        print(("classifier testing time was: {}".format(time.time() - testing_start_time)))

        test_df.loc[:, 'mr_class_probability_predictions'] = self._predict_proba(test_df[self.feature_cols_to_use])

        if self.predict_untested is not False:
            prediction_start_time = time.time()
            self.predict_untested.loc[:, 'mr_class_predictions'] = self._predict(
                self.predict_untested[self.feature_cols_to_use])
            self.predict_untested.loc[:, 'mr_class_probability_predictions'] = self._predict_proba(
                self.predict_untested[self.feature_cols_to_use])
            print(("class prediction time was: {}".format(time.time() - prediction_start_time)))
            self.predict_untested.sort_values('mr_class_predictions', inplace=True, ascending=False)

        return test_df

    # TODO: incorporate timing for topology specific predictions
    def _topology_specific_predictions(self, train, test, predict):
        train_df = train.copy()
        test_df = test.copy()

        topologies_in_train_df = set(train_df[self.topology_col].tolist())
        topologies_in_test_df = set(test_df[self.topology_col].tolist())
        if topologies_in_train_df != topologies_in_test_df:
            raise ValueError(
                'Topologies in train_df and test_df are not matching up. They must consist of the same elements.')
        topologies = list(topologies_in_train_df.intersection(topologies_in_test_df))

        for t in topologies:
            t_train_df = train_df.loc[train_df[self.topology_col] == t].copy()
            t_test_df = test_df.loc[test_df[self.topology_col] == t].copy()
            if isinstance(self.feature_cols_to_use, list):
                cols_to_use = list(self.feature_cols_to_use)
            elif isinstance(self.feature_cols_to_use, dict):
                cols_to_use = self.feature_cols_to_use[t]
            else:
                raise ValueError("'self.feature_cols_to_use' must be a list or a dict")

            self._fit(t_train_df[cols_to_use], t_train_df[self.col_to_predict])
            test_df.loc[test_df[self.topology_col] == t, 'mr_class_predictions'] = self._predict(t_test_df[cols_to_use])
            test_df.loc[test_df[self.topology_col] == t, 'mr_class_probability_predictions'] = self._predict_proba(
                t_test_df[cols_to_use])

        return test_df

    def model_factory(self):
        mf = ModelFactory
        auditor = BBA.Auditor()
        auditor.ModelFactory = mf
        auditor.model_options = {"model_runner": self}

        types_list = [float for x in self.feature_cols_to_use]
        to_pred_index = self.training_data.columns.get_loc(self.col_to_predict)
        types_list[to_pred_index] = str
        datatuple = (
        [self.col_to_predict] + self.feature_cols_to_use, self.training_data[self.feature_cols_to_use].values.tolist(),
        self.testing_data[self.feature_cols_to_use].values.tolist(), self.col_to_predict,
        [], types_list)
        auditor(data=datatuple,
                output_dir='/Users/he/PycharmProjects/SD2/protein-design/test_harness/model_factory_output',
                features_to_audit=self.feature_cols_to_use)
        # TODO: parse summary.txt

    def run_model(self, train=None, test=None, predict=None):
        if train is None:
            train = self.training_data
        if test is None:
            test = self.testing_data
        if predict is None:
            predict = self.predict_untested

        leaderboard_cols = ['Run ID', 'AUC Score', 'Classification Accuracy', 'Model Description',
                            'Number Of Features Used', 'Column Predicted', 'Data Set Description',
                            'Train/Test Split Description', 'Topology Specific or General?']
        this_run_results = pd.DataFrame(columns=leaderboard_cols)

        if self.topology_specific_or_general == 'general':
            df_test = self._topology_general_predictions(train, test, predict)
        elif self.topology_specific_or_general == 'specific':
            df_test = self._topology_specific_predictions(train, test, predict)
        else:
            raise ValueError("'self.topology_specific_or_general' must take on a value of 'specific' or 'general'")

        self.test_predictions_df = df_test.copy()

        if isinstance(self.feature_cols_to_use, list):
            num_features_used = len(self.feature_cols_to_use)
        elif isinstance(self.feature_cols_to_use, dict):
            num_features_used = {}
            for topology, feature_list in list(self.feature_cols_to_use.items()):
                num_features_used[topology] = len(feature_list)
        else:
            raise ValueError("'self.feature_cols_to_use' must be of type list or dict.")

        num_rows = len(df_test)
        total_equal = sum(df_test[self.col_to_predict] == df_test['mr_class_predictions'])
        percent_accuracy = float(total_equal) / float(num_rows)
        # auc = roc_auc_score(df_test[self.col_to_predict], df_test['mr_class_probability_predictions'])
        auc = 'n/a'

        # self.model_factory()

        row_to_add = {'Model Description': self.model_description,
                      'Data Set Description': self.data_set_description,
                      'Train/Test Split Description': self.train_test_split_description,
                      'Column Predicted': self.col_to_predict, 'Number Of Features Used': num_features_used,
                      'Topology Specific or General?': self.topology_specific_or_general,
                      'Classification Accuracy': percent_accuracy, 'AUC Score': auc, 'Run ID': 'n/a'}
        this_run_results = this_run_results.append(row_to_add, ignore_index=True)
        # print(tabulate(this_run_results, headers='keys', tablefmt='fancy_grid'))

        return this_run_results


class RegressionModelRunner(ModelRunner, metaclass=ABCMeta):
    def __init__(self, model=LinearRegression(),
                 model_description='Default sklearn Linear Regression', training_data=None, testing_data=None,
                 data_set_description=None, train_test_split_description=None, col_to_predict='stabilityscore',
                 feature_cols_to_use=None, id_col='name', topology_col='topology',
                 topology_specific_or_general='general', predict_untested=None):
        super(RegressionModelRunner, self).__init__(model, model_description, training_data,
                                                    testing_data, data_set_description,
                                                    train_test_split_description, col_to_predict,
                                                    feature_cols_to_use, id_col, topology_col,
                                                    topology_specific_or_general, predict_untested)
        self.type = 'regression'

    def _topology_general_predictions(self, train, test, predict):
        train_df = train.copy()
        test_df = test.copy()

        training_start_time = time.time()
        self._fit(train_df[self.feature_cols_to_use], train_df[self.col_to_predict])
        print(("regressor training time was: {}".format(time.time() - training_start_time)))

        testing_start_time = time.time()
        test_df.loc[:, 'mr_value_predictions'] = self._predict(test_df[self.feature_cols_to_use])
        test_df['residuals'] = test_df[self.col_to_predict] - test_df['mr_value_predictions']
        print(("regressor testing time was: {}".format(time.time() - testing_start_time)))

        _feature_importances = getattr(self, "_feature_importances", None)
        if callable(_feature_importances):
            self.feature_importances = _feature_importances(train_df[self.feature_cols_to_use])

        _permutation_importances = getattr(self, "_permutation_importances", None)
        if callable(_permutation_importances):
            permutation_start_time = time.time()
            self.permutation_importances = _permutation_importances(test_df[self.feature_cols_to_use],
                                                                    test_df[self.col_to_predict])
            print(("permutation feature importance time was: {}".format(time.time() - permutation_start_time)))

        if self.predict_untested is not False:
            prediction_start_time = time.time()
            self.predict_untested.loc[:, 'mr_value_predictions'] = self._predict(
                self.predict_untested[self.feature_cols_to_use])
            print(("regressor prediction time was: {}".format(time.time() - prediction_start_time)))
            self.predict_untested.sort_values('mr_value_predictions', inplace=True, ascending=False)

        return test_df

    # TODO: incorporate timing for topology specific predictions
    def _topology_specific_predictions(self, train, test, predict):
        train_df = train.copy()
        test_df = test.copy()
        topologies_in_train_df = set(train_df[self.topology_col].tolist())
        topologies_in_test_df = set(test_df[self.topology_col].tolist())
        if topologies_in_train_df != topologies_in_test_df:
            raise ValueError(
                'Topologies in train_df and test_df are not matching up. They must consist of the same elements.')
        topologies = list(topologies_in_train_df.intersection(topologies_in_test_df))

        for t in topologies:
            t_train_df = train_df.loc[train_df[self.topology_col] == t].copy()
            t_test_df = test_df.loc[test_df[self.topology_col] == t].copy()
            if isinstance(self.feature_cols_to_use, list):
                cols_to_use = list(self.feature_cols_to_use)
            elif isinstance(self.feature_cols_to_use, dict):
                cols_to_use = self.feature_cols_to_use[t]
            else:
                raise ValueError("'self.feature_cols_to_use' must be a list or a dict")

            self._fit(t_train_df[cols_to_use], t_train_df[self.col_to_predict])
            test_df.loc[test_df[self.topology_col] == t, 'mr_value_predictions'] = self._predict(t_test_df[cols_to_use])
        test_df['residuals'] = test_df[self.col_to_predict] - test_df['mr_value_predictions']

        return test_df

    def run_model(self, train=None, test=None, predict=None):
        if train is None:
            train = self.training_data
        if test is None:
            test = self.testing_data
        if predict is None:
            predict = self.predict_untested

        leaderboard_cols = ['Run ID', 'RMSE', 'Percent Error', 'R Squared', 'Model Description',
                            'Number Of Features Used', 'Column Predicted', 'Data Set Description',
                            'Train/Test Split Description', 'Topology Specific or General?']
        this_run_results = pd.DataFrame(columns=leaderboard_cols)

        if self.topology_specific_or_general == 'general':
            df_test = self._topology_general_predictions(train, test, predict)
        elif self.topology_specific_or_general == 'specific':
            df_test = self._topology_specific_predictions(train, test, predict)
        else:
            raise ValueError("'self.topology_specific_or_general' must take on a value of 'specific' or 'general'")

        self.test_predictions_df = df_test.copy()

        if isinstance(self.feature_cols_to_use, list):
            num_features_used = len(self.feature_cols_to_use)
        elif isinstance(self.feature_cols_to_use, dict):
            num_features_used = {}
            for topology, feature_list in list(self.feature_cols_to_use.items()):
                num_features_used[topology] = len(feature_list)
        else:
            raise ValueError("'self.feature_cols_to_use' must be of type list or dict.")

        rmse = sqrt(mean_squared_error(df_test[self.col_to_predict], df_test['mr_value_predictions']))
        range = df_test[self.col_to_predict].max() - df_test[self.col_to_predict].min()
        percent_error = (float(rmse) / range) * 100
        r_squared = r2_score(df_test[self.col_to_predict], df_test['mr_value_predictions'])

        row_to_add = {'Model Description': self.model_description,
                      'Data Set Description': self.data_set_description,
                      'Train/Test Split Description': self.train_test_split_description,
                      'Column Predicted': self.col_to_predict, 'Number Of Features Used': num_features_used,
                      'Topology Specific or General?': self.topology_specific_or_general, 'RMSE': rmse,
                      'Run ID': 'n/a', 'Percent Error': percent_error, 'R Squared': r_squared}
        this_run_results = this_run_results.append(row_to_add, ignore_index=True)
        # print(this_run_results)

        return this_run_results
