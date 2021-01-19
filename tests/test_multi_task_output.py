from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import warnings
import pandas as pd
import time
import numpy as np
from harness.test_harness_class import TestHarness
from pathlib import Path
from harness.th_model_classes.class_keras_regression import KerasRegression
import itertools


class DE_Network_Embedding_Regression(KerasRegression):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=1000, verbose=0):
        super(DE_Network_Embedding_Regression, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        # checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        # checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        # stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        # callbacks_list = [checkpoint_callback, stopping_callback]
        X.loc[:, 'gene_cat'] = pd.Categorical(X['gene'])
        X.loc[:, 'gene_cat'] = X.gene_cat.cat.codes
        X.loc[:, 'gene_2_cat'] = pd.Categorical(X['gene_2'])
        X.loc[:, 'gene_2_cat'] = X.gene_2_cat.cat.codes

        X1 = X['gene_cat']
        X2 = X['gene_2_cat']
        X3 = X.drop(['gene', 'gene_2', 'gene_cat', 'gene_2_cat'], axis=1)

        # Configure the output
        y_temp = pd.DataFrame(y.tolist(), index=y.index, columns=['logFC_col', 'edge_col'])
        y1 = y_temp['edge_col']
        y2 = y_temp['logFC_col']

        print("Length of ys", len(y1), len(y2))
        self.model.fit([X1, X2, X3], [y1, y2], epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # self.model.load_weights(checkpoint_filepath)
        # os.remove(checkpoint_filepath)

    def _predict(self, X):
        X.loc[:, 'gene_cat'] = pd.Categorical(X['gene'])
        X.loc[:, 'gene_cat'] = X.gene_cat.cat.codes
        X.loc[:, 'gene_2_cat'] = pd.Categorical(X['gene_2'])
        X.loc[:, 'gene_2_cat'] = X.gene_2_cat.cat.codes

        X1 = X['gene_cat']
        X2 = X['gene_2_cat']
        X3 = X.drop(['gene', 'gene_2', 'gene_cat', 'gene_2_cat'], axis=1)

        y1, y2 = self.model.predict([X1, X2, X3])

        merged1 = list(itertools.chain(*y1))

        merged2 = list(itertools.chain(*y2))

        preds = list(zip(merged1, merged2))
        kk = [len(item) for item in preds]

        return preds


def DE_Embedding_Regression_with_network_reg(num_tokens, num_condition_cols, emb_dim=32, batch_size=10000, epochs=25,
                                             learning_rate=1e-3):
    # Gene 1 embedding layer
    input_row_1 = Input(shape=(1,))
    model1 = Embedding(input_dim=num_tokens, output_dim=emb_dim, input_length=1, batch_size=batch_size)(input_row_1)
    model1 = Flatten()(model1)
    model1 = Dense(4, input_dim=emb_dim, activation='relu')(model1)

    # Gene 2 embedding layer
    input_row_2 = Input(shape=(1,))
    model2 = Embedding(input_dim=num_tokens, output_dim=emb_dim, input_length=1, batch_size=batch_size)(input_row_2)
    model2 = Flatten()(model2)
    model2 = Dense(4, input_dim=emb_dim, activation='relu')(model2)

    # Combine gene layers and attach to output
    network_layer = Concatenate()([model1, model2])
    gene_network = Dense(1, activation='softmax', name='network_embedding')(network_layer)

    # Experiment condition input layer
    cond = Input(shape=(num_condition_cols,))
    model3 = Dense(4, input_dim=num_condition_cols, activation='relu')(cond)

    # Combine experiment layer with gene layer
    gene_cond = Concatenate()([model1, model3])
    de_prediction_layer = Dense(1, activation='linear', name='de_pred')(gene_cond)

    # Create model
    model = Model(inputs=[input_row_1, input_row_2, cond], outputs=[de_prediction_layer, gene_network])

    model.compile(loss={'network_embedding': 'binary_crossentropy',
                        'de_pred': 'mean_absolute_error'}, optimizer=Adam(lr=learning_rate),
                  metrics={'network_embedding': 'binary_crossentropy',
                           'de_pred': 'mse'})

    th_model = DE_Network_Embedding_Regression(model=model, model_author="Mohammed",
                                               model_description='DE Gene Embedding Model', batch_size=100, epochs=25)

    return th_model


def generate_gene_network_df(df, network_df, num_random_subset=5):
    unique_genes = np.unique(df['gene'])
    if num_random_subset >= len(unique_genes):
        warnings.warn("The value you chose for num_random_subset is greater than or equal to the number of unique genes.\n"
                      "All unique genes will be used instead of a random subset.")
        random_subset_of_genes = unique_genes.copy()
    else:
        random_subset_of_genes = list(np.random.choice(unique_genes, size=num_random_subset, replace=False))

    start_time = time.time()
    final_df = pd.concat([df.assign(gene_2=g) for g in random_subset_of_genes], ignore_index=True)
    print("\nloop 1 took {} seconds.".format(round(time.time() - start_time, 2)))

    network_df = network_df[["Source", "Target"]]
    edges = list(zip(network_df["Source"], network_df["Target"])) + list(zip(network_df["Target"], network_df["Source"]))

    final_df["gene_pair"] = list(zip(final_df["gene"], final_df["gene_2"]))
    final_df["edge_present"] = 0

    start_time = time.time()
    final_df.loc[final_df["gene_pair"].isin(edges), "edge_present"] = 1
    print("loop 2 took {} seconds.\n".format(round(time.time() - start_time, 2)))
    final_df.drop(columns=["gene_pair"], inplace=True)
    final_df["(logFC, edge_present)"] = list(zip(final_df["logFC"], final_df["edge_present"]))

    col_order_beginning = ["gene", "FDR", "nlogFDR", "logFC", "gene_2", "edge_present", "(logFC, edge_present)"]
    col_order = col_order_beginning + [c for c in list(final_df.columns) if c not in col_order_beginning]
    final_df = final_df[col_order]

    return final_df.copy()


def main():
    # Setup paths and test harness
    dir_path = Path(__file__).parent
    th = TestHarness(output_location=dir_path)

    ###Setup the data
    df = pd.read_csv(
        '/Volumes/GoogleDrive/Shared drives/Netrias_All/Projects/SD2/Novel Chassis/Inducer 1.0/Bacillus/additive_design_df.csv')
    # df = df.sample(frac=0.01, replace=False, random_state=5)

    df.rename({df.columns[0]: 'gene'}, axis=1, inplace=True)
    experiment_cols = ['Cuminic_acid', 'Vanillic_acid', 'Xylose', 'IPTG', 'Timepoint_5']

    network_df = pd.read_csv(
        '/Volumes/GoogleDrive/Shared drives/Netrias_All/Projects/SD2/Novel Chassis/Inducer 1.0/Bacillus/bacillus_net.csv')
    # network_df = network_df.sample(frac=0.01, replace=False, random_state=5)

    df = generate_gene_network_df(df, network_df, 5)

    train_df = df[(~(((df['IPTG'] == 1) & (df['Cuminic_acid'] == 1)) |
                     ((df['IPTG'] == 1) & (df['Vanillic_acid'] == 1))))]
    # (hrm.existing_data['emb_present']==1)]
    test_df = df[(((df['IPTG'] == 1) & (df['Cuminic_acid'] == 1)) |
                  ((df['IPTG'] == 1) & (df['Vanillic_acid'] == 1)))]

    print(len(train_df), len(test_df))
    print()
    print('Train/test columns:')
    print(train_df.columns, test_df.columns)
    print()

    th.run_custom(function_that_returns_TH_model=DE_Embedding_Regression_with_network_reg,
                  dict_of_function_parameters={"num_tokens": len(df['gene'].unique()),
                                               "num_condition_cols": len(experiment_cols),
                                               "batch_size": 100},
                  training_data=train_df,
                  testing_data=test_df,
                  description="random_gene_network_test",
                  target_cols=['(logFC, edge_present)'],
                  feature_cols_to_use=experiment_cols + ['gene', 'gene_2'],
                  index_cols=['gene', 'gene_2'] + experiment_cols,
                  feature_extraction=False, predict_untested_data=False)


if __name__ == '__main__':
    main()
