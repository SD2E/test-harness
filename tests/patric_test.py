import os
import pandas as pd
import numpy as np
from pathlib import Path
from harness.test_harness_class import TestHarness
from harness.th_model_instances.graph_models.label_propagation_classification import label_propagation_classification
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

def mic_kernel(X,Y=None,func=None,COL_NAME=None):
    '''
    Kernel that creates edges between things below the max/min S/R, respectively
    :param X: X dataframe
    :param Y: Y dataframe, if None, Y=X
    :return: numpy matrix that is the kernel
    '''
    max_val = X[X['resistant_phenotype'] == 'Susceptible']['measurement_value'].max()
    min_val = X[X['resistant_phenotype'] == 'Resistant']['measurement_value'].min()
    X['S_pred'] = (X['measurement_value'] <= max_val).apply(int)
    X['R_pred'] = (X['measurement_value'] >= min_val).apply(int)
    if Y is None:
        Y = X
    else:
        Y['S_pred'] = (Y['measurement_value'] <= max_val).apply(int)
        Y['R_pred'] = (Y['measurement_value'] >= min_val).apply(int)
    S = np.matmul(np.matrix(X['S_pred'].values).transpose(), np.matrix(Y['S_pred'].values))
    np.fill_diagonal(S, 0)
    R = np.matmul(np.matrix(X['R_pred'].values).transpose(), np.matrix(Y['R_pred'].values))
    np.fill_diagonal(R, 0)
    K = S+R
    K=K.astype(float)
    return K

def kernel_per_val(X,Y=None,func=mic_kernel,COL_NAME='antibiotic'):
    '''
    Kernel for finding creating edges per antibiotic OR organism. Change COL_NAME for that
    :param X: X dataframe
    :param Y: Y dataframe, if None, Y=X
    :return: dataframe that is the adjacency matrix
    '''
    dfs = []
    if Y is None:
        Y = X
    for val in Y[COL_NAME].unique():
        X_sub = X[X[COL_NAME]==val]
        Y_sub = Y[Y[COL_NAME]==val]
        if len(X_sub)==0:
            K=np.zeros((len(X_sub.index),len(Y_sub.index)))
        else:
            K = func(X_sub,Y_sub)
        df = pd.DataFrame(K,index=X_sub.index,columns=Y_sub.index)
        dfs.append(df)
    df_tot = pd.concat(dfs)
    df_tot = df_tot.fillna(0)
    df_tot.groupby(level=0).sum() #This is where you can define a consensus network function
    return df_tot

def rbf_kernel_single_feature(X,Y=None,func=None,COL_NAME=None):
    if Y is None:
        Y = X
    return rbf_kernel(np.log(X['measurement_value'].values).reshape(-1, 1),np.log(Y['measurement_value'].values).reshape(-1,1))

def index_and_col_debugger(stuff1,stuff2):
    return set(stuff1).intersection(set(stuff2))

def main():
    ## Read in PATRIC data
    df_tot = pd.read_csv('/Users/meslami/Downloads/patric_training.csv')
    df_tot['node'] = df_tot['antibiotic'] + '__' + df_tot['genome_id'].apply(str)
    df_tot.set_index('node', inplace=True)
    print(len(df_tot))
    df_tot=df_tot.loc[~df_tot.index.duplicated(keep='first')]
    sample_sizes = [100,200,500,1000,2000]
    kernels = {'mic':mic_kernel,'rbf':rbf_kernel_single_feature,\
               'mic__antibiotic':kernel_per_val,'mic__genome_id':kernel_per_val,\
               'rbf__antibiotic':kernel_per_val, 'rbf__genome_id': kernel_per_val}
    #kernels={'rbf__antibiotic':kernel_per_val,'rbf':rbf_kernel_single_feature}
    iteration = range(5)
    investigate=[]
    for sample_size in sample_sizes:
        df = df_tot.sample(sample_size)
        #df = df_tot.head(10)
        for kernel in kernels.keys():
            COL_NAME = None
            func = None
            for i in iteration:
                ## Unlabel some of the points to train an SSL model
                rng = np.random.RandomState(42)
                random_unlabeled_points = rng.rand(len(df)) < 0.3
                labels = np.asarray([1 if x == 'Susceptible' else 0 for x in df['resistant_phenotype']])
                labels[random_unlabeled_points] = -1
                df['labels'] = labels
                cols_to_predict = ['labels']
                train1, test1 = train_test_split(df, test_size=0.2, random_state=5)
                test1 = test1[test1['labels'] != -1]
                if '__' in kernel:
                    COL_NAME = kernel.split('__')[1]
                    func_name = kernel.split('__')[0]
                    func = kernels[func_name]
                    adj_train1 = kernels[kernel](X=train1,Y=None,func=func,COL_NAME=COL_NAME)
                    adj_train1['labels'] = train1['labels']
                    adj_test1 = kernels[kernel](X=test1, Y=train1,func=func,COL_NAME=COL_NAME)
                    adj_test1['labels'] = test1['labels']
                    description = '{0} samples;{1} iteration;{2} kernel;{3} object'.format(sample_size,i,func_name,COL_NAME)
                else:
                    adj_train1 = pd.DataFrame(kernels[kernel](train1), index=train1.index, columns=train1.index)
                    adj_train1['labels'] = train1['labels']
                    adj_test1 = pd.DataFrame(kernels[kernel](test1, train1), index=test1.index, columns=train1.index)
                    adj_test1['labels'] = test1['labels']
                    description = '{0} samples;{1} iteration;{2} kernel;'.format(sample_size,i,kernel)

                print("Description:")
                print(description)
                ###########DEBUG!!!
                # print("Index and col of adj train1")
                # print(adj_train1.index.to_list())
                # print(adj_train1.columns.to_list())
                # print(index_and_col_debugger(adj_train1.index.to_list(),adj_train1.columns.to_list()))
                # print("legnth of train1",len(adj_train1))
                # print(adj_train1)
                # print("Index of adj train1 and col of adj test1")
                # print(len(index_and_col_debugger(adj_train1.index.to_list(), adj_test1.columns.to_list())))
                # print("Legnth of test1",len(adj_test1))
                # print("Num cols of test1",len(adj_test1.columns))
                # print(adj_test1)
                # return
                # TestHarness usage starts here, all code before this was just data input and pre-processing.
                #print(df.loc['unchanged_meropenem__Acinetobacter baumannii strain AR_0056'])
                # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
                # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
                # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
                # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.
                examples_folder_path = os.getcwd()
                print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
                th = TestHarness(output_location=examples_folder_path)
                try:
                    th.run_custom(function_that_returns_TH_model=label_propagation_classification, dict_of_function_parameters={'kernel':'graph'}, training_data=adj_train1,
                                  testing_data=adj_test1, data_and_split_description=description,
                                  cols_to_predict=cols_to_predict,index_cols=adj_train1.index.to_list(),
                                  feature_cols_to_use=adj_train1.index.to_list(), normalize=False, feature_cols_to_normalize=None,
                                  feature_extraction=False, predict_untested_data=False,sparse_cols_to_use=None)
                except:
                    #ask Hamed how to write things out to file
                    investigate.append(description)
                    pass
    print("INVESTIIGATE THE FOLLOWING:")
    for item in investigate:
        print(item)



if __name__ == '__main__':
    main()
    ## Read in PATRIC data
    # df = pd.read_csv('/Users/meslami/Downloads/patric_training.csv',nrows=10)
    # df['node'] = df['antibiotic'] + '__' + df['genome_name'].apply(str)
    # df.set_index('node', inplace=True)
    # print(rbf_kernel_single_feature(df))

