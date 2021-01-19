import os
import pandas as pd
import numpy as np
from pathlib import Path
from harness.test_harness_class import TestHarness
from harness.th_model_instances.graph_models.label_propagation_classification import label_propagation_classification
from harness.utils.names import Names
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel


def main():
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.

    '''
    adj_matrix = np.asarray([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,1,1],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0]])
    num_nodes = adj_matrix.shape[0]
    #TODO: Insert the train/test split here
    df = pd.DataFrame(adj_matrix,columns=['node_'+str(i) for i in range(num_nodes)])
    df['label']=['red','red',-1,'green','green','green']
    print(df)
    '''
    iris = datasets.load_iris()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    labels = np.copy(iris.target)
    labels[random_unlabeled_points] = -1
    df = pd.DataFrame(iris.data,columns=['col_'+str(i) for i in range(iris.data.shape[1])])
    df['labels']=labels
    df = df[df['labels']!=2]
    feature_cols = ['col_'+str(i) for i in range(iris.data.shape[1])]
    cols_to_predict = ['labels']

    train1, test1 = train_test_split(df, test_size=0.2, random_state=5)
    test1 = test1[test1['labels']!=-1]
    print(train1['labels'].value_counts())
    print(test1['labels'].value_counts())
    kernel = {'name':'rbf_test','method':rbf_kernel}
    # TestHarness usage starts here, all code before this was just data input and pre-processing.

    # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
    # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
    # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
    # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.

    examples_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
    print()
    th = TestHarness(output_location=examples_folder_path)

    th.run_custom(function_that_returns_TH_model=label_propagation_classification, dict_of_function_parameters={'kernel':kernel}, training_data=train1,
                  testing_data=test1, description="Testing label propagation with iris dataset",
                  cols_to_predict=cols_to_predict, index_cols=feature_cols+cols_to_predict,
                  feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=feature_cols,
                  feature_extraction=False, predict_untested_data=False, sparse_cols_to_use=None)




if __name__ == '__main__':
    main()