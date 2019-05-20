from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from harness.th_model_classes.class_sklearn_classification import SklearnClassification
import numpy as np
def _graph_kernel(X,Y=None):
    '''
    Define a graph kernel that just returns the adjancency matrix
    :param X: Adjacency matrix of training data
    :param Y: Adjacency matrix of testing data
    :return: return the adjacency matrix
    '''
    if Y is None:
        print("Y is none")
        return np.nan_to_num(X)
    else:
        return np.nan_to_num(Y.transpose())


def label_propagation_classification(kernel='rbf', gamma=20, n_neighbors=7,
                  max_iter=30, tol=1e-3, n_jobs=None):
    '''
    Take in the parameters for label propation. Kernel can be a string or a dict. If it's a dict, it must have two keys:
    a "name" for the kernel and the "method" that is a callable function
    :param kernel: string (rbf, graph, knn) or dict
    :param gamma: if rbf this be used
    :param n_neighbors:  if knn this will be used
    :param max_iter: max number of iterations, default is 30
    :param tol: tolerance for convergence, default is 1e-3
    :param n_jobs: number of cores
    :return: th_model
    '''
    if isinstance(kernel,dict):
        if 'name' not in kernel.keys() or 'method' not in kernel.keys() or len(kernel.keys())>2:
            ValueError("input dictionary must have two keys: name and method")
        kernel_method = kernel['method']
        temp = kernel['name']
        kernel = temp
    elif kernel == 'graph':
        kernel_method = _graph_kernel
    else:
        kernel_method = kernel

    # Creating an sklearn random forest classification model:

    lp = LabelPropagation(kernel=kernel_method, gamma=gamma, n_neighbors=n_neighbors,
                 max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    # Creating an instance of the SklearnClassification TestHarnessModel subclass
    if kernel == 'rbf':
        th_model = SklearnClassification(model=lp, model_author='Mohammed', \
                                         model_description="Label Propagation: kernel={0}, \
                                         gamma={1},  max_iter={2},\
                                         tol={3},n_jobs={4}".format(
                                         kernel, gamma,  max_iter,tol,n_jobs))
    elif kernel == 'knn':
        th_model = SklearnClassification(model=lp, model_author='Mohammed', \
                                         model_description="Label Propagation: kernel={0}, \
                                         n_neighbors={1},  max_iter={2},\
                                         tol={3},n_jobs={4}".format(
                                             kernel, n_neighbors,  max_iter, tol, n_jobs))
    else:
        th_model = SklearnClassification(model=lp, model_author='Mohammed', \
                                         model_description="Label Propagation: kernel={0}, \
                                              max_iter={1},\
                                             tol={2},n_jobs={3}".format(
                                             kernel,  max_iter, tol, n_jobs))
    return th_model