import numpy
import numpy as np
import scipy
import scipy.sparse
import os

"""
example modified from
https://srome.github.io/Async-SGD-in-Python-Implementing-Hogwild!/
"""

DEFAULT_NJOBS = 10000
njobs = DEFAULT_NJOBS

# extreme case, least number of elements to update
DEFAULT_BATCH_SIZE = 1
batch_size = DEFAULT_BATCH_SIZE

DEFAULT_NSAMPLES_PER_JOB = 10
nsamples_per_job = DEFAULT_NSAMPLES_PER_JOB

ndims = 100
sparse_d = 0.2
learning_rate = 0.001
tol = 1e-2

def init():
    """
    Return an initialization of weights
    """
    return np.zeros(ndims)

def print_learning_rate():
    print('learning rate now', learning_rate)


def shared_train_hogwild(idx, w, coef_shared, data_val):
    for k in idx:
        err = data_val[k, -1] - np.matmul(data_val[k, :-1], w)
        nonzero_ind = np.nonzero(data_val[k, :-1])[0]
        for i in nonzero_ind:
            grad = -2 * err * data_val[k, i]
            coef_shared[i] -= learning_rate * grad

def get_data_shared(total):
    """
    Return a list of data
    Each element in the list is the fraction of data that will be processed by the same worker
    """

    filename = '_test_LR_data_%d.npy' % total
    
    if os.path.exists(filename):
        arr = numpy.load(filename)
        return arr[0], arr[1]
    
    gt_w = np.random.rand(ndims)
    
    X = scipy.sparse.random(total, ndims, density=sparse_d).toarray()
    y = np.matmul(X, gt_w)
    ls = np.concatenate((X, np.expand_dims(y, 1)), 1)

    numpy.save(filename, [ls, gt_w])
    return ls, gt_w

def finish(w, gt):
    """
    process trained model
    """
    err = np.sum((w - gt) ** 2)
    print("l2 error with gt, ", np.sum((w - gt) ** 2))
    return err