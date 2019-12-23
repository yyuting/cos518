import numpy
import numpy as np
import scipy
import scipy.sparse
import os

"""
example modified from
https://srome.github.io/Async-SGD-in-Python-Implementing-Hogwild!/
"""

njobs = 10000
# extreme case, least number of elements to update
batch_size = 1

ndims = 1000
sparse_d = 0.2
learning_rate = 0.001
tol = 1e-2

def init():
    """
    Return an initialization of weights
    """
    return np.zeros(ndims)

def train(data, w, coef_shared):
    
    err = data[1] - np.matmul(data[0], w)
    grad = -2 * np.matmul(err, data[0]) / batch_size
        
    for i in np.where(np.abs(grad) > tol)[0]:
        coef_shared[i] -= learning_rate * grad[i]
        
    return

def get_data():
    """
    Return a list of data
    Each element in the list is the fraction of data that will be processed by the same worker
    """
    
    filename = '_test_LR_data.npy'
    if os.path.exists(filename):
        arr = numpy.load(filename)
        return arr[0], arr[1]
    
    gt_w = np.random.rand(ndims)
    
    ls = []
    for i in range(njobs):
        X = scipy.sparse.random(batch_size, ndims, density=sparse_d).toarray()
        y = np.matmul(X, gt_w)
        ls.append((X, y))
        
    numpy.save(filename, [ls, gt_w])
    return ls, gt_w

def finish(w, gt):
    """
    process trained model
    """
    print("l2 error with gt, ", np.sum((w - gt) ** 2))