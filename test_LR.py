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
    
    for k in range(len(data)):
        err = data[k][1] - np.matmul(data[k][0], w)
        grad = -2 * np.matmul(err, data[k][0]) / batch_size

        for i in np.where(np.abs(grad) > tol)[0]:
            coef_shared[i] -= learning_rate * grad[i]
        
    return

def compute_gradient(data, w):
    err = data[1] - np.matmul(data[0], w)
    grad = -2 * np.matmul(err, data[0]) / batch_size
    return grad

def get_data(total=None):
    """
    Return a list of data
    Each element in the list is the fraction of data that will be processed by the same worker
    """
    
    if total is None:
        filename = '_test_LR_data_%d_%d_%d.npy' % (batch_size, njobs, nsamples_per_job)

        print(batch_size, njobs, nsamples_per_job)
    else:
        filename = '_test_LR_data_%d.npy' % total
        print(total)
        njobs = 1
        nsamples_per_job = total
    
    if os.path.exists(filename):
        arr = numpy.load(filename)
        return arr[0], arr[1]
    
    gt_w = np.random.rand(ndims)
    
    ls = []
    for i in range(njobs):
        current_data = []
        for k in range(nsamples_per_job):
            X = scipy.sparse.random(batch_size, ndims, density=sparse_d).toarray()
            y = np.matmul(X, gt_w)
            current_data.append((X, y))
        ls.append(current_data)
        
    numpy.save(filename, [ls, gt_w])
    return ls, gt_w

def finish(w, gt):
    """
    process trained model
    """
    print("l2 error with gt, ", np.sum((w - gt) ** 2))