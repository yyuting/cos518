import numpy
import numpy as np
import scipy
import scipy.sparse
import os

"""
Implementation of
HOGWILD!: A Lock-Free Approach to Parallelizing
Stochastic Gradient Descent
Section 2, Sparse SVM
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
tol = 0

lambda_val = 0.001

def init():
    """
    Return an initialization of weights
    """
    return np.zeros(ndims)

def print_learning_rate():
    print('learning rate now', learning_rate)
                
def shared_train_hogwild(idx, w, coef_shared, data_val):
    for k in idx:
        current_predict = 1 - data_val[k, -1] * np.sum(w * data_val[k, :-1])
        
        nonzero_ind = np.nonzero(data_val[k, :-1])[0]
        du = nonzero_ind.shape[0]
        
        update_vector = 0 if current_predict <= 0 else 1
        
        for i in nonzero_ind:
            current_grad = 2 * lambda_val * w[i] / du
            if current_predict > 0:
                current_grad -= data_val[k, -1] * data_val[k, i]
            coef_shared[i] -= learning_rate * current_grad
        
                
def get_data_shared(total):
    """
    Return a list of data
    Each element in the list is the fraction of data that will be processed by the same worker
    """
    
    filename = '_test_sparse_SVM_data_%d_%d.npy' % (total, sparse_d * 100)
    print(total)
    
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