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

lambda_val = 10

has_val = True
has_test = True

update_single_entry = False

def init():
    """
    Return an initialization of weights
    """
    return np.zeros(ndims)

def print_learning_rate():
    print('learning rate now', learning_rate)
    
    
def get_grad(idx, w, coef_shared, data_val):
    grad = np.zeros(w.size)
    for k in idx:
        current_predict = 1 - data_val[k, -1] * np.sum(w * data_val[k, :-1])

        nonzero_ind = np.nonzero(data_val[k, :-1])[0]
        du = nonzero_ind.shape[0]

        for i in nonzero_ind:
            current_grad = 2 * lambda_val * w[i] / du
            if current_predict > 0:
                current_grad -= data_val[k, -1] * data_val[k, i]
            grad[i] -= current_grad
            #coef_shared[i] -= learning_rate * current_grad
    return grad

def shared_train_wrapper(alg, lock=None):
    assert alg in ['hogwild', 'RR']
    def func(idx, w, coef_shared, data_val):
        for k in idx:
            current_predict = 1 - data_val[k, -1] * np.matmul(data_val[k, :-1], w)

            # only approximate, e.g. if one sample has entry -1 and another entry has sample 1, the entry will be incorrectely ignored
            # but this way is much more efficient than taking the union of nonzero sets for each sample
            nonzero_ind = np.nonzero(np.sum(data_val[k, :-1], 0))[0]
            du = nonzero_ind.shape[0]
            
            update_predict = np.any(current_predict) > 0
            
            if alg == 'RR':
                lock.acquire()
            if update_single_entry:
                for i in nonzero_ind:
                    current_grad = 2 * lambda_val * w[i] / du
                    if update_predict:
                        current_grad -= np.sum(data_val[k, -1] * data_val[k, i])
                    coef_shared[i] -= learning_rate * current_grad
            else:
                current_grad = 2 * lambda_val * w / du
                if update_predict:
                    current_grad -= np.sum(np.expand_dims(data_val[k, -1], 1) * data_val[k, :-1], 0)
                current_grad[np.sum(data_val[k, :-1], 0) == 0] = 0
                coef_shared[:] -= learning_rate * current_grad

            if alg == 'RR':
                lock.release()
        
    return func

shared_train_hogwild = shared_train_wrapper('hogwild')
                
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
    
    gt_w = np.random.rand(ndims) - 0.5
    
    X = scipy.sparse.random(total, ndims, density=sparse_d).toarray()
    y = np.sign(np.matmul(X, gt_w))
    ls = np.concatenate((X, np.expand_dims(y, 1)), 1)
    
    X_validate = scipy.sparse.random(total // 10, ndims, density=sparse_d).toarray()
    y_validate = np.sign(np.matmul(X_validate, gt_w))
    ls_validate = np.concatenate((X_validate, np.expand_dims(y_validate, 1)), 1)
    
    X_test = scipy.sparse.random(total // 10, ndims, density=sparse_d).toarray()
    y_test = np.sign(np.matmul(X_test, gt_w))
    ls_test = np.concatenate((X_test, np.expand_dims(y_test, 1)), 1)
        
    numpy.save(filename, [[ls, ls_validate, ls_test], gt_w])
    return [ls, ls_validate, ls_test], gt_w

def finish(w, data, mode='validation'):
    """
    process trained model
    """
    err = np.sum(np.maximum(1 - data[:, -1] * np.matmul(data[:, :-1], w), 0))
    reg = lambda_val * np.sum(w * w)
    print("%s error, " % mode, err, err + reg)
    return err
