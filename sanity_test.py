import numpy
import numpy as np

njobs = 1000
batch_size = 200
nsamples_per_job = 1

def init():
    """
    Return an initialization of weights
    """
    return np.zeros(10)

def train(data, w, coef_shared):
    
    for i in range(data[0].shape[0]):
        idx = int(np.floor(data[0][i] * 10))
        coef_shared[idx] += 1
    return

def compute_gradient(data, w):
    grad = np.zeros(w.shape)
    for i in range(data.shape[0]):
        idx = int(np.floor(data[i] * 10))
        grad[idx] -= 1.
    return grad
        
def get_data():
    """
    Return a list of data
    Each element in the list is the fraction of data that will be processed by the same worker
    """
    ls = []
    for i in range(njobs):
        ls.append([np.random.rand(batch_size)])
    return ls, None

def finish(w, gt):
    """
    process trained model
    """
    print(np.sum(w), '/', njobs * batch_size)
    print(w)