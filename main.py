import argparse_util
import importlib
import importlib.util
import os
import time

# Yuting: shared memory modified from
# https://srome.github.io/Async-SGD-in-Python-Implementing-Hogwild!/

from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy
import numpy as np
from multiprocessing import Pool

coef_shared = None
w = None
model_module = None

nthreads = 8

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TOL = 1e-2

learning_rate = DEFAULT_LEARNING_RATE
tol = DEFAULT_TOL

def hogwild_train_wrapper(data):
    grad = model_module.compute_gradient(data, w)
    for i in np.where(np.abs(grad) > tol)[0]:
        coef_shared[i] -= learning_rate * grad[i]

def async_ML(args):    
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
    global model_module
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    init_weights = model_module.init()
    data, gt = model_module.get_data()
    
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)
    
    p = Pool(nthreads) 
    
    # TODO: only training 1 epoch
    # TODO: should add code to stop at timeout
    
    T0 = time.time()
    p.map(hogwild_train_wrapper, data)
    T1 = time.time()
    
    print('async job finished in', T1 - T0, 's')
    model_module.finish(w, gt)
    
def serial_ML(args):
    """
    baseline: train the model sequentially
    """    
    global model_module
    if model_module is None:
        spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
    
    init_weights = model_module.init()
    data, gt = model_module.get_data()
    
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)
    
    T0 = time.time()
    for i in range(len(data)):
        hogwild_train_wrapper(data[i])
    T1 = time.time()
    
    print('sequential job finished in', T1 - T0, 's')
    model_module.finish(w, gt)


def main():
    parser = argparse_util.ArgumentParser(description='asyn_ML')
    parser.add_argument('--model_file', dest='model_file', default='sanity_test.py', help='py file that contains model-specific methods, must include init(), compute_gradient(), get_data(), finish()')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='set training learning rate')
    parser.add_argument('--tol', dest='tol', type=float, default=DEFAULT_TOL, help='set the threshold when to update the sparse weight entries')
    args = parser.parse_args()
    
    global learning_rate, tol
    learning_rate = args.learning_rate
    tol = args.tol
    
    print(tol, learning_rate)
    
    async_ML(args)
    serial_ML(args)
    
if __name__ == '__main__':
    main()