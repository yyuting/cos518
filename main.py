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

def train_wrapper(data):
    return model_module.train(data, w, coef_shared)

def async_ML(args):
    print(args.name)
    
    
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
    global model_module
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    init_weights = model_module.init()
    data = model_module.get_data()
    
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)
    
    p = Pool(8) 
    
    T0 = time.time()
    p.map(train_wrapper, data)
    T1 = time.time()
    
    print('async job finished in', T1 - T0, 's')
    model_module.finish(w)
    
def serial_ML(args):
    """
    baseline: train the model sequentially
    """
    print(args.name)
    global model_module
    if model_module is None:
        spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
    
    init_weights = model_module.init()
    data = model_module.get_data()
    
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)
    
    T0 = time.time()
    for i in range(len(data)):
        train_wrapper(data[i])
    T1 = time.time()
    
    print('sequential job finished in', T1 - T0, 's')
    model_module.finish(w)


def main():
    parser = argparse_util.ArgumentParser(description='asyn_ML')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--model_file', dest='model_file', default='sanity_test.py', help='py file that contains model-specific methods, must include init(), train(), get_data(), finish()')
    args = parser.parse_args()
    async_ML(args)
    serial_ML(args)
    
if __name__ == '__main__':
    main()