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


DEFAULT_NTHREADS = 8
nthreads = DEFAULT_NTHREADS

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TOL = 1e-2

learning_rate = DEFAULT_LEARNING_RATE
tol = DEFAULT_TOL

DEFAULT_BATCH_SIZE = 1
batch_size = DEFAULT_BATCH_SIZE

DEFAULT_NJOBS = 10000
njobs = DEFAULT_NJOBS

DEFAULT_NSAMPLES_PER_JOB = 10
nsamples_per_job = DEFAULT_NSAMPLES_PER_JOB

def hogwild_train_wrapper(data):
    for k in range(len(data)):
        grad = model_module.compute_gradient(data[k], w)
        for i in np.where(np.abs(grad) > tol)[0]:
            coef_shared[i] -= learning_rate * grad[i]

def async_ML(args):    
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
    global model_module
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    model_module.batch_size = batch_size
    model_module.njobs = njobs
    model_module.nsamples_per_job = nsamples_per_job
    
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
        
        model_module.batch_size = batch_size
        model_module.njobs = njobs
        model_module.nsamples_per_job = nsamples_per_job
    
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
    
def eval_njobs_nsamples_tradeoff(args):
    """
    evaluation mode.
    explores the tradeoff between njobs and nsamples in async training
    when total training data is fixed
    """
    global model_module
    if model_module is None:
        spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        model_module.batch_size = batch_size
    
    data, gt = model_module.get_data(total=args.total_training_data)
    
    eval_schedule = open(args.eval_schedule).read().split('\n')
    # first line of file gives schedule of njobs
    njobs_schedule = [int(i) for i in eval_schedule[0].split(' ')]
    
    init_weights = model_module.init()
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)

    p = Pool(nthreads) 
    
    
    
    all_times = []
    
    for njobs in njobs_schedule:
        # otherwise nsamples_per_job is not an integer
        assert args.total_training_data % njobs == 0
        nsamples_per_job = args.total_training_data // njobs
        coef_shared[:] = init_weights[:]
        
        if njobs == 1:
            # serial experiment
            T0 = time.time()
            for i in range(len(data)):
                hogwild_train_wrapper(data[i])
            T1 = time.time()
        else:
            reshaped_data = []
            for n in range(njobs):
                reshaped_data.append(data[0][n*nsamples_per_job:n*nsamples_per_job+nsamples_per_job])

            T0 = time.time()
            p.map(hogwild_train_wrapper, reshaped_data)
            T1 = time.time()
        
        model_module.finish(w, gt)
        current_time = T1 - T0
        all_times.append(current_time)
        
        print('eval with njobs ', njobs, 'finished in ', current_time, 's')
        
    out_str = 'model: %s\ntotal samples: %d\nnjobs schedule: %s\neval_times: %s\n' % (args.model_file, args.total_training_data, str(njobs_schedule), str(all_times))
    open(args.output_file, 'w').write(out_str)


def main():
    parser = argparse_util.ArgumentParser(description='asyn_ML')
    parser.add_argument('--model_file', dest='model_file', default='sanity_test.py', help='py file that contains model-specific methods, must include init(), compute_gradient(), get_data(), finish()')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='set training learning rate')
    parser.add_argument('--tol', dest='tol', type=float, default=DEFAULT_TOL, help='set the threshold when to update the sparse weight entries')
    parser.add_argument('--nthreads', dest='nthreads', type=int, default=DEFAULT_NTHREADS, help='set number of thread in async training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='set batch size per training step')
    parser.add_argument('--njobs', dest='njobs', type=int, default=DEFAULT_NJOBS, help='set number of jobs launched on different threads')
    parser.add_argument('--nsamples_per_job', dest='nsamples_per_job', type=int, default=DEFAULT_NSAMPLES_PER_JOB, help='set number of training examples per job')
    parser.add_argument('--mode', dest='mode', default='single', help='which mode to run experiments')
    parser.add_argument('--total_training_data', dest='total_training_data', type=int, default=10000, help='used in evaluation modes')
    parser.add_argument('--eval_schedule', dest='eval_schedule', default='eval.txt', help='file that provides evaluation schedule')
    parser.add_argument('--output_file', dest='output_file', default='out.txt', help='file to write evaluation result in eval mode')
    args = parser.parse_args()
    
    global learning_rate, tol, nthreads, batch_size, njobs, nsamples_per_job
    learning_rate = args.learning_rate
    tol = args.tol
    nthreads = args.nthreads
    batch_size = args.batch_size
    njobs = args.njobs
    nsamples_per_job = args.nsamples_per_job
    
    print(tol, learning_rate)
    
    if args.mode == 'single':
        # single mode, run experiments once in async setup, once in serial setup
        async_ML(args)
        serial_ML(args)
    elif args.mode == 'eval_njobs_nsamples_tradeoff':
        # evaluation mode, evaluates the tradeoff made between njobs and nsamples_per_job
        eval_njobs_nsamples_tradeoff(args)
    else:
        raise "mode not supported"
    
if __name__ == '__main__':
    main()