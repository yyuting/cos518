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
from multiprocessing import Pool, Queue

coef_shared = None
w = None
model_module = None

data_shared = None
data_val = None
gt = None


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

DEFAULT_BETA = 0.9

def hogwild_train_wrapper(data):
    model_module.train_hogwild(data, w, coef_shared)
    return
    if False:
        for k in range(len(data)):
            grad = model_module.compute_gradient(data[k], w)
            #for i, val in grad:
            #    coef_shared[i] -= learning_rate * val
            for i in np.where(np.abs(grad) > tol)[0]:
                coef_shared[i] -= learning_rate * grad[i]
                
    
def hogwild_shared_train_wrapper(data):
    model_module.shared_train_hogwild(data, w, coef_shared, data_val)
    
def hogwild_shared_train_wrapper_with_queue(q):
    while True:
        val = q.get()
        if val is None:
            break
        else:
            model_module.shared_train_hogwild([val], w, coef_shared, data_val)
                
def async_ML_shared_data(args, mode='per_epoch'):
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
    global model_module
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    model_module.learning_rate = learning_rate
    
    init_weights = model_module.init()
    
    global coef_shared, w
    coef_shared = Array(c_double, init_weights.flat, lock=False)
    w = np.frombuffer(coef_shared)
    
    # assert data is a numpy array itself
    global data_shared, data_val, gt
    if data_shared is None or data_val is None or gt is None:
        data, gt = model_module.get_data_shared(args.total_training_data)
        data_shared = Array(c_double, data.flat, lock=False)
        data_val = np.frombuffer(data_shared).reshape((args.total_training_data, -1))
        
    nsamples_per_job = args.total_training_data // nthreads
    assert nsamples_per_job * nthreads == args.total_training_data
    
    if nthreads == 1:
        mode = 'serial'
    
    indices = np.arange(args.total_training_data)
    if mode in ['per_epoch', 'all']:
        p = Pool(nthreads)
    if mode == 'queue':
        q = Queue(maxsize=nthreads*2)
    T0 = time.time()
    
    if mode == 'per_epoch':
        for e in range(args.epochs):
            np.random.shuffle(indices)

            jobs_idx = []
            for i in range(nthreads):
                jobs_idx.append(indices[i * nsamples_per_job : (i + 1) * nsamples_per_job])

            p.map(hogwild_shared_train_wrapper, jobs_idx)

            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            print('epoch', e)
            model_module.finish(w, gt)
    elif mode == 'all':
        jobs_idx = [np.arange(0)] * nthreads
        for e in range(args.epochs):
            np.random.shuffle(indices)
            for i in range(nthreads):
                jobs_idx[i] = np.concatenate((jobs_idx[i], indices[i * nsamples_per_job : (i + 1) * nsamples_per_job]))
        p.map(hogwild_shared_train_wrapper, jobs_idx)
        model_module.finish(w, gt)
    elif mode == 'queue':
        p = Pool(nthreads, initializer=hogwild_shared_train_wrapper_with_queue, initargs=(q,))
        for e in range(args.epochs):
            np.random.shuffle(indices)
            for ind in indices:
                q.put(ind)
            print('epoch', e)
            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            model_module.finish(w, gt)

        for _ in range(nthreads):  # tell workers we're done
            q.put(None)

        p.close()
        p.join()
    elif mode == 'serial':
        for e in range(args.epochs):
            np.random.shuffle(indices)

            jobs_idx = []
            for i in range(nthreads):
                jobs_idx.append(indices[i * nsamples_per_job: (i + 1) * nsamples_per_job])

            for i in range(len(jobs_idx)):
                hogwild_shared_train_wrapper(jobs_idx[i])

            print('epoch', e)
            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            model_module.finish(w, gt)
    else:
        raise 'async shared data mode not allowed'
    
    T1 = time.time()
    if mode in ['per_epoch', 'all']:
        p.close()
    
    print('mode', mode)
    print('shared async job finished in', T1 - T0, 's')
    err = model_module.finish(w, gt)
    return T1 - T0, err
    
    
def eval_nthreads_tradeoff(args):
    """
    evaluation mode.
    explores the tradeoff between njobs = nthreads in async training
    when total training data is fixed
    """
    eval_schedule = open(args.eval_schedule).read().split('\n')
    # first line of file gives schedule of njobs
    nthreads_schedule = [int(i) for i in eval_schedule[0].split(' ')]
    
    global nthreads, njobs, nsamples_per_job
    
    all_times = np.empty(len(nthreads_schedule))
    all_err = np.empty(len(nthreads_schedule))
    
    for i in range(len(nthreads_schedule)):
        val = nthreads_schedule[i]
        nthreads = val
        assert args.total_training_data % nthreads == 0
        tval, err = async_ML_shared_data(args, mode='per_epoch')
        all_times[i] = tval
        all_err[i] = err
        print('eval with nthreads ', nthreads, 'finished in ', tval, 's')
        
    out_str = 'model: %s\ntotal samples: %d\nnthreads schedule: %s\neval_times: %s\neval_err: %s\n' % (args.model_file, args.total_training_data, str(nthreads_schedule), str(all_times), str(all_err))
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
    parser.add_argument('--mode', dest='mode', default='per_epoch,serial', help='which mode to run experiments')
    parser.add_argument('--total_training_data', dest='total_training_data', type=int, default=10000, help='total number of training examples, used to generate training data')
    parser.add_argument('--eval_schedule', dest='eval_schedule', default='eval.txt', help='file that provides evaluation schedule')
    parser.add_argument('--output_file', dest='output_file', default='out.txt', help='file to write evaluation result in eval mode')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--beta', dest='beta', type=float, default=DEFAULT_BETA, help='beta used to decay learning rate per epoch')
    args = parser.parse_args()
    
    global learning_rate, tol, nthreads, batch_size, njobs, nsamples_per_job
    learning_rate = args.learning_rate
    tol = args.tol
    nthreads = args.nthreads
    batch_size = args.batch_size
    njobs = args.njobs
    nsamples_per_job = args.nsamples_per_job
    
    print(tol, learning_rate)
    
    modes = args.mode.split(',')
    for mode in modes:
        if mode == 'eval_nthreads_tradeoff':
            eval_nthreads_tradeoff(args)
        else:
            async_ML_shared_data(args, mode)
    
if __name__ == '__main__':
    main()