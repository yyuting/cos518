import argparse_util
import importlib
import importlib.util
import os
import time
import sys

# Yuting: shared memory modified from
# https://srome.github.io/Async-SGD-in-Python-Implementing-Hogwild!/

from multiprocessing.sharedctypes import Array
import multiprocessing
from ctypes import c_double
import numpy
import numpy as np
from multiprocessing import Pool, Queue

coef_shared = None
w = None
model_module = None

data_shared = None
data_val = None
data_validate = None
data_test = None
gt = None


DEFAULT_NTHREADS = 8
nthreads = DEFAULT_NTHREADS

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TOL = 1e-2

learning_rate = DEFAULT_LEARNING_RATE
tol = DEFAULT_TOL

DEFAULT_BATCH_SIZE = 1
batch_size = DEFAULT_BATCH_SIZE

DEFAULT_BETA = 0.9

DEFAULT_SPARSITY = 0.1

DEFAULT_REGULARIZATION = -1

def hogwild_shared_train_wrapper(data):
    model_module.shared_train_hogwild(data, w, coef_shared, data_val)
    
def hogwild_shared_train_wrapper_with_queue(q):
    while True:
        val = q.get()
        if val is None:
            break
        else:
            model_module.shared_train_hogwild([val], w, coef_shared, data_val)
            
def RR_shared_train_wrapper(data):
    model_module.shared_train_wrapper('RR', lock)(data, w, coef_shared, data_val)
    #model_module.shared_train_RR(data, w, coef_shared, data_val)

def get_grad_wrapper(data):
    return model_module.get_grad(data, w, coef_shared, data_val)

def async_ML_shared_data(args, mode='per_epoch'):
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(args.model_file))
    global model_module
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    model_module.learning_rate = learning_rate
    model_module.sparse_d = args.dataset_sparsity
    
    init_weights = model_module.init()
    
    global coef_shared, w
    if mode == 'RR':
        do_lock = True
    else:
        do_lock = False
        
    coef_shared = Array(c_double, init_weights.flat, lock=do_lock)
    if do_lock:
        w = np.frombuffer(coef_shared.get_obj())
    else:
        w = np.frombuffer(coef_shared)
    w = w.reshape(init_weights.shape)
        
    
    random_dataset = getattr(model_module, 'random_dataset', True)
    
    # assert data is a numpy array itself
    global data_shared, data_val, gt, data_validate, data_test
    if data_shared is None or data_val is None or gt is None or data_validate is None or data_test is None:
        data_all, gt = model_module.get_data_shared(args.total_training_data)
        if getattr(model_module, 'has_val', False) and getattr(model_module, 'has_test', False):
            data = data_all[0]
            data_validate = data_all[1]
            data_test = data_all[2]
        else:
            data = data_all
            data_validate = data_all
            data_test = data_all
        data_shared = Array(c_double, data.flat, lock=False)
        data_val = np.frombuffer(data_shared).reshape(data.shape) 
    args.total_training_data = data_val.shape[0]
    nsamples_per_job = args.total_training_data // nthreads
    print(args.total_training_data, nsamples_per_job, nthreads)
    assert nsamples_per_job * nthreads == args.total_training_data
    assert nsamples_per_job % args.batch_size == 0
        
        
    if args.regularization > 0:
        if hasattr(model_module, 'lambda_val'):
            model_module.lambda_val = args.regularization
    
    if nthreads == 1:
        mode = 'serial'
    
    indices = np.arange(args.total_training_data)
    if mode in ['per_epoch', 'all', 'sync']:
        p = Pool(nthreads)
    elif mode == 'RR':
        lock_obj = multiprocessing.Lock()
        def init_lock(l):
            global lock
            lock = l
        p = Pool(nthreads, initializer=init_lock, initargs=(lock_obj,))
    if mode == 'queue':
        q = Queue(maxsize=nthreads*2)
    st = 0
    
    best_err = 1e8
    best_epoch = -1
    best_model = np.empty(init_weights.shape)
    
    if mode == 'per_epoch':
        for e in range(args.epochs):
            T0 = time.time()
            np.random.shuffle(indices)

            jobs_idx = []
            for i in range(nthreads):
                jobs_idx.append(indices[i * nsamples_per_job : (i + 1) * nsamples_per_job].reshape((-1, args.batch_size)))

            p.map(hogwild_shared_train_wrapper, jobs_idx)

            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            T1 = time.time()
            st += T1 - T0
            err = model_module.finish(w, data_validate)
            print('epoch', e, 'error', err)
            if best_err > err:
                best_err = err
                best_epoch = e
                best_model[:] = w[:]
            
    elif mode == 'sync':
        # hack for CIFAR dataset, using Darby's code, but passing hyperparameters from the API
        if args.model_file == 'test_CIFAR.py':
            sys.path += ['darby_518']
            from SVM import SVM
            svm=SVM()
            
            sync_batch_size = args.batch_size * args.nthreads
            sync_niters = args.epochs * args.total_training_data // sync_batch_size
            
            assert (args.epochs * args.total_training_data) % sync_batch_size == 0
            
            T0 = time.time()
            history_loss = svm.train(data_val[:, :-1], data_val[:, -1].astype('i'), reg=args.regularization, learning_rate=args.learning_rate, num_iters=sync_niters, batch_size=sync_batch_size, verbose=True, nthreads=args.nthreads)
            T1 = time.time()
            
            y_pre=svm.predict(data_test[:, :-1])
            acc=np.mean(y_pre==data_test[:, -1].astype('i'))
            
            st += T1 - T0
                
            print("learning_rate=%e,regularization_strength=%e,val_accury=%f"%(args.learning_rate,args.regularization,acc))
        
        else:
            for e in range(args.epochs):
                T0 = time.time()
                np.random.shuffle(indices)

                jobs_idx = []
                for i in range(nthreads):
                    jobs_idx.append(indices[i * nsamples_per_job : (i + 1) * nsamples_per_job].reshape((-1, args.batch_size)))

                grads = p.map(get_grad_wrapper, jobs_idx)
                grad = np.sum(grads, axis=0)
                coef_shared[:] -= learning_rate * grad

                model_module.learning_rate *= args.beta
                model_module.print_learning_rate()
                T1 = time.time()
                st += T1 - T0
                print('epoch', e)
                err = model_module.finish(w, data_validate)
                if best_err > err:
                    best_err = err
                    best_epoch = e
                    best_model[:] = w[:]
    elif mode == 'RR':
        for e in range(args.epochs):
            T0 = time.time()
            np.random.shuffle(indices)

            jobs_idx = []
            for i in range(nthreads):
                jobs_idx.append(indices[i * nsamples_per_job : (i + 1) * nsamples_per_job].reshape((-1, args.batch_size)))

            p.map(RR_shared_train_wrapper, jobs_idx)

            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            T1 = time.time()
            st += T1 - T0
            print('epoch', e)
            err = model_module.finish(w, data_validate)
            if best_err > err:
                best_err = err
                best_epoch = e
                best_model[:] = w[:]
    elif mode == 'all':
        T0 = time.time()
        jobs_idx = [np.arange(0)] * nthreads
        for e in range(args.epochs):
            np.random.shuffle(indices)
            for i in range(nthreads):
                jobs_idx[i] = np.concatenate((jobs_idx[i], indices[i * nsamples_per_job : (i + 1) * nsamples_per_job].reshape((-1, args.batch_size))))
        p.map(hogwild_shared_train_wrapper, jobs_idx)
        T1 = time.time()
        st += T1 - T0
        model_module.finish(w, data_validate)
    elif mode == 'queue':
        p = Pool(nthreads, initializer=hogwild_shared_train_wrapper_with_queue, initargs=(q,))
        for e in range(args.epochs):
            T0 = time.time()
            np.random.shuffle(indices)
            for ind in indices:
                q.put(ind)
            print('epoch', e)
            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            T1 = time.time()
            st += T1 - T0
            err = model_module.finish(w, data_validate)
            if best_err > err:
                best_err = err
                best_epoch = e
                best_model[:] = w[:]

        for _ in range(nthreads):  # tell workers we're done
            q.put(None)

    elif mode == 'serial':
        for e in range(args.epochs):
            T0 = time.time()
            np.random.shuffle(indices)

            jobs_idx = []
            for i in range(nthreads):
                jobs_idx.append(indices[i * nsamples_per_job: (i + 1) * nsamples_per_job].reshape((-1, args.batch_size)))

            for i in range(len(jobs_idx)):
                hogwild_shared_train_wrapper(jobs_idx[i])

            
            model_module.learning_rate *= args.beta
            model_module.print_learning_rate()
            T1 = time.time()
            st += T1 - T0
            print('epoch', e)
            err = model_module.finish(w, data_validate)
            if best_err > err:
                best_err = err
                best_epoch = e
                best_model[:] = w[:]
    else:
        raise 'async shared data mode not allowed'
    
    if mode in ['per_epoch', 'all', 'RR', 'queue']:
        p.close()
        p.join()
    
    print('mode', mode)
    print('shared async job finished in', st, 's')
    err = model_module.finish(w, data_test, mode='test')
    
    best_test_err = model_module.finish(best_model, data_test, mode='test')
    print('best model at epoch %d' % best_epoch, 'validation score', best_err, 'test score', best_test_err)
    
    return st, best_test_err
    
    
def eval_nthreads_tradeoff(args):
    """
    evaluation mode.
    explores the tradeoff between njobs = nthreads in async training
    when total training data is fixed
    """
    eval_schedule = open(args.eval_schedule).read().split('\n')
    # first line of file gives schedule of njobs
    nthreads_schedule = [int(i) for i in eval_schedule[0].split(' ')]
    
    global nthreads, nsamples_per_job
    
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
    parser.add_argument('--mode', dest='mode', default='per_epoch,serial', help='which mode to run experiments')
    parser.add_argument('--total_training_data', dest='total_training_data', type=int, default=10000, help='total number of training examples, used to generate training data')
    parser.add_argument('--eval_schedule', dest='eval_schedule', default='eval.txt', help='file that provides evaluation schedule')
    parser.add_argument('--output_file', dest='output_file', default='out.txt', help='file to write evaluation result in eval mode')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--beta', dest='beta', type=float, default=DEFAULT_BETA, help='beta used to decay learning rate per epoch')
    parser.add_argument('--dataset_sparsity', dest='dataset_sparsity', type=float, default=DEFAULT_SPARSITY, help='if creating random dataset, set the dataset sparsity')
    parser.add_argument('--regularization', dest='regularization', type=float, default=DEFAULT_REGULARIZATION, help='if >0, use this to schedule regularization in SVM')
    
    args = parser.parse_args()
    
    global learning_rate, tol, nthreads, batch_size
    learning_rate = args.learning_rate
    tol = args.tol
    nthreads = args.nthreads
    batch_size = args.batch_size

    print(tol, learning_rate)
    
    modes = args.mode.split(',')
    for mode in modes:
        if mode == 'eval_nthreads_tradeoff':
            eval_nthreads_tradeoff(args)
        else:
            async_ML_shared_data(args, mode)
    
if __name__ == '__main__':
    main()
