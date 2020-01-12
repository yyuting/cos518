import sys
import os
sys.path += ['darby_518']
from SVM_CIFAR10 import data_processing
import numpy
import numpy as np

has_val = True
has_test = True

random_dataset = False

ndims = 3073
nclasses = 10

lambda_val = 0.001

debug_mode = False


def init():
    """
    Return an initialization of weights
    """
    return 0.005*np.random.randn(ndims, nclasses)

def print_learning_rate():
    print('learning rate now', learning_rate)
    
def shared_train_wrapper(alg, lock=None):
    assert alg in ['hogwild', 'RR']
    def func(idx, w, coef_shared, data_val):
        
        for k in idx:
            if isinstance(k, (int, np.int64)):
                k = [k]
            x = data_val[k, :-1]
            y = data_val[k, -1].astype('i')

            reg = lambda_val

            loss=0.0

            num_train=x.shape[0]
            scores=x.dot(w)
            margin=scores-scores[np.arange(num_train),y].reshape(num_train,1)+1
            margin[np.arange(num_train),y]=0.0

            margin=(margin>0)*1
            row_sum=np.sum(margin,axis=1)
            margin[np.arange(num_train),y]=-row_sum
            
            
            
            if debug_mode:
                dW = (x.T.dot(margin) / num_train).reshape(-1)
                nonzero_ind = np.nonzero(dW)[0]
                du = nonzero_ind.shape[0]
            else:
                dW=x.T.dot(margin)/num_train+reg*w
            
            

            if alg == 'RR':
                lock.acquire()
             
            if debug_mode:
                for i in nonzero_ind:
                    current_grad = 2 * reg * coef_shared[i] / du + dW[i]
                    coef_shared[i] -= learning_rate * current_grad
            else:
                coef_shared[:] -= learning_rate * dW.reshape(-1)
            if alg == 'RR':
                lock.release()
        
    return func

shared_train_hogwild = shared_train_wrapper('hogwild')

def get_data_shared(total):
    
    cwd = os.getcwd()
    
    os.chdir('darby_518')
    
    x_train,y_train,x_val,y_val,x_test,y_test,x_check,y_check= data_processing()
    os.chdir(cwd)
    
    data_train = np.concatenate((x_train, np.expand_dims(y_train, 1)), 1)
    data_validate = np.concatenate((x_val, np.expand_dims(y_val, 1)), 1)
    data_test = np.concatenate((x_test, np.expand_dims(y_test, 1)), 1)
    
    return [data_train, data_validate, data_test], None

def finish(w, data):
    """
    process trained model
    """
    x = data[:, :-1]
    y = data[:, -1].astype('i')
    num_train=x.shape[0]
    scores=x.dot(w)
    margin=scores-scores[np.arange(num_train),y].reshape(num_train,1)+1
    margin[np.arange(num_train),y]=0.0
    margin=(margin>0)*margin
    err = margin.sum()/num_train
    reg = 0.5*lambda_val*np.sum(w * w)
    print("training error, ", err, err + reg)
    return err