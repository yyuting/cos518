# cos518
COS 518 course project

Example commands:

synthetic SVM dataset using HogWild!

python main.py --model_file test_sparse_SVM.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode per_epoch --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4

synthetic SVM dataset using synchronize SGD

python main.py --model_file test_sparse_SVM.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode sync --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4

synthetic SVM datset using serial SGD

python main.py --model_file test_sparse_SVM.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode serial --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4

CIFAR datset using HogWild!

python main.py --model_file test_CIFAR.py --learning_rate 0.000002 --epochs 5 --batch_size 125 --mode per_epoch --nthreads 4

CIFAR dataset using synchronize SGD

!!! This is not in the report, but we actually implemented this and it's working!

python main.py --model_file test_CIFAR.py --learning_rate 0.000002 --epochs 5 --batch_size 125 --mode sync --nthreads 4

CIFAR dataset using serial SGD

python main.py --model_file test_CIFAR.py --learning_rate 0.000002 --epochs 5 --batch_size 125 --mode serial --nthreads 4

synthetic LR datset using HogWild!

python main.py --model_file test_LR.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode per_epoch --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4

synthetic LR datset using synchronize SGD

python main.py --model_file test_LR.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode sync --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4

synthetic LR dataset using serial SGD

python main.py --model_file test_LR.py --learning_rate 0.000001 --epochs 5 --batch_size 2 --mode serial --total_training_data 100000 --dataset_sparsity 0.1 --nthreads 4