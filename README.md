# cos518
COS 518 course project

Test command (simple counting, not ML):

python main.py --model_file sanity_test.py --learning_rate 1 --njobs 1000 --batch_size 200 --nsamples_per_job 1

Updated 12/23/2019:

simple linear regression with async training (Hogwild! algorithm)

~~python main.py --model_file test_LR.py

Updated 01/02/2020:

Script to evaluate async training with different job schedules

~~python main.py --model_file test_LR.py --nthreads 4 --eval_schedule eval_njobs_schedule.txt --mode eval_njobs_nsamples_tradeoff --total_training_data 8192 --output_file out_njobs_schedule_test_LR.txt

Updated 01/04/2020:

add sparse SVM

Refactor code, data is now in shared memory, only indices are parsed to each processes

can train multiple epochs with learning rate decay now

Run sparse SVM and linear regression:

python main.py --model_file test_sarse_SVM.py --nsamples_per_job 10000 --learning_rate 0.01 --epochs 20 --nthreads 4 --njobs 4 --beta 0.9

python main.py --model_file test_LR.py --nsamples_per_job 10000 --learning_rate 0.01 --epochs 10 --nthreads 4 --njobs 4 --beta 0.9

Benchmark sparse SVM and linear regression on different number of threads:

python main.py --model_file test_sarse_SVM.py --learning_rate 0.01 --epochs 10 --nthreads 4 --beta 0.9 --total_training_data 42000 --mode eval_nthreads_tradeoff --eval_schedule eval_njobs_schedule.txt --output_file out_njobs_schedule_test_SVM.txt

python main.py --model_file test_LR.py --learning_rate 0.01 --epochs 10 --nthreads 4 --beta 0.9 --total_training_data 42000 --mode eval_nthreads_tradeoff --eval_schedule eval_njobs_schedule.txt --output_file out_njobs_schedule_test_LR.txt

Updated 01/09/2020:

provide API to set random dataset sparsity:

python main.py --model_file test_sarse_SVM.py --nsamples_per_job 10000 --learning_rate 0.01 --epochs 20 --nthreads 4 --njobs 4 --beta 0.9 --dataset_sparsity 0.1

Updated 01/10/2020:

Add round-robin baseline, super slow, may need to find more efficient locking later.

python main.py --model_file test_sarse_SVM.py --nsamples_per_job 10000 --learning_rate 0.01 --epochs 20 --nthreads 2 --njobs 2 --beta 0.9 --dataset_sparsity 0.1 --mode RR

Updated 01/12/2020:

Add CIFAR SVM

python main.py --model_file test_CIFAR.py --learning_rate 0.01 --epochs 2 --beta 0.9 --nthreads 4 --njobs 4 --mode per_epoch --batch_size 125