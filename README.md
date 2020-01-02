# cos518
COS 518 course project

Test command (simple counting, not ML):

python main.py --model_file sanity_test.py --learning_rate 1 --njobs 1000 --batch_size 200 --nsamples_per_job 1

Updated 12/23/2019:

simple linear regression with async training (Hogwild! algorithm)

python main.py --model_file test_LR.py

Updated 01/02/2019:

Script to evaluate async training with different job schedules

python main.py --model_file test_LR.py --nthreads 4 --eval_schedule eval_njobs_schedule.txt --mode eval_njobs_nsamples_tradeoff --total_training_data 8192 --output_file out_njobs_schedule_test_LR.txt