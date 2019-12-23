# COS 518 Project plan

#Goal 

Implement Synchronous and Asyncrhonous SGD and compare their performance for 2-3 different tasks (regression, SVM, and perhaps CIFAR?)

#Background

Synchronous Distributed SGD

In the synchronous setting, all replicas average all of their gradients at every timestep (minibatch). Doing so, we're effectively multiplying the batch size M by the number of replicas R, so that our overall minibatch size is B = R.M. This has several advantages.

1. The computation is completely deterministic.
2. We can work with fairly large models and large batch sizes even on memory-limited GPUs.
3. It's very simple to implement, and easy to debug and analyze.

However, synchronous SGD can be needlessly inefficient as all workers need to finish the gradient update for the algorithm to move to the next iteration. Therefore, as the number of workers increases, synchronous SGD becomes slower. 

Asynchronous Distributed SGD

One popular asynchronous SGD algorithm is HogWild! In Hogwild!, workers can overwrite each other by writing at the same time and compute gradients using a stale version of the “current solution.” The advantage of adding asynchrony to our training is that replicas can work at their own pace, without waiting for others to finish computing their gradients. Although this seems almost too good to be true -- it actually works quite well in practice. 

#Potential Experiment plan

1. Compare accuracy of synchronous vs. asynchronous SGD (HogWild!) for the different algorithms as a function of number of workers (i.e. 1, 3, 5, 10, 15 etc) -- there should be minimal difference b/w the two (but synchronous maybe better).

2. Compare execution time of synchronous vs. asynchronous SGD (HogWild!) for the different algorithms as a function of number of workers (i.e. 1, 3, 5, 10, 15 etc) -- synchronous SGD should worsen as the number of workers increase. 

