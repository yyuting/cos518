import multiprocessing as mp
import time
from time import sleep

NCORE = 4

count = 0

def process(q):
    from time import sleep
    while True:
        stuff = q.get()
        if stuff is None:
            break
        sleep(1)

if __name__ == '__main__':
    
    T0 = time.time()
    q = mp.Queue(maxsize=NCORE*2)
    pool = mp.Pool(NCORE, initializer=process, initargs=(q,))
    for stuff in range(20):
        q.put(stuff)  # blocks until q below its max size
    for _ in range(NCORE):  # tell workers we're done
        q.put(None)
    pool.close()
    pool.join()
    T1 = time.time()
    
    print('parallel', T1 - T0)
    
    T2 = time.time()
    q = mp.Queue(maxsize=NCORE*2)
    for stuff in range(20):
        q.put(stuff)  # blocks until q below its max size
        stuff = q.get()
        if stuff is not None:
            sleep(1)
    for _ in range(NCORE):  # tell workers we're done
        q.put(None)
        stuff = q.get()
    T3 = time.time()
    
    print('serial', T3 - T2)