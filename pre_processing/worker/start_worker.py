from dask.distributed import Worker
from contextlib import closing

def start_worker(scheduler_address='tcp://localhost:8786'):
    with closing(Worker(scheduler_address)) as w:
        w.start()

if __name__ == '__main__':
    start_worker()
