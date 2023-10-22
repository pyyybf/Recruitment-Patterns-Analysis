from dask.distributed import Scheduler
from contextlib import closing

def start_scheduler():
    with closing(Scheduler()) as s:
        s.listen('tcp://:8786')  # 默认端口是8786
        s.start()

if __name__ == '__main__':
    start_scheduler()
