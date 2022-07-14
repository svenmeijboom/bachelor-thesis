from multiprocessing import Queue
from typing import Optional

from tqdm import tqdm


def status_bar(stat_queue: Queue, total: int, desc: Optional[str] = None):
    encountered = 0

    with tqdm(desc=desc, total=total) as pbar:
        while encountered < total:
            message = stat_queue.get()
            pbar.update()
            encountered += 1

            if message != 'success':
                pbar.write(message)
