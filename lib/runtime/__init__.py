import multiprocessing as mp
from itertools import chain
from selectors import DefaultSelector, EVENT_READ
from typing import Dict, List

import torch
import tqdm
from prefetch_generator import BackgroundGenerator

from .expert_backend import ExpertBackend
from .task_pool import TaskPool, TaskPoolBase
from ..utils import check_numpy


class TesseractRuntime:
    def __init__(self, expert_backends: Dict[str, ExpertBackend], prefetch_batches=64, sender_threads: int = 1,
                 device: torch.device = None):
        """
        A group of processes that process tasks for multiple experts on a shared device
        :param expert_backends: a dict [expert uid -> ExpertBackend]
        :param prefetch_batches: generate up to this many batches in advance
        :param start: start process at the end of __init__
        """
        super().__init__()
        self.expert_backends = expert_backends
        self.pools = tuple(chain(*(expert.get_pools() for expert in expert_backends.values())))
        self.device, self.prefetch_batches, self.sender_threads = device, prefetch_batches, sender_threads

    def main(self):
        progress = tqdm.tqdm(bar_format='{desc}, {rate_fmt}')
        for pool in self.pools:
            if not pool.is_alive():
                pool.start()
        if self.device is not None:
            for expert_backend in self.expert_backends.values():
                expert_backend.to(self.device)

        with mp.pool.ThreadPool(self.sender_threads) as output_sender_pool, DefaultSelector() as sel:
            for pool in self.pools:
                sel.register(pool.batch_receiver, EVENT_READ, pool)
            try:
                for pool, batch_index, batch in BackgroundGenerator(
                        self.iterate_minibatches_from_pools(selector=sel), self.prefetch_batches):
                    outputs = pool.process_func(*batch)
                    progress.update(len(outputs[0]))
                    progress.desc = f'{pool.uid=} {len(outputs[0])=}'
                    output_sender_pool.apply_async(self.send_outputs_to_pool, args=(pool, batch_index, outputs))
            except KeyboardInterrupt:
                print('Runtime caught KeyboardInterrupt, exiting')
        for pool in self.pools:
            pool.join()

    def iterate_minibatches_from_pools(self, selector, timeout=None):
        """
        Chooses pool according to priority, then copies exposed batch and frees the buffer
        """
        while True:
            try:
                # wait until at least one batch_receiver becomes available
                ready_objects = selector.select()
                ready_pools = (key.data for (key, events) in ready_objects)
                pool = max(ready_pools, key=lambda pool: pool.priority)

                batch_index, batch_tensors = pool.load_batch_to_runtime(timeout, self.device)
                yield pool, batch_index, batch_tensors
            except KeyboardInterrupt:
                break

    def send_outputs_to_pool(self, pool: TaskPool, batch_index: int, outputs: List[torch.Tensor]):
        return pool.send_outputs_from_runtime(batch_index, [check_numpy(output) for output in outputs])
