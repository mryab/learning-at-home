"""
Task pool is responsible for receiving tasks and grouping them together for processing (but not processing itself)
"""
import ctypes
import multiprocessing as mp
import os
import threading
import uuid
from collections import namedtuple
from concurrent.futures import Future
from queue import Empty
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

from ..utils import SharedFuture, SharedArrays, BatchTensorProto, SharedArray, check_numpy, time

Task = namedtuple("Task", ("future", "args"))


class TaskPoolBase(mp.Process):
    """ A pool that accepts tasks and forms batches for parallel processing, interacts with TesseractRuntime """

    def __init__(self, process_func: callable):
        super().__init__()
        self.process_func = process_func
        self._priority = mp.Value(ctypes.c_double, 1.0)  # higher priority = the more urgent to process this pool

    def run(self):
        raise NotImplementedError()

    def submit_task(self, *args: torch.Tensor) -> Future:
        raise NotImplementedError()

    def form_batch(self, *args, **kwargs) -> List[Task]:
        raise NotImplementedError()

    def iterate_minibatches(self, *args, **kwargs):
        while True:
            yield self.form_batch(*args, **kwargs)

    @property
    def priority(self):
        return self._priority.value

    @priority.setter
    def priority(self, value):
        self._priority.value = float(value)

    @property
    def empty(self):
        raise NotImplementedError()


class TaskPool(TaskPoolBase):

    def __init__(self, process_func: callable,
                 inputs_schema: Tuple[BatchTensorProto, ...], outputs_schema: Tuple[BatchTensorProto, ...],
                 max_batch_size: int, min_batch_size=1, timeout=None, pool_size=None, prefetch_batches=1, uid=None,
                 shm_manager=None, array_headers=None, start=False):
        """
        Naive implementation of task pool that forms batch from earliest submitted tasks
        :param process_func: function to be applied to every formed batch; called by TesseractRuntime
            Note: process_func should accept only *args Tensors and return a list of output Tensors
        :param inputs_schema: description of arguments to process_func, list of BatchTensorProto, positional only
        :param outputs_schema: description of outputs from process_func, list of BatchTensorProto, must return a list
        :param max_batch_size: process at most this many inputs in a batch (task contains have one or several inputs)
        :param min_batch_size: process at least this many inputs in a batch, otherwise wait for more
        :param timeout: wait for a subsequent task for at most this many seconds
        :param pool_size: store at most this many unprocessed tasks in a queue
        :param prefetch_batches: prepare up to this many *batches* in background for faster off-loading to runtime
        :param uid: pool identifier used for shared array allocation
        :param start: if True, start automatically at the end of __init__
        """

        super().__init__(process_func)
        self.min_batch_size, self.max_batch_size, self.timeout = min_batch_size, max_batch_size, timeout
        self.inputs_schema, self.outputs_schema = list(inputs_schema), list(outputs_schema)
        self.uid = uid or uuid.uuid4()
        self.prefetch_batches = prefetch_batches

        # interaction with ConnectionHandlers
        self.tasks = mp.Queue(maxsize=pool_size or 0)
        self.undispatched_task_timestamps = mp.SimpleQueue()

        # interaction with TesseractRuntime
        self.shared_arrays = SharedArrays(array_headers=array_headers, shm_manager=shm_manager)

        self.batch_receiver, self.batch_sender = mp.Pipe(duplex=False)  # send/recv array names that contain batch inputs
        self.batch_received = mp.Event()  # runtime can notify pool that it can send next batch

        self.outputs_receiver, self.outputs_sender = mp.Pipe(duplex=False)  # send/recv array names that contain outputs
        self.outputs_received = mp.Event()  # pool can notify runtime that it can send next outputs

        if start:
            self.start()

    def submit_task(self, *args: torch.Tensor) -> Future:
        future1, future2 = SharedFuture.make_pair()
        self.tasks.put(Task(future1, args))
        self.undispatched_task_timestamps.put(time.time())
        return future2

    def form_batch(self) -> List[Task]:
        batch_tasks = []
        total_size = 0

        while total_size < self.max_batch_size:
            if total_size >= self.min_batch_size and self.tasks.empty():
                break  # timeout reached, returning incomplete batch

            try:
                task = self.tasks.get(timeout=self.timeout)
            except Empty:
                exc = TimeoutError(f"Timeout reached but batch doesn't contain >={self.min_batch_size} elements yet.")
                for task in batch_tasks:
                    task.future.set_exception(exc)
                raise exc

            if task.future.set_running_or_notify_cancel():
                batch_tasks.append(task)
                total_size += self.get_task_size(task)

        return batch_tasks

    def run(self, *args, status_timeout=0.1, **kwargs):
        print(f'Starting pool, {os.getpid()=}')
        pending_batches = {}  # Dict[batch uuid, List[SharedFuture]] for each batch currently in runtime
        self.batch_received.set(), self.outputs_received.set()  # initial state: no batches/outputs pending
        output_thread = threading.Thread(target=self._pool_output_loop, args=[pending_batches])
        try:
            output_thread.start()
            self._pool_input_loop(pending_batches, *args, **kwargs)
        except KeyboardInterrupt:
            print('Pool caught KeyboardInterrupt, exiting')
        finally:
            output_thread.join()
            self.shared_arrays.shm_manager.shutdown()

    def _pool_input_loop(self, pending_batches: Dict[Any, List[Task]], *args, **kwargs):
        """ Thread method that continually forms batches and sends them to runtime """
        prev_num_tasks = 0  # number of tasks currently in shared buffer
        batch_index = max(pending_batches.keys(), default=0)
        batch_iterator = self.iterate_minibatches(*args, **kwargs)

        while True:
            self.batch_received.wait()  # wait for runtime to receive (copy) previous batch

            # SIDE-EFFECT - compute pool priority from timestamp of earliest undispatched task
            # assumes that tasks are processed in the same order as they are created
            for skip_i in range(prev_num_tasks):
                finished_task_timestamp = self.undispatched_task_timestamps.get()  # earlier timestamp = higher priority
                if skip_i == prev_num_tasks - 1:
                    self.priority = finished_task_timestamp

            batch_tasks = next(batch_iterator)
            # save batch futures, _output_loop will deliver on them later
            pending_batches[batch_index] = batch_tasks

            # find or create shared arrays for current batch size
            batch_size = sum(map(self.get_task_size, batch_tasks))
            shared_keys, shared_buffers = zip(
                *self.get_or_create_buffers(batch_size, self.inputs_schema, name='inputs'))

            self.batch_received.clear()  # sending next batch...
            for i, buffer in enumerate(shared_buffers):
                np.concatenate([task.args[i] for task in batch_tasks], out=buffer)  # assemble batch from tasks

            self.batch_sender.send((batch_index, shared_keys))  # send input keys, trigger runtime to receive batch
            batch_index += 1
            prev_num_tasks = len(batch_tasks)

    def _pool_output_loop(self, pending_batches: Dict[Any, List[Task]]):
        """ Thread method that continually receives results from runtime and dispatches them to task Futures """

        while True:
            try:
                batch_index, output_keys = self.outputs_receiver.recv()
                batch_outputs = [self.shared_arrays[key].copy() for key in output_keys]
                self.outputs_received.set()  # runtime can now send next output

                # split batch into partitions for individual tasks
                batch_tasks = pending_batches.pop(batch_index)
                task_sizes = [self.get_task_size(task) for task in batch_tasks]
                task_sections = np.cumsum(task_sizes)[:-1]  # index in batch where task begins, for all tasks expert first
                outputs_per_task = zip(*(np.split(array, task_sections) for array in batch_outputs))

                # dispatch results to futures
                for task, task_outputs in zip(batch_tasks, outputs_per_task):
                    task.future.set_result(tuple(
                        proto.convert_array_to_tensor(array) for proto, array in zip(self.outputs_schema, task_outputs)
                    ))
            except KeyboardInterrupt:
                break

    @property
    def empty(self):
        return not self.batch_receiver.poll()

    def load_batch_to_runtime(self, timeout=None, device=None) -> Tuple[Any, List[torch.Tensor]]:
        """ receive next batch of numpy arrays """
        if not self.batch_receiver.poll(timeout):
            raise TimeoutError()

        batch_index, input_keys = self.batch_receiver.recv()
        batch_inputs = [self.shared_arrays[key].copy() for key in input_keys]
        self.batch_received.set()  # pool can now prepare next batch
        batch_inputs = [tensor_proto.convert_array_to_tensor(array).to(device, non_blocking=True)
                        for array, tensor_proto in zip(batch_inputs, self.inputs_schema)]
        return batch_index, batch_inputs

    def send_outputs_from_runtime(self, batch_index: int, batch_outputs: List[np.ndarray]):
        """ send results for a processed batch, previously loaded through receive_batch """
        batch_size = len(batch_outputs[0])
        shared_keys, shared_buffers = zip(*self.get_or_create_buffers(batch_size, self.outputs_schema, name='outputs'))
        self.outputs_received.wait(), self.outputs_received.clear()  # wait for pool to receive (copy) previous outputs

        for output, buffer in zip(batch_outputs, shared_buffers):
            np.copyto(dst=buffer, src=output)

        self.outputs_sender.send((batch_index, shared_keys))

    def get_task_size(self, task: Task) -> int:
        """ compute task processing complexity (used for batching); defaults to batch size """
        return len(task.args[0]) if task.args else 1

    def get_or_create_buffers(self, batch_size, schema: List[BatchTensorProto], name: str = '') -> List[SharedArray]:
        """ get or create a shared arrays for inputs and outputs with a given batch dimension """

        for i, proto in enumerate(schema):
            key = f"pool_{self.uid}__batchsize_{batch_size}__{name}_{i}"
            if key in self.shared_arrays:
                arr = self.shared_arrays[key]
            else:
                self.shared_arrays[key] = arr = SharedArray.from_array(check_numpy(proto.make_empty(batch_size)))
            yield key, arr
