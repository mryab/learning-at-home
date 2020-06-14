import os
import time
from argparse import ArgumentParser
from functools import partial
from itertools import chain, repeat

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from layers import name_to_block, name_to_input
from torch.distributed.rpc import rpc_sync
from tqdm import trange


class BlockWorker(nn.Module):
    def __init__(self, hid_dim, block_type):
        super().__init__()
        self.block = name_to_block[block_type](hid_dim).cuda()

    def forward(self, x):
        return self.block(x.cuda()).cpu()


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class ModelParallelRPC(nn.Module):
    def __init__(self, hid_dim, block_type, workers, layers_per_gpu):
        super().__init__()
        self.workers = list(chain.from_iterable(repeat(workers, layers_per_gpu)))
        self.layer_rrefs = [rpc.remote(worker, BlockWorker, args=(hid_dim, block_type)) for worker in self.workers]

    def forward(self, x):
        for layer_rref in self.layer_rrefs:
            x = _remote_method(BlockWorker.forward, layer_rref, x)
        return x


def measure_perf(model_class, batches_for_latency, batches_for_throughput, throughput_runs, input_factory,
                 batch_size, hid_dim,
                 **kwargs):
    m = model_class(hid_dim=hid_dim, **kwargs)
    time_per_batch = []
    z = input_factory(batch_size, hid_dim)
    with torch.no_grad():
        # throughput: examples/sec when results are asynchronous
        throughputs = []
        for run in trange(throughput_runs):
            start = time.time()
            for _ in range(batches_for_throughput):
                output = m(z)
            throughputs.append(batch_size * batches_for_throughput / (time.time() - start))
    print(throughputs)
    return np.mean(throughputs), np.std(throughputs, ddof=1)


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    measure_func = partial(measure_perf, batches_for_latency=args.batches_for_latency,
                           batches_for_throughput=args.batches_for_throughput,
                           throughput_runs=args.throughput_runs,
                           layers_per_gpu=args.layers_per_gpu,
                           hid_dim=args.hid_dim,
                           block_type=args.block_type, batch_size=args.batch_size,
                           input_factory=name_to_input[args.block_type],
                           workers=[f'worker{rank}' for rank in range(1, args.world_size)])

    avg_throughput, std_throughput = measure_func(ModelParallelRPC)
    print(f'ModelParallel:\t{avg_throughput:.2f}±{std_throughput:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hid-dim', type=int, default=1024)
    parser.add_argument('--batches-for-latency', type=int, default=10)
    parser.add_argument('--batches-for-throughput', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--throughput-runs', type=int, default=10)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--layers-per-gpu', type=int, default=56)
    parser.add_argument('--block-type', choices=name_to_block.keys(), required=True)
    args = parser.parse_args()
    rpc.init_rpc(f"worker{args.rank}", rank=args.rank, world_size=args.world_size)
    if args.rank == 0:
        main(args)
    rpc.shutdown()
