import time
from argparse import ArgumentParser
from functools import partial
from itertools import chain, repeat

import numpy as np
import torch
import torch.nn as nn

from layers import name_to_block, name_to_input


class ModelParallelNetwork(nn.Module):
    def __init__(self, hid_dim, block_factory, gpus, layers_per_gpu):
        super().__init__()

        self.gpus = gpus
        self.blocks = nn.ModuleList(
            [nn.Sequential(*(torch.jit.script(block_factory(hid_dim)) for _ in range(layers_per_gpu))).to(device) for device in gpus]
        )

    def forward(self, x, ping=None):
        for device, layer_list in zip(self.gpus, self.blocks):
            x = x.to(device, non_blocking=True)
            x = layer_list(x)
        return x


class DummyCrowdsourcedNetwork(nn.Module):
    def __init__(self, hid_dim, block_factory, gpus, layers_per_gpu):
        super().__init__()
        self.gpu_devices = list(chain.from_iterable(repeat(gpus, layers_per_gpu)))
        self.layers = nn.ModuleList(
            [torch.jit.script(block_factory(hid_dim)).to(device) for device in self.gpu_devices]
        )

    def forward(self, x, ping):
        for device, layer in zip(self.gpu_devices, self.layers):
            x = x.to(device, non_blocking=True)
            x = layer(x)
            # emulate network lag
            time.sleep(ping * np.random.weibull(1))
        return x


def measure_perf(model_class, batches_for_latency, batches_for_throughput, throughput_runs, ping, input_factory, batch_size, hid_dim,
                 **kwargs):
    m = model_class(hid_dim=hid_dim, **kwargs)
    time_per_batch = []
    z = input_factory(batch_size, hid_dim).pin_memory()
    out_buf = input_factory(batch_size, hid_dim).pin_memory()
    with torch.no_grad():
        # latency: avg time to obtain a result per single processed batch
        for _ in range(batches_for_latency + 1):
            start = time.time()
            output = m(z, ping=ping)
            out_buf.copy_(output, non_blocking=True)
            torch.cuda.synchronize()
            time_per_batch.append(time.time() - start)
        # throughput: examples/sec when results are asynchronous
        throughputs = []
        for run in range(throughput_runs):
            start = time.time()
            for _ in range(batches_for_throughput):
                output = m(z, ping=ping)
                out_buf.copy_(output, non_blocking=True)
            torch.cuda.synchronize()
            throughputs.append(batch_size * batches_for_throughput / (time.time() - start))
    return np.mean(time_per_batch[1:]), np.std(time_per_batch[1:], ddof=1), np.mean(throughputs), np.std(throughputs, ddof=1)


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    measure_func = partial(measure_perf, batches_for_latency=args.batches_for_latency, batches_for_throughput=args.batches_for_throughput,
                           throughput_runs=args.throughput_runs, gpus=args.gpus, layers_per_gpu=args.layers_per_gpu, hid_dim=args.hid_dim,
                           block_factory=name_to_block[args.block_type], batch_size=args.batch_size,
                           input_factory=name_to_input[args.block_type])

    avg_latency, std_latency, avg_throughput, std_throughput = measure_func(ModelParallelNetwork, ping=0)
    print(f'ModelParallel (fast, ping=0.00):\t{avg_latency:.2f}±{std_latency:.2f}\t{avg_throughput:.2f}±{std_throughput:.2f}')

    for ping in np.linspace(0, args.max_ping, args.linspace_points):
        avg_latency, std_latency, avg_throughput, std_throughput = measure_func(DummyCrowdsourcedNetwork, ping=ping)
        print(f'ModelParallel (slow, ping={ping:.2f}):\t{avg_latency:.2f}±{std_latency:.2f}\t{avg_throughput:.2f}±{std_throughput:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hid-dim', type=int, default=1024)
    parser.add_argument('--batches-for-latency', type=int, default=10)
    parser.add_argument('--batches-for-throughput', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--throughput-runs', type=int, default=10)
    parser.add_argument('--max-ping', type=float, default=0.2)
    parser.add_argument('--linspace-points', type=int, default=10)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--layers-per-gpu', type=int, default=56)
    parser.add_argument('--block-type', choices=name_to_block.keys(), required=True)
    args = parser.parse_args()
    main(args)
