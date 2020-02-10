import sys
from argparse import ArgumentParser
from functools import partial
from itertools import chain
from multiprocessing import Pool
from time import time, sleep

import numpy as np
import torch
import torch.nn as nn

from layers import name_to_block, name_to_input

sys.path.append('../../')
import lib.client


class ExpertsWithLatency(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.Sequential(*experts)

    def forward(self, x, ping):
        for layer in self.experts:
            x = layer(x)
            sleep(ping * np.random.weibull(1))
        return x


@torch.no_grad()
def measure_perf(ping, model, x, num_batches):
    latencies = []
    for batch in range(num_batches + 1):
        start = time()
        output = model(x, ping=ping)
        latencies.append(time() - start)
    return latencies[1:]


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    remote_experts = []
    for address in args.hosts:
        host, port = address.split(':')
        experts_at_host = [lib.RemoteExpert(f'expert{i}', host=host, port=int(port)) for i in range(args.layers_per_gpu)]
        remote_experts.append(experts_at_host)
    experts = list(chain.from_iterable(zip(*remote_experts)))
    experts_with_delay = ExpertsWithLatency(experts)

    input = name_to_input[args.block_type](args.batch_size, args.hid_dim)
    measure_func = partial(measure_perf, model=experts_with_delay, x=input, num_batches=args.batches_for_throughput)

    with Pool(args.jobs) as p:
        for ping in np.linspace(0, args.max_ping, args.linspace_points):
            latencies = measure_perf(ping, experts_with_delay, input, args.batches_for_latency)

            throughputs = []
            for run in range(args.throughput_runs):
                processing_start = time()
                results = p.map(measure_func, [ping for _ in range(args.jobs)])
                # assume that all processes were working synchronously, then number of processed examples per second
                # needs to be summed by number of processes
                throughput = args.jobs * args.batch_size * (args.batches_for_throughput + 1) / (time() - processing_start)
                throughputs.append(throughput)
            avg_latency, std_latency = np.mean(latencies), np.std(latencies, ddof=1)
            avg_throughput, std_throughput = np.mean(throughputs), np.std(throughputs, ddof=1)
            print(f'ModelParallel (ours, ping={ping:.2f}):\t{avg_latency:.2f}±{std_latency:.2f}\t{avg_throughput:.2f}±{std_throughput:.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--hid-dim', type=int, default=1024)
    parser.add_argument('--batches-for-latency', type=int, default=10)
    parser.add_argument('--batches-for-throughput', type=int, default=100)
    parser.add_argument('--throughput-runs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--linspace-points', type=int, default=10)
    parser.add_argument('--layers-per-gpu', type=int, default=56)
    parser.add_argument('--block-type', choices=name_to_block.keys(), required=True)
    parser.add_argument('--max-ping', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
