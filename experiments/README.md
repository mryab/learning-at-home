# Running the expertiments

This folder contains scripts and notebooks necessary for reproducing the results reported in the paper. 
To run them, please refer to the corresponding subsections of this guide.

### Throughput

All three scripts are contained in the folder `throughput` and are ready for customized benchmark runs. 

To run the baseline with parameters from the paper, use 

```python baseline_throughput.py --batches-for-latency 5 --batches-for-throughput 10 --batch-size 4 --throughput-runs 5 --linspace-points 10 --block-type transformer --layers-per-gpu 56 --gpus 0 1 2 3``` 

For testing Learning@home throughput under latency, first start the server for each GPU you have with 

```python throughput_server.py -a 16 -p PORT_NUMBER --block_type BLOCK_TYPE --gpu GPU_NUMBER```
 
 and then run a multiple-trainer client with commands like
 
```python throughput_client.py -j 64 --batches-for-latency 5 --batches-for-throughput 2 --throughput-runs 5 --linspace-points 10 --layers-per-gpu 56 --block-type ffn --hosts HOSTAME1:PORT_NUMBER1 HOSTAME2:PORT_NUMBER2”```
 
```python throughput_client.py -j 64 --batches-for-latency 5 --batches-for-throughput 2 --throughput-runs 5 --linspace-points 10 --layers-per-gpu 56 --block-type transformer --max-ping 0.2 --hosts HOSTAME1:PORT_NUMBER1 HOSTAME2:PORT_NUMBER2 --batch-size 4```

### Convergence
This experiment can be conducted both in a distributed setting and with an emulator. We recommend using the emulator to make results hardware-agnostic and reduce variance due to CPU and network interference from other processes.

You can find notebooks for [DMoE with 64 experts](./convergence/convergence_mnist_64workers_1000ms_seed1337_dmoe64x4.ipynb) and [large FFN](./convergence/convergence_mnist_64workers_1000ms_seed1337_largeffn.ipynb) in [`./convergence`](./convergence).

In order to reproduce our results, one should run these notebooks with 5 different random seeds aggregate statistics saved in the last cell. Please note that these experiments can take up a lot of GPU memory due to storing "stale" gradients. With 16 workers, the code should fit well into consumer GPU. For 64 workers, we bypassed the memory limit by sending gradients to `.cpu()` at the cost of longer experiment time.

### Gating function over DHT
We also provide a reference implementation of DMoE gating function over Kademlia DHT via `lib.GatingFunction`.

In order to test our implementation, you need to do two things:

First, set up DHT with at least one server process:
```python
import torch
import lib

# initial kademlia node
node_zero = lib.TesseractNetwork(port=ROOT_PORT, start=True)


# create experts. Warning: expert uids must be unique
experts = {}
for expert_uid in expert_uids:
    expert = torch.jit.script(NetworkBlock(1024))
    expert_backend = lib.ExpertBackend(
        name=expert_uid, expert=expert, opt=torch.optim.Adam(expert.parameters(), amsgrad=True),
        args_schema=(lib.BatchTensorProto(1024),), outputs_schema=lib.BatchTensorProto(1024),
        max_batch_size=2048, pool_size=8)
    experts[expert_uid] = expert_backend

# set up server(s)
runtime = lib.TesseractServer(lib.TesseractNetwork(('127.0.0.1', ROOT_PORT), port=SOME_OTHER_PORT, start=rue),
                              experts, port=PORTS[0], conn_handler_processes=64,
                              sender_threads=1, device=torch.device('cuda'),
                              start=True)
# after creating node_zero you can create additional TesseractServer instances in separate processes
```

Second, create a client process and connect to any DHT node:
```python
import torch
import lib

# create one or several backends with expert uids following the "expert.[0-32).[0-32)" pattern
# all backends must have TesseractNetwork active

network = lib.TesseractNetwork(('127.0.0.1', ROOT_PORT), port=SOME_NEW_PORT, start=True)
dmoe = lib.GatingFunction(in_features=1024, grid_size=[32, 32], k_best=4, network=network, uid_prefix='expert')

average_out = dmoe(torch.randn(32, 1024))
average_out.sum().backward()
```
