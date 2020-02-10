## Learning@home
![img](./scheme.png)

Supplementary code for arXiv submission "Learning@home: Crowdsourced Training of Large Neural Networks with Decentralized Mixture-of-Experts".

__TL;DR:__ Learning@home is an approach for training large (up to multi-terabyte) neural networks on hardware provided by volunteers with unreliable and slow connection.

__This repository__ contains a snapshot of Learning@home that was used to conduct initial experiments. While this snapshot implements the main functionality of Learning@home, it should be treated as a testbed to reproduce our experiments, __not__ as a finished library (see limitations below). If the paper is accepted, we will release a newer version of the code suitable for large-scale training.


## What do I need to run it?
* One or several computers, each equipped with at least one GPU
* Each computer should have at least two open ports (if not, consider ssh port forwarding)
* Some popular Linux x64 distribution
  * Tested on Ubuntu16.04, should work fine on any popular linux64 and even MacOS;
  * Running on Windows natively is not supported, please use vm or docker;

## How do I run it?
1. Clone or download this repo. `cd` to its root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. Install packages from `requirements.txt`
4. Go to [./experiments](./experiments) and follow the README.md from there


## Learning@home quick tour

__Trainer process:__
  * __`RemoteExpert`__(`lib/client/remote_expert.py`) behaves like a pytorch module with autograd support but actually sends request to a remote runtime.
  * __`GatingFunction`__(`lib/client/gating_function.py`) finds best experts for a given input and either returns them as `RemoteExpert` or applies them right away.

__Runtime process:__
  * __`TesseractRuntime`__ (`lib/runtime/__init__.py`) aggregates batches and performs inference/training of experts according to their priority. 
  * __`TesseractServer`__ (`lib/server/__init__.py`) wraps runtime and periodically uploads experts into DHT.

__DHT:__
   * __`TesseractNetwork`__(`lib/network/__init__.py`) is a node of Kademlia-based DHT that stores metadata used by trainer and runtime.

## Limitations
As stated above, this implementation is a testbed for experiments, not the final learning@home library. More specifically:

* After finding best experts across DHT, a client still connects to these experts via hostname/port. Updated version connects to experts via DHT, allowing users to host servers with no public hostname or under NAT.
* Runtime processes do not handle errors. In the updated version, any errors on server are reported to the client.
* Runtime processes report basic rudimentary logs. Updated version reports multiple health indicators through Tensorboard.
* This implementation uses basic Kademlia protocol. Updated version modifies Kademlia to speed up searching for alive experts.
