import multiprocessing.managers
import sys
from argparse import ArgumentParser

import torch

from layers import name_to_block

sys.path.append('../../')
import lib


def main(args):
    inp_shape = (args.hid_dim,) if args.block_type == 'ffn' else (512, args.hid_dim)
    with multiprocessing.managers.SharedMemoryManager() as shm_manager, multiprocessing.Manager() as array_headers_manager:
        try:
            array_headers = array_headers_manager.dict()
            experts = {}
            for i in range(args.layers_per_gpu):
                expert = torch.jit.script(name_to_block[args.block_type](args.hid_dim))
                experts[f'expert{i}'] = lib.ExpertBackend(name=f'expert{i}',
                                                          expert=expert, opt=torch.optim.Adam(expert.parameters()),
                                                          args_schema=(lib.BatchTensorProto(*inp_shape),),
                                                          outputs_schema=lib.BatchTensorProto(*inp_shape),
                                                          max_batch_size=args.max_batch_size,
                                                          shm_manager=shm_manager, array_headers=array_headers,
                                                          pool_size=8)

            lib.TesseractServer(None, experts, port=args.port, conn_handler_processes=args.handler_processes,
                                sender_threads=4, device=torch.device('cuda', args.gpu),
                                start=True)
        except KeyboardInterrupt:
            print('Finishing')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-a', '--handler-processes', type=int, default=256)
    parser.add_argument('-p', '--port', type=int, required=True)
    parser.add_argument('--hid-dim', type=int, default=1024)
    parser.add_argument('--max-batch-size', type=int, default=2048)

    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--layers-per-gpu', type=int, default=56)
    parser.add_argument('--block-type', choices=name_to_block.keys(), required=True)
    args = parser.parse_args()
    main(args)
