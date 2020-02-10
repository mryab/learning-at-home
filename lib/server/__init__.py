import multiprocessing as mp
import os
from socket import socket, AF_INET, SOCK_STREAM, SO_REUSEADDR, SOL_SOCKET, timeout
from typing import Dict

from .connection_handler import handle_connection
from .network_handler import NetworkHandlerThread
from ..network import TesseractNetwork
from ..runtime import TesseractRuntime, ExpertBackend


class TesseractServer:
    def __init__(self, network: TesseractNetwork, expert_backends: Dict[str, ExpertBackend], addr='127.0.0.1',
                 port: int = 8080, conn_handler_processes: int = 1, update_period: int = 30, start=False,
                 **kwargs):
        self.network, self.experts, self.update_period = network, expert_backends, update_period
        self.addr, self.port = addr, port
        self.conn_handlers = conn_handler_processes
        self.runtime = TesseractRuntime(self.experts, **kwargs)

        if start:
            self.start()

    def start(self):
        if self.network:
            if not self.network.is_alive():
                self.network.start()

            network_thread = NetworkHandlerThread(experts=self.experts, network=self.network,
                                                  addr=self.addr, port=self.port, update_period=self.update_period)
            network_thread.start()

        processes = self.spawn_connection_handlers()
        try:
            self.runtime.main()
        finally:
            for process in processes:
                process.join()
            if self.network:
                network_thread.join()

    def spawn_connection_handlers(self):
        sock = socket(AF_INET, SOCK_STREAM)
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.listen()
        sock.settimeout(self.update_period)

        processes = [mp.Process(target=socket_loop, args=(sock, self.experts)) for _ in range(self.conn_handlers)]
        for process in processes:
            process.start()
        return processes


def socket_loop(sock, experts):
    """ catch connections, send tasks to processing, respond with results """
    print(f'Spawned connection handler pid={os.getpid()}')
    while True:
        try:
            handle_connection(sock.accept(), experts)
        except KeyboardInterrupt as e:
            print(f'Socket loop has caught {type(e)}, exiting')
            break
        except (timeout, BrokenPipeError, ConnectionResetError, NotImplementedError):
            continue
