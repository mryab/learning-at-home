"""
A dictionary of numpy arrays stored in shared memory. Multiprocessing-friendly
"""
import multiprocessing as mp
import multiprocessing.shared_memory

import numpy as np

from .proto import ArrayProto


class SharedArrays:
    def __init__(self, array_headers=None, shm_manager=None):
        """
        A dict-like collection of numpy arrays shared between processes using multiprocessing.shared_memory
        :param array_headers: a shared dictionary of { key -> (ArrayProto, SharedMemory.name) }
            if not specified, creates a new shared dictionary using new multiprocessing.Manager()
        :param shm_manager: shared memory manager to be used when allocating new shared arrays
            if not specified, creates a new one
        """
        assert array_headers is None or isinstance(array_headers, mp.managers.DictProxy)
        assert shm_manager is None or isinstance(shm_manager, mp.managers.SharedMemoryManager)
        if array_headers is not None:
            self.array_headers = array_headers
        else:
            self.array_headers_manager = mp.Manager()
            self.array_headers = self.array_headers_manager.dict()
        if shm_manager is None:
            shm_manager = mp.managers.SharedMemoryManager()
            shm_manager.start()
        self.shm_manager = shm_manager

    def fork(self):
        """ create a linked instance of SharedArrays that uses the same data and shm_manager """
        return SharedArrays(self.array_headers, self.shm_manager)

    def __getitem__(self, key):
        proto, shmem_name = self.array_headers[key]
        return SharedArray(proto, mp.shared_memory.SharedMemory(name=shmem_name))

    def __contains__(self, key):
        return key in self.array_headers

    def __setitem__(self, key, arr):
        if not isinstance(arr, SharedArray):
            raise ValueError("setitem only works with SharedArray values. For normal arrays, use:\n"
                             "arr_shared = SharedArrays.create_array(key, ArrayProto.from_array(arr))\n"
                             "arr_shared[...] = arr  # note that arr not shared itself, but copied into a SharedArray")
        self.array_headers[key] = (ArrayProto.from_array(arr), arr.shared_memory.name)

    def __delitem__(self, key):
        del self.array_headers[key]

    def __repr__(self):
        return repr({key: self[key] for key in self.keys()})

    def __len__(self):
        return len(self.array_headers)

    def keys(self):
        return self.array_headers.keys()

    def create_array(self, key, proto: ArrayProto):
        """ Create and return a shared array under the specified key. if key already exists, overwrite """
        self[key] = shared_array = SharedArray(proto, self.shm_manager.SharedMemory(size=proto.nbytes))
        return shared_array


class SharedArray(np.ndarray):
    """
    A subclass of numpy array that stores SharedMemory as an attribute;
    Use this class to prevent SharedMemory buffer from accidentally getting deallocated
    Details on subclassing numpy: https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    #simple-example-adding-an-extra-attribute-to-ndarray
    """

    def __new__(subtype, proto: ArrayProto, shared_memory: mp.shared_memory.SharedMemory, offset=0):
        obj = super(SharedArray, subtype).__new__(
            subtype, proto.shape, proto.dtype, shared_memory.buf, offset, proto.strides, proto.order)
        obj.shared_memory = shared_memory
        return obj

    def __array_finalize__(self, obj):
        # make sure that shared memory is passed along to tensors that share its data, e.g. arr[::2]
        if obj is None: return  # explicit creation: do nothing
        self.shared_memory = getattr(obj, 'shared_memory', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.asarray(out_arr)  # after out-of-place operation we no longer need to store sharedmemory

    @classmethod
    def from_array(cls, arr: np.ndarray, shared_memory: mp.shared_memory.SharedMemory = None):
        """ Create SharedArray from a regular numpy array (out-of-place) """
        proto = ArrayProto.from_array(arr)
        shared_memory = shared_memory or mp.shared_memory.SharedMemory(create=True, size=proto.nbytes)
        proto.make_from_buffer(shared_memory.buf)[...] = arr
        return cls(proto, shared_memory)

    def __repr__(self):
        return super().__repr__() + '; shared_memory={}'.format(self.shared_memory)
