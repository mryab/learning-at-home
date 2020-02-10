from dataclasses import dataclass, asdict

import numpy as np
import torch

DUMMY_BATCH_SIZE = 3  # used for dummy runs only


@dataclass(init=True, repr=True, frozen=True)
class ProtoBase:
    pass


@dataclass(init=True, repr=True, frozen=True)
class ArrayProto(ProtoBase):
    shape: tuple
    dtype: np.dtype
    strides: tuple = None
    order: str = 'C'

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(arr.shape, arr.dtype, strides=arr.strides, order='CF'[np.isfortran(arr)])

    def make_empty(self, **kwargs):
        properties = asdict(self)
        properties.update(kwargs)
        return np.ndarray(**properties)

    def make_from_buffer(self, buffer, offset=0):
        return np.ndarray(self.shape, self.dtype, buffer, offset,
                          strides=self.strides, order=self.order)

    @property
    def nbytes(self):
        return np.dtype(self.dtype).itemsize * np.prod(self.shape)


@dataclass(init=True, repr=True, frozen=True)
class TensorProto(ProtoBase):
    size: tuple
    dtype: torch.dtype = None
    layout: torch.layout = torch.strided
    device: torch.device = None
    requires_grad: bool = False
    pin_memory: bool = False

    @property
    def shape(self):
        return self.size

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor.shape, tensor.dtype, tensor.layout, tensor.device, tensor.requires_grad, tensor.is_pinned())

    def make_empty(self, **kwargs):
        properties = asdict(self)
        properties.update(kwargs)
        return torch.empty(**properties)

    def convert_array_to_tensor(self, array: np.ndarray):
        tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device)
        tensor = tensor.requires_grad_(self.requires_grad).to(self.device, non_blocking=True)
        return tensor.pin_memory() if self.pin_memory else tensor


@dataclass(init=True, repr=True, frozen=True)
class BatchTensorProto(TensorProto):
    """ torch Tensor with a variable 0-th dimension, used to describe batched data """

    def __init__(self, *instance_size, **kwargs):  # compatibility: allow initializing with *size
        if len(instance_size) == 1 and isinstance(instance_size[0], (list, tuple, torch.Size)):
            instance_size = instance_size[0]  # we were given size as the only parameter instead of *parameters
        super().__init__((None, *instance_size), **kwargs)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(*tensor.shape[1:], dtype=tensor.dtype, layout=tensor.layout,
                   device=tensor.device, requires_grad=tensor.requires_grad, pin_memory=tensor.is_pinned())

    def make_empty(self, batch_size, **kwargs):
        assert self.shape[0] is None, "Make sure 0-th dimension is not specified (set to None)"
        return super().make_empty(size=(batch_size, *self.shape[1:]), **kwargs)
