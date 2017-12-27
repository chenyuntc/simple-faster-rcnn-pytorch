"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, t.autograd.Variable):
        return tonumpy(data.data)


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t._TensorBase):
        tensor = data
    if isinstance(data, t.autograd.Variable):
        tensor = data.data
    if cuda:
        tensor = tensor.cuda()
    return tensor


def tovariable(data):
    if isinstance(data, np.ndarray):
        return tovariable(totensor(data))
    if isinstance(data, t._TensorBase):
        return t.autograd.Variable(data)
    if isinstance(data, t.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" %type(data))


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t._TensorBase):
        return data.view(1)[0]
    if isinstance(data, t.autograd.Variable):
        return data.data.view(1)[0]
