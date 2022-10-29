import torch
import numpy as np
from torch.autograd import Variable


def f(x):
    return x ** 2


x = torch.autograd.Variable(torch.Tensor([3]), requires_grad=True)
g1 = torch.autograd.grad(f(x), x)
print(g1)
x = x * 2
g2 = torch.autograd.grad(f(x), x)
print(g2)
