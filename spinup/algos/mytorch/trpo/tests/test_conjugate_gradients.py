import pytest

import torch

from spinup.algos.mytorch.trpo.core import conjugate_gradients

@pytest.mark.parametrize("A,x,b", [
    (torch.eye(2), 10.*torch.rand(2), torch.ones(2)),
    (torch.diag(torch.tensor([1., 2., 3., 4.])), torch.rand(4), torch.ones(4)),
    (torch.diag(torch.rand(20)+1.), torch.rand(20), torch.ones(20)),
])
def test_conjugate_gradients(A, x, b, max_iters=10, eps=1e-5):
    def Ax(v):
        return A @ v

    s = conjugate_gradients(Ax, x, b, max_iters, eps)

    assert torch.allclose(s, torch.inverse(A) @ b, atol=eps)
