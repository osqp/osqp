Torch
=============

In this example we show how to use OSQP.torch.nn.


Python
------

.. code:: python

    import numpy.random as npr
    import numpy as np
    import torch
    import scipy.sparse as spa

    from osqp.nn.torch import OSQP

    n_batch=1
    n=10
    m=3
    P_scale=1.0
    A_scale=1.0
    u_scale=1.0
    l_scale=1.0

    npr.seed(1)
    L = np.random.randn(n, n)
    P = spa.csc_matrix(P_scale * L.dot(L.T))
    x_0 = npr.randn(n)
    s_0 = npr.rand(m)
    A = spa.csc_matrix(A_scale * npr.randn(m, n))
    u = A.dot(x_0) + A_scale * s_0
    l = -10 * A_scale * npr.rand(m)
    q = npr.randn(n)
    true_x = npr.randn(n)

    P_idx = P.nonzero()
    P_shape = P.shape
    A_idx = A.nonzero()
    A_shape = A.shape

    P_torch, q_torch, A_torch, l_torch, u_torch, true_x_torch = [
        torch.DoubleTensor(x) if len(x) > 0 else torch.DoubleTensor() for x in [P.data, q, A.data, l, u, true_x]
    ]

    for x in [P_torch, q_torch, A_torch, l_torch, u_torch]:
        x.requires_grad = True

    problem = OSQP(
        P_idx,
        P_shape,
        A_idx,
        A_shape,
    )
    x_hats = problem(P_torch, q_torch, A_torch, l_torch, u_torch)

    dl_dxhat = x_hats.data - true_x_torch
    x_hats.backward(dl_dxhat)
