import numpy as np
import torch
from qp_ode_simulator.temporal_gnn import TemporalGNN, preprocess


def test_matched_parameters_and_no_cross_node_messages():
    a = np.ones((8,8),dtype=np.float32)-np.eye(8,dtype=np.float32)
    f = np.zeros((8,4),dtype=np.float32)
    torch.manual_seed(41)
    g = TemporalGNN(a,f,True)
    torch.manual_seed(41)
    control = TemporalGNN(a,f,False)
    assert sum(p.numel() for p in g.parameters())==sum(p.numel() for p in control.parameters())
    for p,q in zip(g.parameters(),control.parameters()):
        torch.testing.assert_close(p,q)
    torch.testing.assert_close(control.adjacency,torch.eye(8))
    g.eval()
    with torch.no_grad():
        y=g(torch.zeros(2,8,2047))
    assert y.shape==(2,2) and torch.isfinite(y).all()


def test_preprocessing_preserves_short_final_bin():
    x = np.zeros((2, 8, 17), dtype=np.uint8)
    x[..., -1] = 1
    q = np.full(8, .1)
    out = preprocess(x, q, 'binned')
    assert out.shape == (2, 8, 2)
    np.testing.assert_allclose(out[..., 0], -.1)
    np.testing.assert_allclose(out[..., 1], .9)
    assert preprocess(x, q, 'raw') is x
    np.testing.assert_allclose(preprocess(x, q, 'centered')[..., -1], .9)
