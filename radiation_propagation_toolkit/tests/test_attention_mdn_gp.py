import numpy as np
import torch
from qp_ode_simulator.attention_mdn_gp import AttentionPosition,MixturePosition,nn_predict


def test_mdn_loss_and_shapes():
    model=MixturePosition(4,64);x=torch.zeros(5,57);output=model(x)
    loss=MixturePosition.nll(output,torch.zeros(5,2)).mean()
    assert torch.isfinite(loss);loss.backward()
    torch.testing.assert_close(output[0].exp().sum(1),torch.ones(5))
    p,posterior=nn_predict(model,x.numpy(),np.zeros(2),2.)
    assert p.shape==(5,2) and posterior[1].shape==(5,4,2)


def test_attention_reload():
    torch.set_num_threads(2)
    a=AttentionPosition(np.zeros((8,4),dtype=np.float32),1);b=AttentionPosition(np.zeros((8,4),dtype=np.float32),1)
    b.load_state_dict(a.state_dict());x=np.zeros((2,8,2047),dtype=np.float32)
    p,_=nn_predict(a,x,np.zeros(2),1.);q,_=nn_predict(b,x,np.zeros(2),1.)
    np.testing.assert_allclose(p,q);assert p.shape==(2,2)
