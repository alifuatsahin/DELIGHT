import torch
import torch.nn as nn

class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            if len(states) == 3:  # conditional CNF
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=1)
                dy = self.diffeq(tc, y)
                divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)
                return dy, -divergence, torch.zeros_like(c).requires_grad_(True)
            elif len(states) == 2:  # unconditional CNF
                dy = self.diffeq(t, y)
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
                return dy, -divergence
            else:
                assert 0, "`len(states)` should be 2 or 3"


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
        % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx
