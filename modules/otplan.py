from geomloss import SamplesLoss
from serialization import encode
import torch
import numpy as np
from loguru import logger

def serialize(pc, grid_size=0.01, depth=16, order="z"):
    _, order, _ = encode(pc, grid_size=grid_size, depth=depth, order=order)
    order = order.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    return torch.gather(pc, 1, order)

class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        p: int = 2,
        blur: float = 0.05,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        p: int, optional
            the power of the norm to use for the cost matrix (default is 2 for squared Euclidean distance)
        reg: float, optional
            regularization parameter to use for iterative solvers.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        assert p in [1, 2], "Only p=1 (L1) and p=2 (L2) norms are supported."
        # ot_fn should take (a, x0, b, x1) as arguments where a, b are marginals and
        # x0, x1 are the source and target coordinates
        self.ot_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True, backend="online")

        self.p = p
        self.blur = blur
        self.warn = warn
        self.orders = ["z", "hilbert", "z-trans", "hilbert-trans"]
    
    def get_map(self, x0, x1):
        B, N, D = x0.shape
        _, M, _ = x1.shape

        if getattr(self, "use_keops", False):
            from pykeops.torch import LazyTensor

            x0_i = LazyTensor(x0[:, :, None, :])   # (B, N, 1, D)
            x1_j = LazyTensor(x1[:, None, :, :])   # (B, 1, M, D)

            if self.p == 1:
                C_ij = ((x0_i - x1_j) ** 2).sum(-1).sqrt()   # (B, N, M, 1)
            else:
                C_ij = ((x0_i - x1_j) ** 2).sum(-1) / 2      # (B, N, M, 1)

            F, G = self.ot_fn(x0, x1)
            epsilon = self.blur ** self.p

            F_i = LazyTensor(F[:, :, None, None])   # (B, N, 1, 1)
            G_j = LazyTensor(G[:, None, :, None])   # (B, 1, M, 1)

            T = ((F_i + G_j - C_ij) / epsilon).exp()
            indices = T.argmax(dim=2)

        else:
            # Pure PyTorch fallback: works on Windows
            diff = x0[:, :, None, :] - x1[:, None, :, :]     # (B, N, M, D)
            sqdist = (diff ** 2).sum(dim=-1)                 # (B, N, M)

            if self.p == 1:
                C = sqdist.sqrt()                            # (B, N, M)
            else:
                C = sqdist / 2                               # (B, N, M)

            F, G = self.ot_fn(x0, x1)                        # F: (B, N), G: (B, M)
            epsilon = self.blur ** self.p

            scores = ((F[:, :, None] + G[:, None, :] - C) / epsilon).exp()  # (B, N, M)
            indices = scores.argmax(dim=2)                  # (B, N)

        indices = indices.unsqueeze(-1).expand(-1, -1, D)   # (B, N, D)
        x1_sel = torch.gather(x1, 1, indices)               # (B, N, D)
        return x0, x1_sel

    def sample_map(self, pi, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (B, N, M)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape ((B, N), (B, N))
            represents the indices of source and target data samples from $\pi$
        """
        B, N, M = pi.shape
        pi_flat = pi.reshape(B, -1)
        pi_flat = pi_flat / pi_flat.sum(dim=1, keepdim=True)
        # Sample indices for each batch
        choices = torch.multinomial(pi_flat, N, replacement=replace)  # (B, N)
        return torch.div(choices, M).long(), torch.remainder(choices, M).long()

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (B, N, D)
            represents the source minibatch
        x1 : Tensor, shape (B, M, D)
            represents the target minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (B, N, D)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (B, N, D)
            represents the target minibatch drawn from $\pi$
        """
        # B = x0.shape[0]
        x0, x1 = self.get_map(x0, x1)
        # i, j = self.sample_map(pi, replace=replace)
        # batch_indices = torch.arange(B, device=x0.device).unsqueeze(1).long()  # (B, 1)
        # return x0[batch_indices, i], x1[batch_indices, j]
        return x0, x1

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (B, N, D)
            represents the source minibatch
        x1 : Tensor, shape (B, M, D)
            represents the target minibatch
        y0 : Tensor, shape (B, N)
            represents the source label minibatch
        y1 : Tensor, shape (B, M)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (B, N, D)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (B, M, D)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (B, N)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (B, M)
            represents the target label minibatch drawn from $\pi$
        """
        B = x0.shape[0]
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, replace=replace)
        batch_indices = torch.arange(B, device=x0.device).unsqueeze(1)  # (B, 1)
        return (
            x0[batch_indices, i],
            x1[batch_indices, j],
            y0[batch_indices, i] if y0 is not None else None,
            y1[batch_indices, j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (B, times, N, D)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (B, times, N, D)
            represents the OT sampled trajectories over time.
        """
        B, times, N, D = X.shape
        indices = torch.arange(N, device=X.device).unsqueeze(0).expand(B, -1)  # (B, N)
        traj_indices = [indices]
        for t in range(times - 1):
            pi = self.get_map(X[:, t], X[:, t + 1])
            pi_flat = pi / pi.sum(dim=2, keepdim=True)  # Normalize over targets for each source
            # For each batch and each source, sample a target according to pi
            sampled_j = torch.multinomial(pi_flat, 1).squeeze(-1)  # (B, N)
            traj_indices.append(sampled_j)

        to_return = []
        for t in range(times):
            batch_indices = torch.arange(B, device=X.device).unsqueeze(1).long()
            to_return.append(X[:, t][batch_indices, traj_indices[t]])
        to_return = torch.stack(to_return, dim=1)
        return to_return