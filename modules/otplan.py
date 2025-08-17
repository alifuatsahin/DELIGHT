import warnings

import numpy as np
from geomloss import SamplesLoss
import torch
from loguru import logger


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
        self.ot_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, potentials=True)

        self.p = p
        self.blur = blur
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (B, N, D)
            represents the source minibatch
        x1 : Tensor, shape (B, M, D)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (B, N, M)
            represents the OT plan between minibatches
        """
        B, N, D = x0.shape
        _, M, _ = x1.shape

        a = torch.ones(B, N, device=x0.device) / N
        b = torch.ones(B, M, device=x1.device) / M

        if self.p == 1:
            C = torch.cdist(x0, x1, p=2) # Norm2(X-Y)
        else:
            C = 0.5 * torch.cdist(x0, x1, p=2) ** 2 # (SqDist(X,Y) / IntCst(2))

        F, G = self.ot_fn(a, x0, b, x1) # F: (B,N), G: (B,M)

        epsilon = self.blur ** self.p

        T = torch.exp((F[:, :, None] + G[:, None, :] - C) / epsilon) * (a[:, :, None] * b[:, None, :])

        if not torch.all(torch.isfinite(T)):
            logger.error("ERROR: T is not finite")
            logger.error(T)
            logger.error("Cost mean, max", C.mean(), C.max())
            logger.error(x0, x1)
        if torch.abs(T.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            T = torch.ones_like(T) * (a[:, :, None] * b[:, None, :])
        return T

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
        x1[j] : Tensor, shape (B, M, D)
            represents the target minibatch drawn from $\pi$
        """
        B = x0.shape[0]
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, replace=replace)
        batch_indices = torch.arange(B, device=x0.device).unsqueeze(1).long()  # (B, 1)
        return x0[batch_indices, i], x1[batch_indices, j]

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