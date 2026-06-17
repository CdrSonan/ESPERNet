import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchInvariantVAELoss(nn.Module):
    """
    Loss for a VAE where each batch contains N augmentations of a SINGLE
    underlying sample. Encourages within-batch invariance of the latent
    while preserving a non-degenerate distribution across batches.

    Expected inputs per forward call:
        mu    : (N, T, D)   encoder means for the N augmentations
        logvar: (N, T, D)   encoder log-variances
        recon : (N, T, C)  decoder outputs
        target: (T, C) or (N, T, C)  shared reconstruction target
                (e.g. a clean sample, or one chosen augmentation)

    T (time dimension) is allowed to vary between batches.
    """

    def __init__(
            self,
            latent_dim: int,
            lambda_rec: float = 1.0,
            lambda_inv: float = 1.0,
            lambda_kl:  float = 1e-3,
            lambda_var: float = 0.01,
            lambda_cov: float = 0.04,
            var_gamma:  float = 1.0,          # target std for variance hinge
            ema_momentum: float = 0.99,
            eps: float = 1e-4,
            stop_grad_on_mean: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_rec = lambda_rec
        self.lambda_inv = lambda_inv
        self.lambda_kl  = lambda_kl
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.var_gamma  = var_gamma
        self.eps = eps
        self.stop_grad_on_mean = stop_grad_on_mean

        # EMA buffers for tracking the distribution of batch-mean latents
        # across training steps (no meta-batch required).
        self.ema_momentum = ema_momentum
        self.register_buffer("ema_mean",    torch.zeros(latent_dim))
        self.register_buffer("ema_sqmean",  torch.ones(latent_dim))
        self.register_buffer("initialized", torch.tensor(False))

    # ---------- individual loss terms ----------

    def _reconstruction(self, recon, target):
        # Broadcast target if it is a single (T, C) sample shared across the batch
        if target.dim() == recon.dim() - 1:
            target = target.unsqueeze(0).expand_as(recon)
        # Mean over batch, time and feature dims
        return F.mse_loss(recon, target, reduction="mean")

    def _invariance(self, mu):
        """
        Encourage consistency across batch items at each time step.
        mu: (N, T, D)
        """
        # Mean across batch dimension, keeping time and channel
        mean_mu = mu.mean(dim=0, keepdim=True)  # (1, T, D)
        if self.stop_grad_on_mean:
            mean_mu = mean_mu.detach()
        # Penalize deviation from batch mean at each time step
        # Sum over D, mean over N and T
        return ((mu - mean_mu) ** 2).sum(dim=2).mean()

    def _kl_div(self, mu, logvar):
        """
        Per-sample KL divergence: KL( N(mu_i, var_i) || N(0, I) )
        for each sample in the batch.

        mu, logvar: (N, T, D)
        Computes per-sample KL and averages over batch and time.
        """
        # KL per sample, timestep and dimension
        var = logvar.exp()
        kl = 0.5 * (var + mu.pow(2) - 1.0 - logvar)
        # sum over D, mean over N and T
        return kl.sum(dim=2).mean()

    @torch.no_grad()
    def _update_ema(self, batch_mean):
        """
        batch_mean: (D,) - the time-averaged, batch-averaged latent mean
        """
        m = self.ema_momentum
        if not bool(self.initialized):
            self.ema_mean = batch_mean.clone()
            self.ema_sqmean = batch_mean.clone().pow(2) + 1.0  # seed with var=1
            self.initialized.fill_(True)
        else:
            self.ema_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            self.ema_sqmean.mul_(m).add_(batch_mean.pow(2), alpha=1 - m)

    def _variance_term(self, batch_mean):
        """
        VICReg-style hinge on the std of batch-mean latents *across batches*,
        estimated from EMA stats. The current batch_mean contributes a
        differentiable gradient; the EMA gives the running context.

        batch_mean: (D,) - time-averaged, batch-averaged latent
        """
        running_mean = self.ema_mean
        running_sqmean = self.ema_sqmean
        # Use detached EMA for the "other" batches, current value for "this" one.
        blended_sqmean = 0.5 * batch_mean.pow(2) + 0.5 * running_sqmean
        blended_mean = 0.5 * batch_mean + 0.5 * running_mean
        var = (blended_sqmean - blended_mean.pow(2)).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        return F.relu(self.var_gamma - std).mean()

    def _covariance_term(self, mu):
        """
        Decorrelate latent dimensions across the batch.
        mu: (N, T, D)

        We flatten batch and time together for covariance computation,
        treating each (batch_item, timestep) as a sample.
        """
        N, T, D = mu.shape
        # Reshape to (N*T, D)
        mu_flat = mu.reshape(N * T, D)
        n_samples = N * T
        if n_samples < 2:
            return mu.new_zeros(())
        centered = mu_flat - mu_flat.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (n_samples - 1)  # (D, D)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / self.latent_dim

    # ---------- forward ----------

    def forward(self, mu, logvar, recon, target):
        """
        mu, logvar: (N, T, D)
        recon: (N, T, C)
        target: (T, C) or (N, T, C)
        """
        rec = self._reconstruction(recon, target)
        inv = self._invariance(mu)
        kl = self._kl_div(mu, logvar)

        # For EMA and variance term, use time-averaged then batch-averaged mean
        # This gives a single (D,) vector representing the "average latent" of this batch
        batch_mean = mu.mean(dim=(0, 1)).detach()
        self._update_ema(batch_mean)

        var = self._variance_term(mu.mean(dim=(0, 1)))  # differentiable through current batch
        cov = self._covariance_term(mu)

        total = (
                self.lambda_rec * rec
                + self.lambda_inv * inv
                + self.lambda_kl * kl
                + self.lambda_var * var
                + self.lambda_cov * cov
        )

        return total, {
            "rec": rec.detach().item(),
            "inv": inv.detach().item(),
            "kl": kl.detach().item(),
            "var": var.detach().item(),
            "cov": cov.detach().item(),
        }