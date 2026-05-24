import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchInvariantVAELoss(nn.Module):
    """
    Loss for a VAE where each batch contains N augmentations of a SINGLE
    underlying sample. Encourages within-batch invariance of the latent
    while preserving a non-degenerate distribution across batches.

    Expected inputs per forward call:
        mu       : (N, D)   encoder means for the N augmentations
        logvar   : (N, D)   encoder log-variances
        recon    : (N, *X)  decoder outputs
        target   : (*X,) or (N, *X)  shared reconstruction target
                   (e.g. a clean sample, or one chosen augmentation)
    """

    def __init__(
            self,
            latent_dim: int,
            lambda_rec: float = 1.0,
            lambda_inv: float = 1.0,
            lambda_kl:  float = 1e-3,
            lambda_var: float = 1.0,
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
        self.register_buffer("ema_mean",  torch.zeros(latent_dim))
        self.register_buffer("ema_sqmean", torch.ones(latent_dim))
        self.register_buffer("initialized", torch.tensor(False))

    # ---------- individual loss terms ----------

    def _reconstruction(self, recon, target):
        # Broadcast target if it's a single sample shared across the batch
        if target.dim() == recon.dim() - 1:
            target = target.unsqueeze(0).expand_as(recon)
        # Mean over batch and feature dims
        return F.mse_loss(recon, target, reduction="mean")

    def _invariance(self, mu):
        mean_mu = mu.mean(dim=0, keepdim=True)
        if self.stop_grad_on_mean:
            mean_mu = mean_mu.detach()
        return ((mu - mean_mu) ** 2).sum(dim=(1, 2)).mean()

    def _kl_on_aggregate(self, mu, logvar):
        """
        KL( N(mean_mu, mean_var) || N(0, I) ),
        where mean_var is the average per-sample variance in the batch.
        This regularizes the aggregate posterior, not each q(z|x_i).
        """
        mean_mu  = mu.mean(dim=0)                  # (D,)
        mean_var = logvar.exp().mean(dim=0)        # (D,)
        kl = 0.5 * (mean_var + mean_mu.pow(2) - 1.0 - mean_var.log())
        return kl.sum()

    @torch.no_grad()
    def _update_ema(self, batch_mean):
        m = self.ema_momentum
        if not bool(self.initialized):
            self.ema_mean.copy_(batch_mean)
            self.ema_sqmean.copy_(batch_mean.pow(2) + 1.0)  # seed with var=1
            self.initialized.fill_(True)
        else:
            self.ema_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            self.ema_sqmean.mul_(m).add_(batch_mean.pow(2), alpha=1 - m)

    def _variance_term(self, batch_mean):
        """
        VICReg-style hinge on the std of batch-mean latents *across batches*,
        estimated from EMA stats. The current batch_mean contributes a
        differentiable gradient; the EMA gives the running context.
        """
        # Mix current (differentiable) value with EMA (detached context)
        running_mean   = self.ema_mean
        running_sqmean = self.ema_sqmean
        # Use detached EMA for the "other" batches, current value for "this" one.
        # A simple practical estimator: blend them.
        blended_sqmean = 0.5 * batch_mean.pow(2) + 0.5 * running_sqmean
        blended_mean   = 0.5 * batch_mean        + 0.5 * running_mean
        var = (blended_sqmean - blended_mean.pow(2)).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        return F.relu(self.var_gamma - std).mean()

    def _covariance_term(self, mu):
        """
        Decorrelate latent dimensions across the batch.
        With a small batch this is noisy, but combined with the EMA-based
        variance term it still discourages collapse to a low-rank subspace.
        """
        N = mu.size(0)
        if N < 2:
            return mu.new_zeros(())
        centered = mu - mu.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (N - 1)         # (D, D)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / self.latent_dim

    # ---------- forward ----------

    def forward(self, mu, logvar, recon, target):
        rec = self._reconstruction(recon, target)
        inv = self._invariance(mu)
        kl  = self._kl_on_aggregate(mu, logvar)

        batch_mean = mu.mean(dim=0).detach()  # used only to update EMA
        var = self._variance_term(mu.mean(dim=0))   # differentiable through current batch
        cov = self._covariance_term(mu)

        self._update_ema(batch_mean)

        total = (
                self.lambda_rec * rec
                + self.lambda_inv * inv
                + self.lambda_kl  * kl
                + self.lambda_var * var
                + self.lambda_cov * cov
        )

        return total, {
            "rec": rec.detach(),
            "inv": inv.detach(),
            "kl":  kl.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
        }