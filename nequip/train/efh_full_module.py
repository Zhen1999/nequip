"""
EFHFullHessianLightningModule: train with energy/force loss plus FULL Hessian supervision or regularization.

Usage in config (example):

training_module:
  _target_: nequip.train.efh_full_module.EFHFullHessianLightningModule
  hessian_weight: 0.05
  hessian_supervised: true
  hessian_target_key: hessian         # target shape: (D,D) or (N,3,N,3) or flat D*D
  dtype_for_hessian: float64          # optional for more stable second derivatives
  loss:
    _target_: nequip.train.EnergyForceLoss
    per_atom_energy: true
    coeffs:
      total_energy: 1.0
      forces: 1.0

Notes:
- Full Hessian is O(D^2) memory/time; recommended for small to moderate systems (e.g., D<=~96).
- If hessian_supervised=false, the module applies a Frobenius-norm regularization on the full Hessian.
"""

from typing import Dict, Optional

import torch

from nequip.data import AtomicDataDict
from .lightning import NequIPLightningModule, _SOLE_MODEL_KEY


class EFHFullHessianLightningModule(NequIPLightningModule):
    def __init__(
        self,
        model: Dict,
        optimizer: Optional[Dict] = None,
        lr_scheduler: Optional[Dict] = None,
        loss: Optional[Dict] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        num_datasets: Optional[Dict[str, int]] = None,
        info_dict: Optional[Dict] = None,
        # EFH options
        hessian_weight: float = 0.1,
        hessian_supervised: bool = True,
        hessian_target_key: str = "hessian",  # target may be (D,D), (N,3,N,3), or flat (D*D)
        hessian_reduction: str = "mean",       # mean or sum over matrix entries
        dtype_for_hessian: Optional[str] = None,  # e.g., 'float64' for stability
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss=loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            num_datasets=num_datasets,
            info_dict=info_dict,
        )

        assert hessian_reduction in ("mean", "sum")
        self.hessian_weight = float(hessian_weight)
        self.hessian_supervised = bool(hessian_supervised)
        self.hessian_target_key = hessian_target_key
        self.hessian_reduction = hessian_reduction
        self.dtype_for_hessian = dtype_for_hessian

    @torch.no_grad()
    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean() if self.hessian_reduction == "mean" else x.sum()

    def _energy_sum_with_pos(self, pos: torch.Tensor, batch: AtomicDataDict.Type) -> torch.Tensor:
        # Re-evaluate the model with given positions and return scalar energy sum
        data = batch.copy()
        data[AtomicDataDict.POSITIONS_KEY] = pos
        out = self.model[_SOLE_MODEL_KEY](data)
        E = out[AtomicDataDict.TOTAL_ENERGY_KEY]
        return E.sum()

    def _full_hessian(self, batch: AtomicDataDict.Type, device: torch.device) -> torch.Tensor:
        # Build differentiable positions tensor matching batch
        pos0 = batch[AtomicDataDict.POSITIONS_KEY].detach().to(device)
        if self.dtype_for_hessian is not None:
            pos0 = pos0.to(getattr(torch, self.dtype_for_hessian))
        pos0.requires_grad_(True)

        # Use torch.autograd.functional.hessian to get full Hessian of scalar energy sum
        def fn(p):
            return self._energy_sum_with_pos(p, batch)

        H = torch.autograd.functional.hessian(fn, pos0, create_graph=False)
        # H shape: (N,3,N,3) — convert to (D,D)
        N = pos0.shape[0]
        D = N * 3
        H = H.permute(0, 1, 2, 3).contiguous().view(D, D)
        return H

    def _prepare_target_hessian(self, target: AtomicDataDict.Type, D: int, N: int, device, dtype) -> torch.Tensor:
        assert self.hessian_target_key in target, (
            f"Target dict missing `{self.hessian_target_key}`; provide full Hessian labels as (D,D), (N,3,N,3), or flat (D*D)."
        )
        Ht = target[self.hessian_target_key]
        Ht = torch.as_tensor(Ht, device=device, dtype=dtype)

        # Allow optional leading batch dim of size 1
        if Ht.ndim == 2 and Ht.shape == (1, D * D):
            Ht = Ht.view(D * D)
        if Ht.ndim == 1 and Ht.numel() == D * D:
            Ht = Ht.view(D, D)
        elif Ht.ndim == 2 and Ht.shape == (D, D):
            pass
        elif Ht.ndim == 4 and Ht.shape == (N, 3, N, 3):
            Ht = Ht.permute(0, 1, 2, 3).contiguous().view(D, D)
        elif Ht.ndim == 2 and Ht.shape[0] == N and (Ht.shape[1] % 9 == 0):
            # Accept per-atom padded representation (N, 9*K) where first N*9 entries encode (N,N,3,3)
            K = int(Ht.shape[1] // 9)
            # slice to actual NxN blocks and reshape
            Hn = Ht[:, : N * 9].contiguous().view(N, N, 9)
            Hn = Hn.view(N, N, 3, 3)
            Ht = Hn.permute(0, 2, 1, 3).contiguous().view(D, D)
        else:
            raise ValueError(
                f"Unexpected Hessian target shape {tuple(Ht.shape)}; expected flat {D*D}, (D,D), or (N,3,N,3)."
            )
        return Ht

    def training_step(self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0):
        target = self.process_target(batch, batch_idx, dataloader_idx)
        output = self(batch)

        # Optional train metrics (not part of loss)
        if self.train_metrics is not None:
            with torch.no_grad():
                train_metric_dict = self.train_metrics(
                    output, target, prefix=f"train_metric_step{self.logging_delimiter}"
                )
            self.log_dict(train_metric_dict)

        # Base loss (energy/force), with DDP scaling correction
        loss_dict = self.loss(output, target, prefix=f"train_loss_step{self.logging_delimiter}")
        self.log_dict(loss_dict)
        base_loss = loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"] * self.world_size

        # Full Hessian term — support batch_size > 1 by looping over frames
        device = base_loss.device
        num_frames = AtomicDataDict.num_frames(batch)
        per_frame_losses = []
        for i in range(num_frames):
            frame = AtomicDataDict.frame_from_batched(batch, i)
            frame_target = AtomicDataDict.frame_from_batched(target, i)
            # predicted full Hessian for this frame
            H_pred = self._full_hessian(frame, device=device)
            N = frame[AtomicDataDict.POSITIONS_KEY].shape[0]
            D = N * 3
            if self.hessian_supervised:
                H_t = self._prepare_target_hessian(
                    frame_target, D=D, N=N, device=device, dtype=H_pred.dtype
                )
                hess_term = (H_pred - H_t).pow(2)
            else:
                hess_term = H_pred.pow(2)
            per_frame_losses.append(self._reduce(hess_term))

        hess_loss = torch.stack(per_frame_losses).mean() if len(per_frame_losses) > 0 else torch.tensor(0.0, device=device)
        self.log(f"train_loss_step{self.logging_delimiter}hessian_full", hess_loss, prog_bar=False)

        total = base_loss + self.hessian_weight * hess_loss
        self.log(f"train_loss_step{self.logging_delimiter}total", total, prog_bar=True)
        return total
