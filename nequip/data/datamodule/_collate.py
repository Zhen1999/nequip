"""
Custom collate utilities.

Provides a collate_fn factory that pads per-atom Hessian rows to the
batch-local maximum number of atoms (K = max(N_i)) so that batching works
without needing dataset-global padding.

Usage in Hydra config, replace dataloader.collate_fn with this factory:

  train_dataloader:
    collate_fn:
      _target_: nequip.data.datamodule._collate.hessian_pad_collate_factory
      hessian_key: hessian

Same for val/test if needed.
"""

from typing import Callable, List

import torch
import torch.nn.functional as F

from .. import AtomicDataDict


class HessianPadCollate:
    """Picklable collate callable that right-pads per-atom Hessian rows to 9*K
    where K is the max number of atoms in the batch.

    Args:
        hessian_key: node field name for per-atom Hessian rows.
    """

    def __init__(self, hessian_key: str = "hessian") -> None:
        self.hessian_key = hessian_key

    def __call__(self, data_list: List[AtomicDataDict.Type]) -> AtomicDataDict.Type:
        # determine batch-local max N
        Ns = []
        for d in data_list:
            if AtomicDataDict.POSITIONS_KEY in d:
                Ns.append(int(d[AtomicDataDict.POSITIONS_KEY].shape[0]))
            else:
                Ns.append(int(AtomicDataDict.num_nodes(d)))
        K_batch = max(Ns) if len(Ns) > 0 else 0

        if K_batch > 0:
            for d, N in zip(data_list, Ns):
                if self.hessian_key not in d:
                    continue
                H = d[self.hessian_key]
                H = torch.as_tensor(H)
                # Expect (N, 9*K_i)
                if H.ndim != 2 or H.shape[0] != N or (H.shape[1] % 9) != 0:
                    continue
                # keep only the meaningful first 9*N columns
                width = 9 * N
                H = H[:, :width]
                # pad to 9*K_batch on the right
                target_width = 9 * K_batch
                if H.shape[1] < target_width:
                    pad_cols = target_width - H.shape[1]
                    H = F.pad(H, (0, pad_cols, 0, 0), mode="constant", value=0.0)
                d[self.hessian_key] = H

        return AtomicDataDict.batched_from_list(data_list)


def hessian_pad_collate_factory(hessian_key: str = "hessian") -> Callable:
    """Backward-compatible factory returning a picklable collate object."""
    return HessianPadCollate(hessian_key=hessian_key)
