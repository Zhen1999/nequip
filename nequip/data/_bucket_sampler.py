from typing import Iterator, List, Dict, Optional

import random
from torch.utils.data import BatchSampler

from . import AtomicDataDict


class BucketByNumAtomsBatchSampler(BatchSampler):
    """BatchSampler that groups samples by similar number of atoms N.

    Args:
        dataset: a dataset whose items are AtomicDataDict.Type
        batch_size: number of samples per batch
        bucket_width: bucket size in atoms; samples with N in [k*w, (k+1)*w) are grouped
        shuffle: shuffle order of buckets and samples each epoch
        drop_last: drop incomplete last batch in each bucket
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        bucket_width: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        # don't call super().__init__ since we don't use sampler argument
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.bucket_width = int(bucket_width)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # Precompute number of atoms per sample
        self._Ns: List[int] = self._compute_Ns()
        self._buckets: Dict[int, List[int]] = {}
        for idx, N in enumerate(self._Ns):
            b = N // self.bucket_width
            self._buckets.setdefault(b, []).append(idx)

        # Materialize bucket keys for deterministic iteration
        self._bucket_ids: List[int] = sorted(self._buckets.keys())

    def _compute_Ns(self) -> List[int]:
        Ns: List[int] = []
        # Prefer direct data_list if available to avoid repeated IO
        data_list = getattr(self.dataset, "data_list", None)
        if data_list is not None:
            for d in data_list:
                Ns.append(int(d[AtomicDataDict.POSITIONS_KEY].shape[0]))
            return Ns
        # Fallback: index into dataset items
        for i in range(len(self.dataset)):
            d = self.dataset[i]
            Ns.append(int(d[AtomicDataDict.POSITIONS_KEY].shape[0]))
        return Ns

    def __iter__(self) -> Iterator[List[int]]:
        # Prepare per-epoch ordering
        bucket_ids = self._bucket_ids.copy()
        if self.shuffle:
            random.shuffle(bucket_ids)

        for b in bucket_ids:
            idxs = self._buckets[b].copy()
            if self.shuffle:
                random.shuffle(idxs)
            # Yield full batches
            bs = self.batch_size
            n_full = len(idxs) // bs
            for k in range(n_full):
                yield idxs[k * bs : (k + 1) * bs]
            # Remainder
            rem = len(idxs) - n_full * bs
            if rem > 0 and not self.drop_last:
                yield idxs[-rem:]

    def __len__(self) -> int:
        total = 0
        for b in self._bucket_ids:
            n = len(self._buckets[b])
            q, r = divmod(n, self.batch_size)
            total += q + (0 if (r == 0 or self.drop_last) else 1)
        return total
