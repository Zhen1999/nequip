# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import os
import torch

# singular source of global dtype that dictates the dtype of the data
# Default to float64 (framework convention) but fall back to float32 on MPS,
# since the MPS backend does not support float64 tensors on device.

_ENV_OVERRIDE = os.environ.get("NEQUIP_GLOBAL_DTYPE", "").lower()

if _ENV_OVERRIDE in ("float32", "fp32", "32"):
	_GLOBAL_DTYPE = torch.float32
elif _ENV_OVERRIDE in ("float64", "fp64", "64"):
	_GLOBAL_DTYPE = torch.float64
else:
	# Auto-detect MPS and choose a safe dtype
	if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
		_GLOBAL_DTYPE = torch.float32
	else:
		_GLOBAL_DTYPE = torch.float64
