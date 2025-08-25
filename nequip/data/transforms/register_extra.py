"""
Transform(s) to register extra data keys at runtime.

Use this to declare custom fields (e.g., 'hessian') as graph/node/edge fields so
the default collate/batching can handle them.

Example (config):

data:
  train_dataset:
    _target_: nequip.data.dataset.NequIPLMDBDataset
    file_path: path/to/train_nequip.lmdb
    transforms:
      - _target_: nequip.data.transforms.register_extra.RegisterKeys
        graph_fields: ["hessian"]
"""

from typing import Sequence, Dict

from .. import _key_registry


class RegisterKeys:
  def __init__(
    self,
    graph_fields: Sequence[str] = (),
    node_fields: Sequence[str] = (),
    edge_fields: Sequence[str] = (),
    long_fields: Sequence[str] = (),
    cartesian_tensor_fields: Dict[str, str] = {},
  ) -> None:
    # Register once at init; repeated calls are harmless.
    _key_registry.register_fields(
      graph_fields=graph_fields,
      node_fields=node_fields,
      edge_fields=edge_fields,
      long_fields=long_fields,
      cartesian_tensor_fields=cartesian_tensor_fields,
    )
    # Cache for reuse in __call__ (workers need to (re)register in their own process)
    self._graph_fields = tuple(graph_fields)
    self._node_fields = tuple(node_fields)
    self._edge_fields = tuple(edge_fields)
    self._long_fields = tuple(long_fields)
    self._cartesian_tensor_fields = dict(cartesian_tensor_fields)

  def __call__(self, data):
    # Re-register in worker processes; idempotent
    _key_registry.register_fields(
      graph_fields=self._graph_fields,
      node_fields=self._node_fields,
      edge_fields=self._edge_fields,
      long_fields=self._long_fields,
      cartesian_tensor_fields=self._cartesian_tensor_fields,
    )
    return data
