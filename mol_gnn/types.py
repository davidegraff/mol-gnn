# ruff: noqa: F401
from typing import Callable, Literal, Required, TypedDict

from rdkit.Chem import Mol, Atom, Bond
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

type Rxn = tuple[Mol, Mol]
type TensorDictKey = tuple[str, ...] | str


class ModelModuleConfig(TypedDict):
    module: Callable
    in_keys: list[TensorDictKey] | dict[TensorDictKey, str]
    out_keys: list[TensorDictKey]


class LossModuleConfig(TypedDict):
    weight: float
    module: Callable[..., Tensor]
    in_keys: list[TensorDictKey] | dict[TensorDictKey, str]


class LRSchedConfig(TypedDict, total=False):
    scheduler: Required[LRScheduler]
    interval: Literal["step", "epoch"]
    frequency: int
    monitor: str
    strict: bool
    name: str | None
