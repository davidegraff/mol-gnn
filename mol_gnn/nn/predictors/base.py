from mol_gnn.nn.loss import LossFunction
from mol_gnn.utils import ClassRegistry, HasHParams


from torch import Tensor, nn


from abc import abstractmethod


class Predictor(nn.Module, HasHParams):
    r"""A :class:`Predictor` is a protocol that defines a differentiable function
    :math:`f : \mathbb R^d \mapsto \mathbb R^o"""

    input_dim: int
    """the input dimension"""
    output_dim: int
    """the output dimension"""
    n_tasks: int
    """the number of tasks `t` to predict for each input"""
    n_targets: int
    """the number of targets `s` to predict for each task `t`"""
    criterion: LossFunction
    """the loss function to use for training"""

    @abstractmethod
    def forward(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, Z: Tensor) -> Tensor:
        pass


PredictorRegistry = ClassRegistry[Predictor]()