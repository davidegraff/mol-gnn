from typing import Iterable

import torch
from torch import Tensor

from mol_gnn.data import BatchMolGraph
from mol_gnn.nn import MultiInputMessagePassing, Aggregation, Predictor
from mol_gnn.models.model import MPNN
from mol_gnn.nn.metrics import Metric


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MultiInputMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        w_t: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            w_t,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )
        self.message_passing: MultiInputMessagePassing

    def fingerprint(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_f is None else torch.cat((H, X_f), 1)
