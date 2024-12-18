from __future__ import annotations

from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from mol_gnn.data.models.batch import MpnnBatch
from mol_gnn.data.models.graph import BatchedGraph
from mol_gnn.nn import LossFunction, Predictor
from mol_gnn.nn.encoder import GraphEncoder
from mol_gnn.nn.metrics import Metric
from mol_gnn.schedulers import NoamLR


class MPNN(pl.LightningModule):
    r"""An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine.

    The first two modules calculate learned fingerprints from an input molecule
    reaction graph, and the final module takes these leared fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassingBlock
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level predictor
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=True
        if `True`, apply batch normalization to the output of the aggregation operation
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    w_t : Tensor | None, default=None
        the weights to use for each task during training. If `None`, use uniform weights
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor function
    """

    def __init__(
        self,
        encoder: GraphEncoder,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        w_t: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "agg", "predictor"])
        # self.hparams.update(
        #     {
        #         "message_passing": message_passing.hparams,
        #         "agg": agg.hparams,
        #         "predictor": predictor.hparams,
        #     }
        # )

        self.encoder = encoder
        self.bn = nn.BatchNorm1d(self.encoder.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor

        # NOTE(degraff): should think about how to handle no supplied metric
        self.metrics = (
            [*metrics, self.criterion]
            if metrics
            else [self.predictor._default_metric, self.criterion]
        )
        w_t = torch.ones(self.n_tasks) if w_t is None else torch.tensor(w_t)
        self.task_weights = nn.Parameter(w_t.unsqueeze(0), False)

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> LossFunction:
        return self.predictor.criterion

    def fingerprint(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """The learned fingerprints for the input molecules"""
        H = self.encoder(G, V_d, len(G))
        H = self.bn(H)

        return H if X_f is None else torch.cat((H, X_f), 1)

    def encoding(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """The final hidden representations for the input molecules"""
        return self.predictor[:-1](self.fingerprint(G, V_d, X_f))

    def forward(
        self, G: BatchedGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(G, V_d, X_f))

    def training_step(self, batch: MpnnBatch, batch_idx):
        G, V_d, X_f, targets, w_s, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(G, V_d, X_f)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, w_s, self.task_weights, lt_mask, gt_mask)

        self.log("train/loss", l, prog_bar=True)

        return l

    def validation_step(self, batch: MpnnBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))
        self.log("val_loss", losses[0], batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: MpnnBatch, batch_idx: int = 0):
        losses = self._evaluate_batch(batch)
        metric2loss = {f"test/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, batch_size=len(batch[0]))

    def _evaluate_batch(self, batch) -> list[Tensor]:
        G, V_d, X_f, targets, _, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(G, V_d, X_f)

        return [
            metric(preds, targets, mask, None, None, lt_mask, gt_mask)
            for metric in self.metrics[:-1]
        ]

    def predict_step(self, batch: MpnnBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
            molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
            per task. The final dimension is flattened, so that the targets for each task are
            grouped. I.e., the first ``t`` elements are the first target for each task, the second
            ``t`` elements the second target, etc.
            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        G, X_vd, X_f, *_ = batch

        return self(G, X_vd, X_f)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)

        lr_sched = NoamLR(
            opt,
            self.warmup_epochs,
            self.trainer.max_epochs,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
            self.init_lr,
            self.max_lr,
            self.final_lr,
        )
        lr_sched_config = {
            "scheduler": lr_sched,
            "interval": "step" if isinstance(lr_sched, NoamLR) else "batch",
        }

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        hparams = torch.load(checkpoint_path)["hyper_parameters"]

        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
        }

        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )
