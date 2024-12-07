from typing import Literal

from jaxtyping import ArrayLike, Bool, Float
import torch
from torch import Tensor
from torch.nn import functional as F, nn
from torchmetrics.functional import classification

from mol_gnn.nn.loss import _BoundedMixin, _LossFunctionBase
from mol_gnn.utils import ClassRegistry


MetricRegistry = ClassRegistry()


@MetricRegistry.register("mae")
class MAEMetric(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        L = (preds - targets).abs()

        return self._reduce(L, mask, sample_weights)


@MetricRegistry.register("rmse")
class RMSEMetric(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        L = F.mse_loss(preds, targets, reduction="none")

        return self._reduce(L.sqrt(), mask, sample_weights)


@MetricRegistry.register("bounded-mae")
class BoundedMAEMetric(_BoundedMixin, MAEMetric):
    pass


@MetricRegistry.register("bounded-rmse")
class BoundedRMSEMetric(_BoundedMixin, RMSEMetric):
    pass


@MetricRegistry.register("r2")
class R2Metric(_LossFunctionBase):
    def forward(
        self,
        preds: Float[Tensor, "b t"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        target_means = (sample_weights * targets).mean(0, keepdim=True)

        rss = torch.sum(sample_weights * (preds - targets).square() * mask, dim=0)
        tss = torch.sum(sample_weights * (targets - target_means).square() * mask, dim=0)

        return ((1 - rss / tss) * self.task_weights).mean(0)


class _ClassificationMetricBase(nn.Module):
    def __init__(self, task: Literal["binary", "multilabel"]) -> None:
        super().__init__()

        self.task = task

    def extra_repr(self) -> str:
        return f"task={self.task}"


@MetricRegistry.register("roc")
class AUROCMetric(_ClassificationMetricBase):
    def forward(
        self,
        preds: Float[Tensor, "b t *k"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        targets = torch.where(mask, targets, -1).long()

        return classification.auroc(preds, targets, self.task, ignore_index=-1)


@MetricRegistry.register("prc")
class AUPRCMetric(_ClassificationMetricBase):
    def forward(
        self,
        preds: Float[Tensor, "b t *k"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        targets = torch.where(mask, targets, -1).long()

        return classification.average_precision(preds, targets, self.task, ignore_index=-1)


@MetricRegistry.register("accuracy")
class AccuracyMetric(_LossFunctionBase):
    def __init__(
        self,
        task_weights: Float[ArrayLike, "t"],
        task: Literal["binary", "multilabel"],
        threshold: float = 0.5,
    ):
        super().__init__(task_weights)

        self.task = task
        self.threshold = threshold

    def forward(
        self,
        preds: Float[Tensor, "b t *k"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        if self.task == "binary":
            preds = preds > self.threshold
        else:
            preds = preds.softmax(-1)

        L = preds == targets

        return self._reduce(L, mask, sample_weights)

    def extra_repr(self) -> str:
        return f"task={self.task}, threshold={self.threshold:0.1f}"


@MetricRegistry.register("F1")
class F1(nn.Module):
    def __init__(self, task: Literal["binary", "multilabel"], threshold: float = 0.5):
        super().__init__()

        self.task = task
        self.threshold = threshold

    def forward(
        self,
        preds: Float[Tensor, "b t *k"],
        targets: Float[Tensor, "b t"],
        mask: Bool[Tensor, "b t"],
        sample_weights: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        targets = torch.where(mask, targets, -1).long()

        return classification.f1_score(preds, targets, self.task, threshold=self.threshold)

    def extra_repr(self) -> str:
        return f"task={self.task}, threshold={self.threshold:0.1f}"
