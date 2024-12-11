from collections.abc import Collection, Mapping
from copy import copy
from dataclasses import dataclass, field
import textwrap

import pandas as pd
from rich.pretty import pretty_repr
import torch
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict

from mol_gnn.conf import INPUT_KEY_PREFIX, REPR_INDENT, TARGET_KEY_PREFIX
from mol_gnn.data.database.base import Database
from mol_gnn.transforms.managed import ManagedTransform
from mol_gnn.types import TransformConfig


@dataclass
class NotorchDataset(Dataset[dict]):
    records: list[dict] = field(init=False)
    targets: Mapping[str, torch.Tensor] = field(init=False)

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Mapping[str, TransformConfig],
        databases: Mapping[str, Database],
        target_groups: Mapping[str, list[str]],
    ):
        self.transforms = {name: ManagedTransform(**kwargs) for name, kwargs in transforms.items()}
        self.databases = databases
        self.target_groups = target_groups

    #     transform_columns = list(set(transform.in_key for transform in self.transforms.values()))
    #     self.records = self.df[transform_columns].to_dict("records")
    #     self.targets = {
    #         name: torch.as_tensor(self.df[columns].values)
    #         for name, columns in self.target_groups.items()
    #     }

    def __getitem__(self, idx: int) -> dict:
        sample = copy(self.records[idx])
        for transform in self.transforms.values():
            sample = transform(sample)
        for name, group in self.targets.items():
            sample[name] = group[idx]
        for name, db in self.databases.items():
            db_key = sample[db.in_key]
            sample[db.out_key] = db[db_key]

        return sample
        # dicts = [transform(record) for transform in self.transforms.values()]
        # out = reduce(lambda a, b: a | b, dicts, record)
        # extra_transform_data = {k: v for k, v in self.extra_transforms.items()}
        # extra_data = {key: value[idx] for key, value in self.extra_data.items()}
        # return sample_data | extra_data | extra_transform_data

    def collate(self, samples: Collection[dict]) -> TensorDict:
        batch = TensorDict({}, batch_size=len(samples))

        for transform in self.transforms.values():
            batch[f"{INPUT_KEY_PREFIX}.{transform.out_key}"] = transform.collate(samples)
        for name in self.target_groups:
            batch[f"{TARGET_KEY_PREFIX}.{name}"] = torch.as_tensor(
                [sample[name] for sample in samples]
            )

        return batch

    def to_dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, collate_fn=self.collate, **kwargs)

    def __repr__(self) -> str:
        prettify = lambda obj: pretty_repr(obj, indent_size=2)  # noqa: E731
        transform_repr = "\n".join(
            [
                "(transforms): {",
                textwrap.indent(
                    "\n".join(
                        f"({name}): {prettify(transform)}"
                        for name, transform in self.transforms.items()
                    ),
                    REPR_INDENT,
                ),
                "}",
            ]
        )
        databases_repr = "\n".join(
            [
                "(databases): {",
                textwrap.indent(
                    "\n".join(f"({name}): {db}" for name, db in self.databases.items()), REPR_INDENT
                ),
                "}",
            ]
        )
        databases_repr = "\n".join([f"(databases): {prettify(self.databases)}"])
        target_groups_repr = "\n".join([f"(target_groups): {prettify(self.target_groups)}"])

        return "\n".join(
            [
                f"{type(self).__name__}(",
                textwrap.indent(transform_repr, REPR_INDENT),
                textwrap.indent(databases_repr, REPR_INDENT),
                textwrap.indent(target_groups_repr, REPR_INDENT),
                ")",
            ]
        )


"""
NotorchDataset(
  (transforms):
    'smi_to_mol': SmiToMol(keep_h=True, add_hs=False)
    'smi_to_graph': Pipeline(
      (0): SmiToMol(...)
      (1): MolToGraph(
        (atom_transform): MultiTypeAtomTransform(
          (elements): [...]
          (num_hs): [...]
        )
        (bond_transform): MultiTypeBondTransform(
          (bond_types): [...]
          (stereos): [...]
        )
      )
    )
  )
  (databases):
    (qm_descs):
  (target_groups): {
    'regression': ['a', 'b', 'c']
    'classification': ['d', 'e']
  }
)
"""
