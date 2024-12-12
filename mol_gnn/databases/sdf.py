from dataclasses import dataclass, field
from os import PathLike
from typing import Iterator, Self

import rdkit.Chem as Chem

from mol_gnn.databases.base import Database
from mol_gnn.types import Mol


@dataclass
class SDFDatabase(Database):
    path: PathLike

    supp: Chem.SDMolSupplier | None = field(init=False, default=None)

    def __post_init__(self):
        with open(self.path) as f:
            self.__size = sum(1 for line in f if line.startswith("$$$$"))

    def __len__(self) -> int:
        return self.__size

    def __getitem__(self, idx: int) -> Mol:
        try:
            return self.supp[idx]
        except TypeError:
            raise ValueError

    def __enter__(self) -> Self:
        self.supp = Chem.SDMolSupplier(str(self.path))

        return self

    def __exit__(self, *exc):
        self.supp = None

    def __iter__(self) -> Iterator:
        with self:
            return iter(self.supp)
