from abc import abstractmethod
from typing import Generic, TypeVar

from mol_gnn.featurizers.graph.graph import Graph

T = TypeVar("T")


class GraphFeaturizer(Generic[T]):
    """A :class:`GraphFeaturizer` featurizes inputs into :class:`Graph`s"""

    @abstractmethod
    def __call__(self, x: T) -> Graph:
        """Featurize the input :attr:`x` into a :class:`MolGraph`

        Parameters
        ----------
        mol : ≠Chem.Mol
            the input molecule
        atom_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated atom features
        bond_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated bond features

        Returns
        -------
        MolGraph
            the molecular graph of the molecule
        """

    @property
    def shape(self) -> tuple[int, int]:
        """the feature dimensions of the vertices and edges, respectively, of
        :class:`Graph`s generated by this featurizer"""