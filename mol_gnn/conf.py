"""Global configuration variables for mol_gnn"""

DEFAULT_ATOM_DIM, DEFAULT_BOND_DIM = MolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUTPUT_DIM = DEFAULT_HIDDEN_DIM

DEFAULT_ATOM_HIDDEN = 2 * DEFAULT_ATOM_DIM
DEFAULT_BOND_HIDDEN = 2 * DEFAULT_BOND_DIM
DEFAULT_MESSAGE_DIM_2 = DEFAULT_ATOM_HIDDEN + DEFAULT_BOND_HIDDEN
DEFAULT_OUTPUT_DIM_2 = DEFAULT_MESSAGE_DIM_2
