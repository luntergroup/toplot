from jax import tree
import pandas as pd

def dataframe_to_pytree(df: pd.DataFrame, to_numpy=True) -> dict:
    """Convert a multi-level column DataFrame to a dictionary of NumPy arrays."""
    assert df.columns.nlevels == 2, "Generalization to deeper trees not implemented."
    keys = df.columns.get_level_values(0).unique()
    pytree = {k: df[k] for k in keys}
    metadata = tree.map(lambda x: x.columns, pytree)
    if to_numpy:
        pytree = tree.map(lambda x: x.to_numpy(), pytree)
    return pytree, metadata