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

def array_to_twolevel_dataframe(array, column_name_dictionary):
    """Convert a numpy array to a two-level dataframe.
    array: {np.array} of dimensions [samples, words]
    column_name_dictionary: dictionary of format {featurename: [words]}, thus dictionary with lists
    """

    two_level_data_dictionary = {}
    for f, (featurename, words) in enumerate(column_name_dictionary.items()):
        if f < np.shape(array)[1]:
            print(f, words)
            two_level_data_dictionary[featurename] = pd.DataFrame(
                array[:, f : f + len(words)], columns=words
            )
    two_level_data_frame = pd.concat(two_level_data_dictionary, axis="columns")
    return two_level_data_frame