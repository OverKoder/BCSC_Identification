from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import ndarray

def histogram(data: ndarray, x_label: str, name: str):
    """
    Plots an histogram (counts) given a data array and a label

    Args:
        data (ndarray): Array containing the data
        x_label: Name of the X axis
        name: Name of file to save the histogram

    Raises:
        TypeError: data is not an ndarray
        TypeError: x_label is not an string
        TypeError: name is not an string
    """

    if not isinstance(data, ndarray): raise TypeError("'data' must be an ndarray")
    if not isinstance(x_label, str): raise TypeError("'x_label' must be an string")
    if not isinstance(name, str): raise TypeError("'name' must be an string")

    sns.histplot(data = data, )