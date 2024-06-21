import numpy as np
from typing import Union

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE 

def plot_pr_curves(true_labels: Union[np.ndarray, list], pred_probs_list: Union[np.ndarray, list], labels_list: Union[np.ndarray, list], title: str, path: str) -> None:
    """
    Shows multiple PR Curves in the same plot given the true labels and a list of predicted probabilities
    
    Args:
        true_labels (np.ndarray): Array of shape (N) where N is the number of samples. The values in the
        array must be exactly {0, 1} where 0 is interpreted as negative, and 1 as positive. These true labels should be the same
        for all the plots.

        pred_probs_list (list): List of arrays of shape (N) where N is the number of samples. The values in the
        array must be probabilities between [0, 1] where 0 is interpreted as negative, and 1 as positive. The length of the list
        must equals the number of curves to show in the plot.

        labels_list (list): List of labels to show in the legend of the plot (usually the names of the models used).

        title (str): Title of the plot.

        path (str): Name to save the plot.

    Raises:
        TypeError: true_labels is not a np.ndarray.
        TypeError: pred_probs_list is not a list.
        TypeError: labels_list is not a list.
        TypeError: title is not a str.
        TypeError: path is not a str.
        ValueError: The shape of true_labels and all the arrays in pred_probs_list do not match.
        ValueError: The length of labels_list does not equal the length of pred_probs_list.
    """

    if not (isinstance(true_labels, list) or isinstance(true_labels, np.ndarray)): raise TypeError(" 'true_labels' is not a np.ndarray.")
    if not (isinstance(pred_probs_list, list) or isinstance(pred_probs_list, np.ndarray)): raise TypeError(" 'pred_probs_list' is not a list or a np.ndarray.")
    if not (isinstance(labels_list, list) or isinstance(labels_list, np.ndarray)): raise TypeError(" 'labels_list' is not a list or a np.ndarray.")
    if not isinstance(title, str): raise TypeError(" 'title' is not a string.")
    if not isinstance(path, str): raise TypeError(" 'path' is not a string.")
    if not all([len(pred_probs) == len(true_labels) for pred_probs in pred_probs_list]): raise ValueError(" The shape of 'true_labels' and all the arrays in 'pred_probs_list' do not match.")
    if not len(pred_probs_list) == len(labels_list): raise ValueError("There should be the same amount of labels (" + str(len(labels_list)) + ") and predictions (" + str(len(pred_probs_list)) + ")")

    # Plot all curves
    for i in range(len(pred_probs_list)):

        # Plot one PR Curve
        # Compute the curve
        precision, recall , _ = precision_recall_curve(true_labels, pred_probs_list[i])

        # Plot the PR Curve
        plt.plot(recall[:-1], precision[:-1], linestyle = 'solid', linewidth = 3,  label = labels_list[i])

    # In the plot we show a blue line of a classifier which has no knowledge (which shows as a flat line)
    no_knowledge = len(true_labels[true_labels==1]) / len(true_labels)
    plt.plot([0, 1], [no_knowledge, no_knowledge], linestyle='--', label='No Knowledge')

    # Axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Title
    plt.title(title)

    # Show the legend
    plt.legend(bbox_to_anchor=(1.1, 1))

    # Save the figure and clear
    plt.savefig(path + '.png', bbox_inches='tight')
    plt.clf()

    return

def plot_roc_auc_curves(true_labels: np.ndarray, pred_probs_list: list, labels_list: list, title: str, path: str) -> None:
    """
    Shows multiple ROC-AUC Curves in the same plot given the true labels and a list of predicted probabilities
    
    Args:
        true_labels (np.ndarray): Array of shape (N) where N is the number of samples. The values in the
        array must be exactly {0, 1} where 0 is interpreted as negative, and 1 as positive. These true labels should be the same
        for all the plots.

        pred_probs_list (list): List of arrays of shape (N) where N is the number of samples. The values in the
        array must be probabilities between [0, 1] where 0 is interpreted as negative, and 1 as positive. The length of the list
        must equals the number of curves to show in the plot.

        labels_list (list): List of labels to show in the legend of the plot (usually the names of the models used).

        title (str): Title of the plot.

        path (str): Path to save the plot.

    Raises:
        TypeError: true_labels is not a np.ndarray.
        TypeError: pred_probs_list is not a list.
        TypeError: labels_list is not a list.
        TypeError: title is not a str.
        TypeError: path is not a str.
        ValueError: The shape of true_labels and all the arrays in pred_probs_list do not match.
        ValueError: The length of labels_list does not equal the length of pred_probs_list.
    """

    if not (isinstance(true_labels, list) or isinstance(true_labels, np.ndarray)): raise TypeError(" 'true_labels' is not a list or a np.ndarray.")
    if not isinstance(pred_probs_list, list): raise TypeError(" 'pred_probs_list' is not a list.")
    if not isinstance(labels_list, list): raise TypeError(" 'labels_list' is not a list.")
    if not isinstance(title, str): raise TypeError(" 'title' is not a string.")
    if not isinstance(path, str): raise TypeError(" 'path' is not a string.")
    if not all([len(pred_probs) == len(true_labels) for pred_probs in pred_probs_list]): raise ValueError(" The shape of 'true_labels' and all the arrays in 'pred_probs_list' do not match.")
    if not len(pred_probs_list) == len(labels_list): raise ValueError("There should be the same amount of labels (" + str(len(labels_list)) + ") and predictions (" + str(len(pred_probs_list)) + ")")

    # Plot all curves
    for i in range(len(pred_probs_list)):

        # Compute Receiver operating characteristic (ROC)
        fpr, tpr, _ = roc_curve(true_labels, pred_probs_list[i])

        # Area Under the Curve (AUC)
        auc_value =  auc(fpr, tpr)

        # Plot one ROC-AUC curve
        plt.plot(fpr, tpr, label = labels_list[i] + ' AUC = %0.3f' % auc_value, ls = 'solid')

    # Plot a no knowledge line (A classifier with no knowledge could be one that always predicts the majority class)
    fpr, tpr, _ = roc_curve(true_labels, [0 for _ in range(len(true_labels))])
    auc_value =  auc(fpr, tpr)
    plt.plot(fpr, tpr, label = 'No Knowledge' + ' AUC = %0.3f' % auc_value, linestyle='--')

    # Axis labels
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Title
    plt.title(title)

    # Show the legend
    plt.legend(bbox_to_anchor=(1.1, 1))

    # Save the figure and clear
    plt.savefig(path + '.png', bbox_inches='tight')
    plt.clf()

    return

def plot_tsne(embeddings: np.ndarray, labels: list, perplexity: int, path: str, title: str, n_classes: int = 2, random_state: int = 42 ) -> None:
    """
    Creates a 2D t-SNE plot of the given embeddings

    Args:
        embeddings (np.ndarray or scipy.sparse): The datapoints (embeddings or just raw/original datapoints) to plot.
        labels: List which contains the labels for the cells (can be strings).
        perplexity (int): Perplexity in t-SNE.
        path (str): Path to save the plot
        title (str): Title of the plot
        n_classes (int): Number of classes in the data (for color legend)
        random_state (int): Seed for initialization.

    Raises:
        TypeError: 'labels' is not a list.
        TypeError: 'perplexity' is not an int.
        TypeError: 'path' is not an str.
        TypeError: 'title' is not an str.
        TypeError: 'n_classes' is not an int.
        TypeError: 'random_state' is not an int.
    """

    if not (isinstance(labels, list) or isinstance(labels, np.ndarray)): raise TypeError(" 'labels' is not a list or a np.ndarray.")
    if not isinstance(perplexity, int): raise TypeError(" 'perplexity' is not an int.")
    if not isinstance(path, str): raise TypeError(" 'path' is not a string.")
    if not isinstance(title, str): raise TypeError(" 'title' is not a string.")
    if not isinstance(n_classes, int): raise TypeError(" 'n_classes' is not an int.")
    if not isinstance(random_state, int): raise TypeError(" 'random_state' is not an int.")

    # Compute TSNE
    tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = random_state)
    tsne = tsne.fit_transform(embeddings)

    # Scatter plot with the TSNE
    scatter = sns.scatterplot(
        x = [elem[0] for elem in tsne], 
        y = [elem[1] for elem in tsne],
        hue = labels,
        palette = sns.color_palette("hls", n_classes),
        alpha = 0.5
    )

    scatter.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=1)
    leg_handles, leg_labels = scatter.get_legend_handles_labels()

    leg = plt.legend(leg_handles, leg_labels, bbox_to_anchor=(1.1,0.5), loc='center right', borderaxespad=0)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        
    plt.xticks([])
    plt.yticks([])

    plt.title(title)

    # Save plot and clear
    plt.savefig(path + '.png', bbox_inches='tight')
    plt.clf()
    return