"""
    Some very basic analysis on the predictions
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, auc, average_precision_score, \
    precision_recall_curve

sns.set()
sns.set_style("whitegrid")

def _load_predictions():
    # Test Items
    y_hat = np.load("/Users/david/deeplearning/data/procsessed/embedding_test_predicted.npy").flatten()
    y = np.load("/Users/david/deeplearning/data/procsessed/embedding_test_real.npy")

    print("Loaded shapes are: ")
    print(y_hat.shape)
    print(y.shape)

    print(y_hat[:3])
    print(y[:3])

    return y_hat, y

def plot_magnitude_histogram(y, y_hat):
    """
        From this histogram, we can decide how decisive our model was

        Put the two histograms on top of each other.
        This can then act as a measure of how well the one distribution is
        captured through the predictions.
    :return:
    """

    sns.distplot(y_hat, bins=100)
    plt.xlim(-1., 1.)
    plt.show()


def main():
    y_hat, y = _load_predictions()
    plot_magnitude_histogram(y, y_hat)

if __name__ == "__main__":
    print("Starting prediction analysis")
    main()