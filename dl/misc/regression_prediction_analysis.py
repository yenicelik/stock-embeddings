"""
    Some very basic analysis on the predictions
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, auc, average_precision_score, \
    precision_recall_curve

from scipy.stats import norm, beta

sns.set()
sns.set_style("whitegrid")

def _load_predictions():
    # Test Items
    y_hat = np.load("/Users/david/deeplearning/data/procsessed/embedding_test_predicted.npy").flatten()
    y = np.load("/Users/david/deeplearning/data/procsessed/embedding_test_real_regression.npy")

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

    sns.distplot(y_hat, bins=200, color="green", kde=True, label="predicted")
    sns.distplot(y, bins=200, color="red", fit=norm, kde=False, label="actual")
    plt.xlim(-1., 1.)
    plt.title("Distribution of Daily Returns (over 200 bins)")
    plt.xlabel("Return in USD")
    plt.ylabel("Frequency (in number of days)")
    plt.legend()
    plt.show()
    plt.clf()

    # sns.distplot(y_hat, fit=norm, bins=1000, color="green", kde=False)
    # sns.distplot(y, fit=norm, bins=1000, color="red", kde=False)
    # plt.xlim(-0.1, 0.1)
    # plt.show()
    # plt.clf()

def mae_plot(y, y_hat):
    print("Plotting histogram of MAEs")

    maes = np.abs(y - y_hat)

    sns.distplot(maes, bins=500, fit=beta, kde=False)
    plt.xlim(0, 0.3)
    plt.title("Error Histogram (over 500 bins)")
    plt.xlabel("Error (Mean Absolute Error)")
    plt.ylabel("Frequency")
    plt.show()


def main():
    y_hat, y = _load_predictions()
    # plot_magnitude_histogram(y, y_hat)
    mae_plot(y, y_hat)

if __name__ == "__main__":
    print("Starting prediction analysis")
    main()