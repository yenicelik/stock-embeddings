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

    y_hat = y_hat > 0.0

    print(y_hat[:3])
    print(y[:3])

    return y_hat, y

def confusion_plot(real, predicted):

    # confusion_matrix = pd.crosstab(real, predicted, rownames=['Actual'], colnames=['Predicted'])
    # sns.heatmap(confusion_matrix, annot=True)
    # plt.show()

    conf_mat = confusion_matrix(real, predicted)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum()
    print("Normalized confusion matrix")
    print(conf_mat_normalized)
    sns.heatmap(conf_mat_normalized)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    precision = conf_mat_normalized[1, 1] / (conf_mat_normalized[1, 1] + conf_mat_normalized[0, 1])
    recall = conf_mat_normalized[1, 1] / (conf_mat_normalized[1, 0] + conf_mat_normalized[1, 1])
    print("Prediction, Recall, F1 are: ")
    print(precision)
    print(recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f1)

def auc_plot(real, predicted):
    auc = roc_auc_score(real, predicted)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(real, predicted)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def precision_recall_plot(real, predicted):
    """ Use when there is a lot of imbalance between classes! """
    precision, recall, thresholds = precision_recall_curve(real, predicted)
    # calculate F1 score
    f1 = f1_score(real, predicted)
    # calculate precision-recall AUC
    auc_ = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(real, predicted)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc_, ap))

    print("Thresholds are: ", thresholds)

    # # plot no skill
    # plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # # plot the precision-recall curve for the model
    # plt.plot(recall, precision, marker='.')
    # # show the plot
    # plt.show()

def main():
    y_hat, y = _load_predictions()
    confusion_plot(y, y_hat)
    auc_plot(y, y_hat)
    precision_recall_plot(y, y_hat)

if __name__ == "__main__":
    print("Starting prediction analysis")
    main()