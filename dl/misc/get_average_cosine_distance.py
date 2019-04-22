"""
    Calculates the average cosine distance.
"""

import numpy as np
import pandas as pd
from sklearn import metrics


def average_cos(X):
    cosine_similarities = metrics.pairwise.cosine_distances(X)
    # print("Cosine similarities: ", cosine_similarities.shape)
    cos_similarities = np.average(cosine_similarities)
    print("Cosine distances: ", cos_similarities)
    print(np.min(cosine_similarities), np.max(cosine_similarities))
    print("Average simiilarity: ", cos_similarities)

if __name__ == "__main__":
    print("Cosine similarity")

    df = pd.read_csv("/Users/david/deeplearning/dl/misc/embeddings_e1_d50.tsv", sep="\t", index_col=None, header=None)
    # print(df.head(4))
    print(len(df))
    arr = df.values
    average_cos(arr)

# sklearn.metrics.pairwise.cosine_similarity(X)