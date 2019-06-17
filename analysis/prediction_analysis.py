"""
    Analysis of the predicted data
"""

import numpy as np
np.set_printoptions(precision=20)

class PredictionAnalysis:
    """
        Analysis of the Predictions
    """

    def __init__(self):
        self.Y_pred = np.load("/Users/david/Desktop/Leonhard Deep Learning Predictions/test_predicted.npy")
        self.Y_real = np.load("/Users/david/Desktop/Leonhard Deep Learning Predictions/test_real.npy")
        print("Loaded: ", self.Y_pred.shape, self.Y_real.shape)

        print(self.Y_pred[:10])
        print("Next")
        print(self.Y_real[:10])

        # Count number of "True"s:
        print(np.count_nonzero(self.Y_real))
        print(np.count_nonzero(self.Y_real)/float(self.Y_real.shape[0]))


if __name__ == "__main__":
    print("Stuff")
    prediction_analysis = PredictionAnalysis()
    # prediction_analysis

