# deeplearning - two sigma challenge

-----

Timeline:

- Report Deadline 19th January -- All
- Programming deadline 17th January
- Implement everything in proposal 14th [need 1 day to train probably?]

- Implement feature selection algorithm others 13th -- Others
- Implement feature selection algorithm PCA 11th -- Kostas
- Implement Loss Function 8th -- David
- Implement Peripheral 8th -- David
- Implement Baseline NN (from Kaggle) 10th [should work on random input data] -- All

-----
Some things to note:

- Make sure to use python 3.


- To install any packages, use the following commands (inside a virtualenvironment!):
 
 ```
 pip install -r requirements.txt
 ```
 
 (Of course one can also use anaconda, which is a better decision - I'm just used to pip and the deadline is soon).
 
 Whenever you install a new package, please append the `requirements.txt` 
 file with the modulename, as well as the most specific module version.

- Here is the kaggle link whose dataset we use.
```
https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories
```
Download the data into the `data` folder inside the root directory of this project.
 
-----


The following are the current todo's.
You can declare new todos in 1. the repo issues, or 2. here in the readme.

Peripheral:
- [ ] Data loader and preprocessor
- [ ] Batch loader for training set
- [ ] Batch loader for validation set
- [ ] Batch loader for test set
- [ ] Automatic deployment to Leonhard, and automated download of saved weights
- [ ] Setup environment for feature algorithms, debug tools, prediction algorithms, and other features

Debug tools:
- [ ] Tensorboard coupled to training loss and validation loss
- [ ] Predictor on validation / test dataset

Feature selection algorithms:
- [ ] Earth quake significance prediction algorithm
- [ ] "tsfresh": https://tsfresh.readthedocs.io/en/latest/
- [ ] PCA
- [ ] "Shap value": https://github.com/slundberg/shap
- [ ] "Word Embeddings" for Ticker symbols

Prediction algorithms:
- [ ] LSTM (Baseline from Kaggle)

Other features:
- [ ] Loss function



---- 
Q & A:

- Do we do the training-test split between stocks, or dates? (or somehow both?)
- X is of shape (n_stocks, n_dates, n_features). 
Should Y be of shape (n_stocks, n_dates, 6) (6, because high, low, open, close, volume, medium)? 
Or what exactly do we want to predict?


----
Backlog (ignore for now)

- Please create a file called ".env" in the root directory of this project,
and write to the file the following item:
 
 ```
LEONHARD_UNAME="yedavid"
 ```
 (i..e instead of yedavid, use your username)
 
-- implement dummy features