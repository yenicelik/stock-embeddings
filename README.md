# deeplearning - two sigma challenge


-----

Some things to note:
- Please create a file called ".env" in the root directory of this project,
and write to the file the following item:
 
 ```
LEONHARD_UNAME=yedavid 
 ```
 (i..e instead of yedavid, use your username)
 
 To install any packages, use the following commands (inside a virtualenvironment!):
 
 ```
 pip install -r requirements.txt
 ```
 
 (Of course one can also use anaconda, which is a better deciison - I'm just used to pip and the deadline is soon).
 
 Whenever you install a new package, please append the `requirements.txt` 
 file with the modulename, as well as the most specific module version.
 
-----


The following are the current todo's.
You can declare new todos in 1. the repo issues, or 2. here in the readme.

Peripheral:
- [ ]  Data loader and preprocessor
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

