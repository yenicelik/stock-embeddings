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
https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/version/3
```
Download the data into the `data` folder inside the root directory of this project.
 
 - Please create a file called ".env" in the root directory of this project,
and write to the file the following item:
 
 ```
LEONHARD_UNAME="yedavid"

DATAPATH="/Users/david/deeplearning/data/Data/Stocks/"

DATAPATH_PROCESSED="/Users/david/deeplearning/data/processed/all.csv"
DATAPATH_PROCESSED_DEV="/Users/david/deeplearning/data/processed/dev.csv"

DATA_PICKLE="/Users/david/deeplearning/data/processed/pickle.pkl"
DATA_PICKLE_DEV="/Users/david/deeplearning/data/processed/pickle_dev.pkl"

MODEL_SAVEPATH_BASEPATH="/Users/david/deeplearning/data/"


 ```
 (i..e instead of yedavid, use your username - the same goes for paths)

-----


The following are the current todo's.
You can declare new todos in 1. the repo issues, or 2. here in the readme.

Peripheral:
- [x] Data loader and preprocessor
- [x] Batch loader for training set
- [x] Batch loader for validation set
- [x] Batch loader for test set
- [x] Automatic deployment to Leonhard, and automated download of saved weights
- [x] Setup environment for feature algorithms, debug tools, prediction algorithms, and other features

Debug tools:
- [] Tensorboard coupled to training loss and validation loss
- [] Predictor on validation / test dataset

Feature selection algorithms:
- [ ] Earth quake significance prediction algorithm
- [ ] "tsfresh": https://tsfresh.readthedocs.io/en/latest/
- [ ] PCA
- [ ] "Shap value": https://github.com/slundberg/shap
- [x] "Word Embeddings" for Ticker symbols

Prediction algorithms:
- [ ] LSTM (Baseline from Kaggle)

Other features:
- [x] Loss function



---- 
Q & A:

- Do we do the training-test split between stocks, or dates? (or somehow both?)
- X is of shape (n_stocks, n_dates, n_features). 
Should Y be of shape (n_stocks, n_dates, 6) (6, because high, low, open, close, volume, medium)? 
Or what exactly do we want to predict?


----
Backlog (ignore for now)

 
-- implement dummy features
-- Implement autoregression visualization (checking if from one timestep to another it is symmetric or not etc.)





#### 

# TODO: Also include the loss (graphs for each one possibly)

Future work:

- [] embedding can be extended to non-linear, deep latent space (by using a sub-network)


---- Some more todo's

- [] Der Teil zu Feature Engineering ist jetzt fehl am Platz und muss weg, dafuer sollten wir den Teil zu Embeddings ausbauen

- [] Und bei dem Stock Experiment nur eine Accuracy zahl zu zeigen, idk -> Do more analysis with prediction ROC or so, or predicision/recall matrix

- [] Der Punkt ist, dass Leute kommen koennen und sagen: Wie viele Returns in eurem Sample waren positiv (in der Periode sind es grob geschaetzt signifikant mehr als 50%), was macht ein Indikator mit 54% Accuracy dann fuer einen Unterschied vs salopp gesagt einfach drauf los zu kaufen

- [] Entweder mehr Sachen isV MAE fuer jeweils positive/negative forecasts jeweils zeigen, oder direkt ein Investmentexperiment machen

- [x] Die ganzen Grafiken und das Format muss nochmal sauber ueberarbeitet werden
