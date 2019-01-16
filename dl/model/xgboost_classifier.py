from xgboost import XGBClassifier

class XGBoostClassifier:

    def __init__(self):
        self.model = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)

    def transform(self, X):
        pass

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
