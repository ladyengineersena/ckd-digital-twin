import joblib
from lightgbm import LGBMClassifier

def train_lgbm(X, y, params=None):
    if params is None:
        params = {"n_estimators":400, "learning_rate":0.05, "max_depth":6, "random_state":42}
    model = LGBMClassifier(**params)
    model.fit(X,y)
    return model

def predict_proba(model, X):
    return model.predict_proba(X)[:,1]

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

