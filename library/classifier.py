import pickle
import pandas as pd

import library.utils as Utils

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class Classifier:
    def create_set(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1).reset_index(drop=True)

        X = df.drop("class", axis=1)
        y = df["class"]

        return X, y

    def model_train(self, X_train, y_train):
        fit_models = {}
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }

        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algo] = model

        return fit_models

    def model_evaluate(self, fit_models, X_test, y_test):
        for algo, model in fit_models.items():
            prediction = model.predict(X_test)
            print(algo, accuracy_score(y_test, prediction))

    def model_export(self, model):
        _, model_name = model.named_steps.keys()

        with open(f"{Utils.OUTPUT_DIR}{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    def model_load(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            return pickle.load(f)