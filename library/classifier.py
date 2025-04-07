import pickle
import pandas as pd

import library.utils as Utils

from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay
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

    def model_accuracy(self, prediction, y_test):
        return accuracy_score(y_test, prediction)

    def model_confusion_matrices(self, prediction, y_test, dataset):
        confusion_matrices = multilabel_confusion_matrix(y_test, prediction)

        for index, confusion_matrix in enumerate(confusion_matrices):
            disp = ConfusionMatrixDisplay(confusion_matrix)
            disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
            disp.ax_.set_title(dataset[index]["class_name"])
            plt.show()

    def model_export(self, model):
        _, model_name = model.named_steps.keys()

        with open(f"{Utils.OUTPUT_DIR}{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    def model_load(self, pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            return pickle.load(f)