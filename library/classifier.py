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
    """
    Classe per la costruzione del modello di classificazione supervisionata.

    Non è stato creato un wrapper per la funzione .prediction(), ma è stata utilizzata direttamente questa funzione nel codice utilizzato sui vari Google Colab.

    Metodi principali:
    - create_set(csv_path): carica un dataset da file CSV, mescola le righe e restituisce le variabili indipendenti (X) e la variabile target (y).
    - model_train(X_train, y_train): addestra diversi modelli di classificazione (Logistic Regression, Ridge Classifier, Random Forest, Gradient Boosting)
      utilizzando pipeline con standardizzazione dei dati, e restituisce i modelli addestrati.
    - model_evaluate(fit_models, X_test, y_test): stampa l'accuratezza di ciascun modello sul set di test.
    - model_accuracy(prediction, y_test): calcola e restituisce l'accuratezza di un modello dato un set di predizioni e i target reali.
    - model_confusion_matrices(prediction, y_test, dataset): genera e visualizza le matrici di confusione per ciascuna classe del dataset.
    - model_export(model): salva un modello addestrato su disco in formato pickle, usando il nome del classificatore come nome file.
    - model_load(pkl_file_path): carica un modello precedentemente salvato da un file pickle.
    """

    def create_set(self, csv_path):
        """
        Carica un dataset da un file CSV, mescola casualmente le righe
        e restituisce le feature (X) e le etichette (y).

        Parametri:
        - csv_path (str): percorso del file CSV contenente il dataset.

        Ritorna:
        - X (DataFrame): dati di input senza la colonna target.
        - y (Series): colonna delle etichette di classificazione.
        """
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1).reset_index(drop=True)

        X = df.drop("class", axis=1)
        y = df["class"]

        return X, y

    def model_train(self, X_train, y_train):
        """
        Addestra diversi modelli di classificazione.

        Parametri:
        - X_train (DataFrame): dati di addestramento.
        - y_train (Series): etichette di addestramento.

        Ritorna:
        - fit_models (dict): dizionario contenente i modelli addestrati con le rispettive chiavi ('lr', 'rc', 'rf', 'gb').
        """
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
        """
        Valuta ciascun modello fornito calcolandone l'accuratezza sul set di test.

        Parametri:
        - fit_models (dict): dizionario dei modelli addestrati.
        - X_test (DataFrame): dati di test.
        - y_test (Series): etichette reali del set di test.
        """
        for algo, model in fit_models.items():
            prediction = model.predict(X_test)
            print(algo, accuracy_score(y_test, prediction))

    def model_accuracy(self, prediction, y_test):
        """
        Calcola l'accuratezza di un modello a partire dalle predizioni e dai valori reali.

        Parametri:
        - prediction (array-like): etichette predette dal modello.
        - y_test (array-like): etichette reali.

        Ritorna:
        - accuracy (float): accuratezza del modello.
        """
        return accuracy_score(y_test, prediction)

    def model_confusion_matrices(self, prediction, y_test, dataset):
        """
        Genera e visualizza le matrici di confusione per ciascuna classe.

        Parametri:
        - prediction (array-like): etichette predette.
        - y_test (array-like): etichette reali.
        - dataset (list of dict): lista contenente le informazioni sulle classi, inclusi i nomi per il titolo dei grafici.
        """
        confusion_matrices = multilabel_confusion_matrix(y_test, prediction)

        for index, confusion_matrix in enumerate(confusion_matrices):
            disp = ConfusionMatrixDisplay(confusion_matrix)
            disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
            disp.ax_.set_title(dataset[index]["class_name"])
            plt.show()

    def model_export(self, model):
        """
        Esporta un modello addestrato salvandolo in un file pickle.

        Parametri:
        - model (Pipeline): modello addestrato (pipeline sklearn).

        Salva:
        - File .pkl nella directory definita in Utils.OUTPUT_DIR, con nome basato sul classificatore usato.
        """
        _, model_name = model.named_steps.keys()

        with open(f"{Utils.OUTPUT_DIR}{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    def model_load(self, pkl_file_path):
        """
        Carica un modello precedentemente salvato in formato pickle.

        Parametri:
        - pkl_file_path (str): percorso del file pickle del modello salvato.

        Ritorna:
        - model (Pipeline): modello caricato.
        """
        with open(pkl_file_path, "rb") as f:
            return pickle.load(f)