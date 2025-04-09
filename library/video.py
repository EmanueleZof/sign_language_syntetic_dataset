import os
import random
import pandas as pd

class Video:
    """
    Classe per la gestione di dataset di video, inclusi i metodi per il caricamento dei dati, 
    la selezione casuale di video e l'estrazione di video per una determinata classe.

    Attributi:
        loader (str): Il tipo di loader da utilizzare per caricare i dati ('autsl' o 'demo').
        path (str): Il percorso principale dove si trovano i dati.
        data_type (str): Tipo di dati (utilizzato solo per 'autsl').
        classes (list): Lista delle classi di interesse.
        dataset (list): Lista contenente i dataset delle classi con i relativi video.
    """
    def __init__(self,
                loader = "default",
                path = "",
                data_type = "",
                classes = [],
                ):
      self.loader = loader
      self.path = path
      self.data_type = data_type
      self.classes = classes
      self.dataset = []

      self.main()

    def _autsl_loader(self, main_folder_path, data_type, classes):
        """
        Carica i video e le etichette dal dataset AUTSL per le classi specificate.

        Args:
            main_folder_path (str): Percorso principale del dataset.
            data_type (str): Tipo di dati ('train', 'test', etc.).
            classes (list): Lista delle classi di interesse.
        
        Aggiorna:
            self.dataset: Aggiunge i video e le informazioni per ciascuna classe.
        """
        classes_path = f"{main_folder_path}/classes/classes.csv"
        data_path = f"{main_folder_path}/{data_type}/{data_type}_labels.csv"

        df_classes = pd.read_csv(classes_path)
        df_test_labels = pd.read_csv(data_path, header=None)

        for class_name in classes:
            hi_class_id = df_classes.loc[df_classes["EN"] == class_name].iloc[0]["ClassId"]
            hi_videos = df_test_labels.loc[df_test_labels[1] == hi_class_id][0].tolist()
            hi_videos =[f"{main_folder_path}/{data_type}/color/{path}_color.mp4" for path in hi_videos]

            self.dataset.append({
                "class_name": class_name,
                "class_id": hi_class_id,
                "video_list": hi_videos
            })

    def _demo_loader(self, main_folder_path):
      """
      Carica i video da una struttura di directory per il caso di uso demo.

      Args:
          main_folder_path (str): Percorso principale delle directory demo.
      
      Aggiorna:
          self.dataset: Aggiunge i video per ciascuna classe trovata nelle directory.
      """
      for subdir, dirs, files in os.walk(main_folder_path):
        if (subdir != main_folder_path):
          self.dataset.append({
              "class_name": subdir.replace(f"{main_folder_path}/", ""),
              "class_id": 0,
              "video_list": [f"{subdir}/{file_name}" for file_name in files]
          })

    def main(self):
      """
      Esegue il caricamento del dataset in base al tipo di loader specificato (autsl o demo).

      Aggiorna:
          self.dataset: Popola il dataset con i video e le informazioni.
      """
      if (self.loader == "autsl" and self.path != "") :
        self._autsl_loader(self.path, self.data_type, self.classes)
      elif (self.loader == "demo" and self.path != ""):
        self._demo_loader(self.path)
      else :
        print("Nessun loader o path specificato")

    def get_dataset(self):
      """
      Restituisce il dataset caricato.

      Returns:
          list: Il dataset contenente le classi e i video associati.
      """
      return self.dataset

    def show_random_video(self, class_name):
      """
      Mostra un video casuale per una determinata classe.

      Args:
          class_name (str): Il nome della classe per la quale si vuole selezionare un video.

      Returns:
          str: Il percorso del video selezionato casualmente.
      """
      selected_class = [e for e in self.dataset if e["class_name"] == class_name]
      return random.choice(selected_class[0]["video_list"])

    def extract_videos(self, class_name):
      """
      Estrae tutti i video associati a una determinata classe.

      Args:
          class_name (str): Il nome della classe per la quale estrarre i video.

      Returns:
          str: Una stringa con i percorsi di tutti i video della classe.
      """
      selected_class = [e for e in self.dataset if e["class_name"] == class_name]
      return " ".join(selected_class[0]["video_list"])
