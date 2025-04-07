import os
import random
import pandas as pd

class Video:
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
      for subdir, dirs, files in os.walk(main_folder_path):
        if (subdir != main_folder_path):
          self.dataset.append({
              "class_name": subdir.replace(main_folder_path, ""),
              "class_id": 0,
              "video_list": [f"{subdir}/{file_name}" for file_name in files]
          })

    def main(self):
      if (self.loader == "autsl" and self.path != "") :
        self._autsl_loader(self.path, self.data_type, self.classes)
      else if (self.loader == "demo" and self.path != ""):
        self._demo_loader(self.path)
      else :
        print("Nessun loader o path specificato")

    def get_dataset(self):
      return self.dataset

    def show_random_video(self, class_name):
      selected_class = [e for e in self.dataset if e["class_name"] == class_name]
      return random.choice(selected_class[0]["video_list"])

    def extract_videos(self, class_name):
      selected_class = [e for e in self.dataset if e["class_name"] == class_name]
      return " ".join(selected_class[0]["video_list"])
