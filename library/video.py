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
    self.dataset = {}

    self.main()

    def _autsl_loader(self, main_folder_path, data_type, classes):
        classes_path = f"{main_folder_path}/classes/classes.csv"
        data_path = f"{main_folder_path}/{data_type}/{data_type}_labels.csv"

        df_classes = pd.read_csv(classes_path)
        df_test_labels = pd.read_csv(data_path, header=None)

        obj = []

        for dataset_class in classes:
            hi_class_id = df_classes.loc[df_classes["EN"] == dataset_class].iloc[0]["ClassId"]
            hi_videos = df_test_labels.loc[df_test_labels[1] == hi_class_id][0].tolist()
            obj.append({
                "class_id": hi_class_id,
                "video_list": hi_videos
            })
        
        return obj


    def main(self):
      if (self.loader == "autsl" and self.path != "") :
        url_list = self._autsl_loader(self.path, self.data_type, self.classes)
      else :
        print("Nessun loader o path specificato")