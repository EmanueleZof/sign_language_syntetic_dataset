import json
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

class Image:
    def __init__(self,
                 loader = "default",
                 path = "",
                 ):
        self.loader = loader
        self.path = path
        self.dataset = {}

        self.main()
    
    def _create_json(self, data, label="default"):
      base = Path(".")
      jsonpath = base / (label + ".json")
      jsonpath.write_text(json.dumps(data))

    def _show_image(self, image_url, bbox):
      img = io.imread(image_url)
      rect = patches.Rectangle((bbox[0], bbox[1]),
                                bbox[2],
                                bbox[3], 
                                linewidth=1, 
                                edgecolor='r', 
                                facecolor='none')
      fig, box = plt.subplots()
      box.imshow(img)
      box.add_patch(rect)
      box.axis('off')
      plt.show()

    def _coco_wholebody_loader(self, annotation_path):
        url_list = []
        with open(annotation_path) as json_data:
          data = json.load(json_data)
          df_img = pd.DataFrame(data["images"])
          df_ann = pd.DataFrame(data["annotations"])

        person_annotations = df_ann.loc[(df_ann['category_id'] == 1) &
                           (df_ann['iscrowd'] == 0) &
                           (df_ann['face_valid'] == True) &
                           (df_ann['lefthand_valid'] == True) &
                           (df_ann['righthand_valid'] == True) &
                           (df_ann['foot_valid'] == True) &
                           (df_ann['area'] > 10000)]

        for index, row in person_annotations.iterrows():
          person_image = df_img.loc[df_img['id'] == row["image_id"]]
          url_list.append(person_image.coco_url.values[0])
        
        self.dataset["images"] = df_img
        self.dataset["annotations"] = person_annotations
        self.dataset["list"] = url_list

    def main(self):
      if (self.loader == "coco_wholebody" and self.path != "") :
        url_list = self._coco_wholebody_loader(self.path)
      else :
        print("Nessun loader o path specificato")

    def extract_images(self):
      if (self.loader == "coco_wholebody") :
        self._create_json(self.dataset["list"], self.loader)
      else :
        print("Il Dataset è vuoto")

    def get_dataset(self):
      return self.dataset

    def show_random_image(self, show_bbox=False):
      if (self.loader == "coco_wholebody") :
        bunding_box = [0,0,0,0]
        
        tot_person_annotations = len(self.dataset["annotations"])
        random_person_annotation = self.dataset["annotations"].iloc[np.random.randint(0,tot_person_annotations)]
        random_person_image = self.dataset["images"].loc[self.dataset["images"]['id'] == random_person_annotation.image_id]
        
        if (show_bbox):
          bunding_box = [random_person_annotation.bbox[0],
                          random_person_annotation.bbox[1],
                          random_person_annotation.bbox[2],
                          random_person_annotation.bbox[3]]

        self._show_image(random_person_image.coco_url.values[0], bunding_box)
      else :
        print("Il Dataset è vuoto")
      
