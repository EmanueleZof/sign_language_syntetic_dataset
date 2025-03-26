class Image:
    def __init__(self,
                 loader = "coco_wholebody",
                 ):
        self.loader = loader

    def _coco_wholebody_loader(self):
        print("Test")

    def extract_images(self):
        if (self.loader == "coco_wholebody") :
            self._coco_wholebody_loader()
        else :
            print("Nessun loader specificato")