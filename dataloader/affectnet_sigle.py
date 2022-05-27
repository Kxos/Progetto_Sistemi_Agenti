import numpy as np
import torch.utils.data as data
import pandas as pd
from PIL import Image


class AffectNet(data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split

        # TODO: implementare il caricamento del CSV e settare la cartella dove sono presenti i file image in self.images
        if self.split == "train_single":
            pass
            # self.data = pd.read_csv("csv/train_set-csv_completo.CSV", sep=";")
            # self.images = "/content/train_set_images"
            # print("Train Set: ")
            # print(self.data.columns.tolist())
            # print(self.data)
        elif self.split == "val_single":
            self.data = None
            self.images = "C:/Dataset AffectNet/val_set/images/0.jpg"
            print("Image: ")
            print(self.images)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: leggere il percorso del file img dal CSV
        img_name = "C:/Dataset AffectNet/val_set/images/0.jpg"

        # TODO: caricare l'immagine con PIL
        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img}


if __name__ == "__main__":
    split = "train"
    affectnet_train = AffectNet(split=split)
    print("AffectNet {} set loaded".format(split))
    print("{} samples".format(len(affectnet_train)))

    for i in range(1):
        print(affectnet_train[i]["label"])
        affectnet_train[i]["image"].show()