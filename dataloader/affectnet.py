import numpy as np
import torch.utils.data as data
import pandas as pd
from PIL import Image


class AffectNet(data.Dataset):

    def __init__(self, split='train', transform=None, target='emotion_class'):
        self.transform = transform
        self.split = split
        self.target = target

        # TODO: implementare il caricamento del CSV e settare la cartella dove sono presenti i file image in self.images
        if self.split == "train":
            self.data = pd.read_csv("csv/train_set-csv_completo.CSV", sep=";")
            self.images = "/content/train_set_images"
            print("Train Set: ")
            print(self.data.columns.tolist())
            print(self.data)
        elif self.split == "val":
            self.data = pd.read_csv("csv/val_set-csv_completo.csv", sep=";")
            self.images = "/content/val_set_images"
            print("Validation Set: ")
            print(self.data.columns.tolist())
            print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: leggere la label associata al file img nel CSV
        label = self.data.loc[idx, self.target]

        # Se target è Valenza od Arousal NON eseguire la conversione int(label)
        if self.target == 'emotion_class':
            label = int(label)
        else:
          label = np.float32(label)

        # TODO: leggere il percorso del file img dal CSV
        img_name = self.data.loc[idx, "image_path"]

        # TODO: caricare l'immagine con PIL
        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label}


if __name__ == "__main__":
    split = "train"
    affectnet_train = AffectNet(split=split)
    print("AffectNet {} set loaded".format(split))
    print("{} samples".format(len(affectnet_train)))

    for i in range(1):
        print(affectnet_train[i]["label"])
        affectnet_train[i]["image"].show()
