
from deepface import DeepFace
from pathlib import Path
import pandas as pd
import numpy as np
import csv

# Constanti
CONST_N = 420000    #val = 5500 , train = 420000
GENDER_TARGET = "Woman"     #Woman o Man
NOME_CSV = "woman_train"
DIR_SET = "train_set"   #val_set o train_set

if __name__ == '__main__':

    header = ['path', 'emotion_class', 'Valenza', 'Arousal']

    with open(NOME_CSV+'.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        i = 0  # Defining index of array

        # Ciclo tutti i file di annotation per recuperare i valori della Valenza
        for i in range(0, CONST_N):

            try:
                #per verificare l'esistenza dell'i-esimo file
                np.load('C:/Dataset AffectNet/'+DIR_SET+'/annotations/' + str(i) + "_exp.npy")

            except:
                pass
            else:

                obj = DeepFace.analyze(img_path="C:/Dataset AffectNet/"+DIR_SET+"/images/" + str(i) + ".jpg",
                                       actions=['gender'],
                                       enforce_detection=False)

                if obj["gender"] == GENDER_TARGET:
                    print("Elaboro immagine: "+str(i)+".jpg")
                    img_path="/content/train_set_images/" + str(i) + ".jpg"
                    emotion_class=np.load('C:/Dataset AffectNet/'+DIR_SET+'/annotations/' + str(i) + "_exp.npy")
                    valenza=np.load('C:/Dataset AffectNet/'+DIR_SET+'/annotations/' + str(i) + "_val.npy")
                    arousal=np.load('C:/Dataset AffectNet/'+DIR_SET+'/annotations/' + str(i) + "_aro.npy")

                    data= [img_path, emotion_class, valenza, arousal]

                    writer.writerow(data)
