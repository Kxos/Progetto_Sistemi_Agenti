import csv

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import metrics
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns


global y_true
y_true = []

global x_pred
x_pred = []

def show_confusion_matrix(y_true, x_pred, labels, path):
    matrix = metrics.confusion_matrix(y_true, x_pred)
    plt.figure(figsize=(7, 7))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(path + "confusion_matrix.png")
    #plt.show()


def get_classification_report(y_true, x_pred, labels):
    return classification_report(y_true, x_pred, target_names=labels)

def getTrueEmotionClass():
    with open('csv/AllGender/val_set-csv_only_emotion_class.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            y_true.append(int(row[0]))
            x_pred.append(int(row[0]))

def getPredEmotionClass():
    with open('csv/AllGender/val_set-csv_only_local_path.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:

            #images = cv2.imread(row[0])
            print(row[0])
            # if images is None:
            #     return None
            # gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # if len(faces) == 0:
            #     return None
            # x, y, w, h = faces[0]
            # images = cv2.resize(images[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_LANCZOS4)
            #
            # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            # images = Image.fromarray(images)
            # images = val_preprocess(images)
            # images = images.unsqueeze(0)
            # images = images.to(device)
            # # labels = batch["label"].to(device)
            #
            # with torch.no_grad():
            #     val_outputs_emo_class = model_emotion_class(images)
            #
            #     #messaggio_di_ritorno["emotion_class"] = label_mapping[val_outputs_emo_class.data.cpu().numpy().argmax()]
            #     # output: emotion_class





# ================ USAGE ================ #


path = "C:/result/"

label_mapping = {
        0: "Neutrale",
        1: "Felicita'",
        2: "Tristezza",
        3: "Sorpresa",
        4: "Paura",
        5: "Disgusto",
        6: "Rabbia",
        7: "Disprezzo",
    }

labels_list = []
for i in range(len(label_mapping)):
    labels_list.append(label_mapping[i])

getTrueEmotionClass()
getPredEmotionClass()
show_confusion_matrix(y_true, x_pred, labels_list, path)
print(get_classification_report(y_true, x_pred, labels_list))

