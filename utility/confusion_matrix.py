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

from test_validation import loadModels, evaluateFrame

global y_true
y_true = []

global x_pred
x_pred = []

val_preprocess, device, model_emotion_class, model_Valenza, model_Arousal = loadModels()


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

def getPredEmotionClass():
    with open('csv/AllGender/val_set-csv_only_local_path.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            messaggio_di_ritorno = evaluateFrame(model_emotion_class, model_Valenza, model_Arousal, row[0], val_preprocess, device)
            try:
                if (messaggio_di_ritorno["emotion_class"] == "Neutrale"):
                    x_pred.append(0)
                if (messaggio_di_ritorno["emotion_class"] == "Felicita'"):
                    x_pred.append(1)
                if (messaggio_di_ritorno["emotion_class"] == "Tristezza"):
                    x_pred.append(2)
                if (messaggio_di_ritorno["emotion_class"] == "Sorpresa"):
                    x_pred.append(3)
                if (messaggio_di_ritorno["emotion_class"] == "Paura"):
                    x_pred.append(4)
                if (messaggio_di_ritorno["emotion_class"] == "Disgusto"):
                    x_pred.append(5)
                if (messaggio_di_ritorno["emotion_class"] == "Rabbia"):
                    x_pred.append(6)
                if (messaggio_di_ritorno["emotion_class"] == "Disprezzo"):
                    x_pred.append(7)
            except:
                x_pred.append(8)

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
        8: "N/D",
    }

labels_list = []
for i in range(len(label_mapping)):
    labels_list.append(label_mapping[i])

getTrueEmotionClass()
getPredEmotionClass()
show_confusion_matrix(y_true, x_pred, labels_list, path)
print(get_classification_report(y_true, x_pred, labels_list))

