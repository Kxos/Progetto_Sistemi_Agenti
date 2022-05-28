import argparse
import json
import logging
import os
import platform

import nni
import torch
from joblib._multiprocessing_helpers import mp

if platform.system() == "Linux":
    import shutil

import warnings

from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm

# from config import args, return_args
# from dataloader.opera7 import Opera7
# from dataloader.demosemovogender import DemosEmovoGender
from dataloader.emovo import Emovo
from dataloader.demos import Demos
from dataloader.demosemovo import DemosEmovo
from dataloader.affectnet_sigle import AffectNet
from models.bam.vggface2_bam import VGGFace2BAM
from models.cbam.vggface2_cbam import VGGFace2CBAM
from models.resnet50.vggface2 import VGGFace2
from models.se.vggface2_se import VGGFace2SE
from utility.checkpoint import load_model
from utility.utility import setup_seed
from utility.confusion_matrix import show_confusion_matrix, get_classification_report
from PIL import Image
import urllib.request
from io import BytesIO
import cv2

warnings.filterwarnings('ignore')

logger = logging.getLogger('mnist_AutoML')


def loadModels():
    # Args for debugging through IDE
    # args['dataset'] = 'demos'                                                          # Replace with you own dataset
    # args['gender'] = 'all'                                                             # Gender for the training dataset
    # args['validation'] = 'emotion'                                                     # Choose on what to do validation ("emotion" / "gender")
    # args['uses_drive'] = False                                                         # Whether to choose Drive to save results
    # args['loadModel'] = 'result/SysAg2022/best_model_gruppo_N_all.pt'                  # Load model for validation
    # args['attention'] = 'bam'                                                          # Choose your model type (Bam) / (CBam)
    # args['batch_size'] = 64                                                            # Batch size for training

    parser = argparse.ArgumentParser(description="Configuration validation phase")
    parser.add_argument("-a", "--attention", type=str, default="no", choices=["no", "se", "bam", "cbam"],
                        help='Chose the attention module')
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
    parser.add_argument("-d", "--dataset", type=str, default="affectnet", choices=["affectnet"],
                        help='Chose the dataset')
    parser.add_argument("-s", "--stats", type=str, default="imagenet", choices=["no", "imagenet"],
                        help='Chose the mean and standard deviation')

    # TODO - Verificare se --target è inteso come --validation
    parser.add_argument("-ta", "--target", type=str, default="Arousal",
                        choices=["emotion_class", "Valenza", "Arousal"], help='Select the type of target')
    parser.add_argument("-ge", "--gender", type=str, default="all",
                        choices=["all", "male", "female"], help='Gender for the validation dataset')

    args = parser.parse_args()

    # setup_seed(args.seed)

    print("Starting validation with the following configuration:")
    print("Attention module: {}".format(args.attention))
    print("Batch size: {}".format(args.batch_size))
    print("Dataset: {}".format(args.dataset))
    print("Gender: {}".format(args.gender))
    print("Validation: {}".format(args.target))
    print("Stats: {}".format(args.stats))

    # TODO - NON DOVREBBERO SERVIRE
    # print("Checkpoint model: {}".format(args['checkpoint']))
    # print("Uses Drive: {}".format(args["uses_drive"]))
    # print("With Augmentation: {}".format(args["withAugmentation"]))
    # print("Workers: {}".format(args['workers']))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("===================================================")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
        print("===================================================")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('Cuda not available. Using CPU.')

    if args.stats == "imagenet":
        # imagenet
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ])
    else:
        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    validation_classes = {
        # "opera7": Opera7,
        "emovo": Emovo,
        "demos": Demos,
        "demosemovo": DemosEmovo,
        "affectnet": AffectNet,
        # "demosemovogender": DemosEmovoGender
    }

    # dataset = args.dataset if args.dataset in validation_classes.keys() else "affectnet"
    # val_data = validation_classes[dataset](split="val", transform=val_preprocess)

    # Carica il modello di emotion_class
    classes = 8

    if args.attention == "no":
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    elif args.attention == "se":
        model = VGGFace2SE(classes=classes).to(device)
    elif args.attention == "bam":
        model = VGGFace2BAM(classes=classes).to(device)
    elif args.attention == "cbam":
        model = VGGFace2CBAM(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=False, classes=classes).to(device)

    pathModel = f"Modelli trainati/emotion_class/result/{args.attention}/best_model.pt"
    model_emotion_class = load_model(pathModel, model, device)

    # Carica il modello di Valenza
    classes = 1

    if args.attention == "no":
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    elif args.attention == "se":
        model = VGGFace2SE(classes=classes).to(device)
    elif args.attention == "bam":
        model = VGGFace2BAM(classes=classes).to(device)
    elif args.attention == "cbam":
        model = VGGFace2CBAM(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=False, classes=classes).to(device)

    pathModel = f"Modelli trainati/Valenza/result/{args.attention}/best_model.pt"
    model_Valenza = load_model(pathModel, model, device)

    # Carica il modello di Arousal
    classes = 1

    if args.attention == "no":
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    elif args.attention == "se":
        model = VGGFace2SE(classes=classes).to(device)
    elif args.attention == "bam":
        model = VGGFace2BAM(classes=classes).to(device)
    elif args.attention == "cbam":
        model = VGGFace2CBAM(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=False, classes=classes).to(device)

    pathModel = f"Modelli trainati/Arousal/result/{args.attention}/best_model.pt"
    model_Arousal = load_model(pathModel, model, device)
    # print("Custom model loaded successfully")
    print("-------------------------------------------------------")

    return val_preprocess, device, model_emotion_class, model_Valenza, model_Arousal


# Valida i 3 modelli restituendo il messaggio_di_ritorno
def evaluateFrame(model_emotion_class, model_Valenza, model_Arousal, imageURL, val_preprocess, device):
    messaggio_di_ritorno = {
        "emotion_class": None,
        "Valenza": None,
        "Arousal": None,
    }

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

    # validate the models
    model_emotion_class.eval()
    model_Valenza.eval()
    model_Arousal.eval()

    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=mp.cpu_count())

    print("\nStarting validation...")
    urllib.request.urlretrieve(imageURL, "image.jpg")

    # Crop the image to obtain the face
    images = cv2.imread("image.jpg")
    if images is None:
        return None
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    images = cv2.resize(images[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_LANCZOS4)

    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    images = Image.fromarray(images)
    images = val_preprocess(images)
    images = images.unsqueeze(0)
    images = images.to(device)
    # labels = batch["label"].to(device)

    with torch.no_grad():
        val_outputs_emo_class = model_emotion_class(images)
        val_outputs_Valenza = model_Valenza(images)
        val_outputs_Arousal = model_Arousal(images)
        # print(target)

        # Valori Valenza ed Arousal
        messaggio_di_ritorno["Valenza"] = float(val_outputs_Valenza.data.cpu().numpy()[0][0])
        messaggio_di_ritorno["Arousal"] = float(val_outputs_Arousal.data.cpu().numpy()[0][0])
        # output: Valenza od Arousal a seconda del Target impostato
        # print(target, ": ", val_outputs.data.cpu().numpy()[0][0])

        messaggio_di_ritorno["emotion_class"] = label_mapping[val_outputs_emo_class.data.cpu().numpy().argmax()]
        # output: emotion_class
        # print(val_outputs.data.cpu().numpy().argmax())

        # _, val_preds = torch.max(val_outputs, 1)
        # val_correct += torch.sum(val_preds == labels.data)

        # y_true.extend(labels.detach().cpu().numpy().tolist())
        # y_pred.extend(val_preds.detach().cpu().numpy().tolist())

    # if target == "emotion_class":
    #     for i in range(len(label_mapping)):
    #         labels_list.append(label_mapping[i])

    # print("Num correct: {}".format(val_correct))
    # print("Num samples: {}".format(len(val_data)))

    # val_acc = (val_correct.double() / len(val_data)) * 100

    delimiter = "\n===================================================================================\n"

    # write = F'Accuracy of the network on the test images: {val_preds:.3f}%'
    # print(F'\n{write}')

    # classificationReport = get_classification_report(y_true, y_pred, labels_list)
    # print(classificationReport)

    # show_confusion_matrix(y_true, y_pred, labels_list,
    # "result/{}/{}/{}/".format(args.dataset, args.attention, args.gender))

    print("===================================Validation Finished===================================")

    # TODO - Effettuare l'invio dei valori di Arousal, Valenza, emotion_class al WebService
    messaggio_di_ritorno_json = json.dumps(messaggio_di_ritorno)
    return messaggio_di_ritorno_json
