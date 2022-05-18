# https://github.com/ox-vgg/vgg_face2

import torch
import torch.nn as nn
import models.resnet50.model.resnet50_ft_dims_2048 as model


class VGGFace2(nn.Module):

    def __init__(self, pretrained=True, classes=8):
        super().__init__()
        self.classes = classes

        if pretrained is True:
            # load weights
            # TODO: cambiare con i pesi di imagenet
            self.vggface2 = model.resnet50_ft("./resnet50/model/resnet50_ft_dims_2048.pth")
        else:
            self.vggface2 = model.resnet50_ft()

        # replace the top layer for classification
        # self.resnet50.pool5_7x7_s1 = nn.AdaptiveAvgPool2d((1, 1))
        self.vggface2.classifier = nn.Linear(2048, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        conv1_7x7_s2 = self.vggface2.conv1_7x7_s2(data)
        conv1_7x7_s2_bn = self.vggface2.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.vggface2.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.vggface2.pool1_3x3_s2(conv1_7x7_s2_bnxx)
        conv2_1_1x1_reduce = self.vggface2.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.vggface2.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.vggface2.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.vggface2.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.vggface2.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.vggface2.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.vggface2.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.vggface2.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        conv2_1_1x1_proj = self.vggface2.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.vggface2.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1x = self.vggface2.conv2_1_relu(conv2_1)
        conv2_2_1x1_reduce = self.vggface2.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.vggface2.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.vggface2.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.vggface2.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.vggface2.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.vggface2.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.vggface2.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.vggface2.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2x = self.vggface2.conv2_2_relu(conv2_2)
        conv2_3_1x1_reduce = self.vggface2.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.vggface2.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.vggface2.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.vggface2.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.vggface2.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.vggface2.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.vggface2.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.vggface2.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3x = self.vggface2.conv2_3_relu(conv2_3)
        conv3_1_1x1_reduce = self.vggface2.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.vggface2.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.vggface2.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.vggface2.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.vggface2.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.vggface2.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.vggface2.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.vggface2.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        conv3_1_1x1_proj = self.vggface2.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.vggface2.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1x = self.vggface2.conv3_1_relu(conv3_1)
        conv3_2_1x1_reduce = self.vggface2.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.vggface2.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.vggface2.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.vggface2.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.vggface2.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.vggface2.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.vggface2.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.vggface2.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2x = self.vggface2.conv3_2_relu(conv3_2)
        conv3_3_1x1_reduce = self.vggface2.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.vggface2.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.vggface2.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.vggface2.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.vggface2.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.vggface2.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.vggface2.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.vggface2.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3x = self.vggface2.conv3_3_relu(conv3_3)
        conv3_4_1x1_reduce = self.vggface2.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.vggface2.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.vggface2.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.vggface2.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.vggface2.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.vggface2.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.vggface2.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.vggface2.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4x = self.vggface2.conv3_4_relu(conv3_4)
        conv4_1_1x1_reduce = self.vggface2.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.vggface2.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.vggface2.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.vggface2.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.vggface2.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.vggface2.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.vggface2.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.vggface2.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        conv4_1_1x1_proj = self.vggface2.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.vggface2.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1x = self.vggface2.conv4_1_relu(conv4_1)
        conv4_2_1x1_reduce = self.vggface2.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.vggface2.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.vggface2.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.vggface2.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.vggface2.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.vggface2.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.vggface2.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.vggface2.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2x = self.vggface2.conv4_2_relu(conv4_2)
        conv4_3_1x1_reduce = self.vggface2.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.vggface2.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.vggface2.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.vggface2.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.vggface2.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.vggface2.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.vggface2.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.vggface2.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3x = self.vggface2.conv4_3_relu(conv4_3)
        conv4_4_1x1_reduce = self.vggface2.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.vggface2.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.vggface2.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.vggface2.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.vggface2.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.vggface2.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.vggface2.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.vggface2.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4x = self.vggface2.conv4_4_relu(conv4_4)
        conv4_5_1x1_reduce = self.vggface2.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.vggface2.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.vggface2.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.vggface2.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.vggface2.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.vggface2.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.vggface2.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.vggface2.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5x = self.vggface2.conv4_5_relu(conv4_5)
        conv4_6_1x1_reduce = self.vggface2.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.vggface2.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.vggface2.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.vggface2.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.vggface2.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.vggface2.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.vggface2.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.vggface2.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6x = self.vggface2.conv4_6_relu(conv4_6)
        conv5_1_1x1_reduce = self.vggface2.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.vggface2.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.vggface2.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.vggface2.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.vggface2.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.vggface2.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.vggface2.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.vggface2.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        conv5_1_1x1_proj = self.vggface2.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.vggface2.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1x = self.vggface2.conv5_1_relu(conv5_1)
        conv5_2_1x1_reduce = self.vggface2.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.vggface2.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.vggface2.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.vggface2.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.vggface2.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.vggface2.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.vggface2.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.vggface2.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2x = self.vggface2.conv5_2_relu(conv5_2)
        conv5_3_1x1_reduce = self.vggface2.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.vggface2.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.vggface2.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.vggface2.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.vggface2.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.vggface2.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.vggface2.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.vggface2.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3x = self.vggface2.conv5_3_relu(conv5_3)
        pool5_7x7_s1 = self.vggface2.pool5_7x7_s1(conv5_3x)
        classifier_flatten = pool5_7x7_s1.view(pool5_7x7_s1.size(0), -1)
        # classifier_flatten = torch.flatten(pool5_7x7_s1)
        classifier = self.vggface2.classifier(classifier_flatten)

        if self.classes == 1:
            classifier = self.sigmoid(classifier)
        return classifier

    def get_mean(self):
        return self.vggface2.meta["mean"]

    def get_std(self):
        return self.vggface2.meta["std"]


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGGFace2(pretrained=True).to(device)
    print("Model archticture: ", model)

    x = np.random.rand(224, 224, 3)
    x = Image.fromarray(x.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        out = model(transforms.ToTensor()(x).unsqueeze_(0))
        print(out)
