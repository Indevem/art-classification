import matplotlib.pyplot as plt
import random
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms as tt

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True),
            nn.Conv2d(8, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )
        self.seq_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )
        self.seq_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.5)
        )
        self.seq_4 = nn.Sequential(
            Flatten(),
            nn.Linear(36864, 1000, bias=True),
            nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(0.4),
            nn.Linear(1000, 4, bias=True),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.seq_1(x)
        x = self.seq_2(x)
        x = self.seq_3(x)
        x = self.seq_4(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "simplenet":
        model_ft = simplenet = SimpleNet()

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ['simplenet', 'resnet', 'alexnet', 'vgg']
models = dict()
for name in model_names:
    models[name] = initialize_model(model_name, 4, True, use_pretrained=True)
    models[name].load_state_dict(torch.load(name + '.pth'))
    models[name] = models[name].to(device)
    models[name].eval()

transforms = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def make_prediction(img, device):
	img = transforms(img.convert("RGB"))
    predictions = dict()
    for model_name in models.keys():
        output = nn.Softmax(dim=1)(models[key](transforms(img.convert("RGB")).view(1, 3, 224, 224).to(DEVICE)).cpu())
        predictions[model_name] = output.tolist()
    return img, predictions
