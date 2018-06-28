import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, num_classes=100, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = Extractor()
        self.embedding = Embedding()
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x, norm=True, scale=True):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        basenet = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x