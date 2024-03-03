import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from .base_mdel import BaseModel
from torchvision.models.resnet import BasicBlock

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cpu()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cpu()


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cpu')[x.is_cpu])().long(), :]
    return x.view(xsize)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class CADBasicBlock(BasicBlock):

    def __init__(self, *args, **kwargs):
        super(CADBasicBlock, self).__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(self.bn1(out))

        return out


class SVCNN(BaseModel):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)

        self.classnames = ['00_Gear', '01_Washer', '02_Steel', '03_Nut', '04_Screw', '05_Spring', '06_Bearing',
                           '07_Flange',
                           '08_Ball', '09_Bolt', '10_Elbow', '11_Grooved_Pin', '12_Stud', '13_Round_Nut',
                           '14_Lock_Washer',
                           '15_Bevel_Gear', '16_Helical_Gear', '17_Key', '18_BearingHouse', '19_Distributor',
                           '20_HeaveTightCouplingSleeve',
                           '21_TemplateAndPlate', '22_LiftingHook', '23_WireTensioner', '24_ForgedShackle',
                           '25_SplineEndWrenches',
                           '26_BoringBar', '27_KeylessDrillChuck', '28_HydraulicComponent', '29_BallValve']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cpu()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cpu()

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net_2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                )
                # classification
                self.net_3 = nn.Linear(256, 30)
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.layer1 = nn.Sequential(CADBasicBlock(64, 64), CADBasicBlock(64, 64))
                self.net.layer2 = nn.Sequential(CADBasicBlock(64, 128, stride=2,
                                                              downsample=nn.Sequential(conv1x1(64, 128, 2),
                                                                                       nn.BatchNorm2d(128))),
                                                CADBasicBlock(128, 128))
                self.net.layer3 = nn.Sequential(CADBasicBlock(128, 256, stride=2,
                                                              downsample=nn.Sequential(conv1x1(128, 256, 2),
                                                                                       nn.BatchNorm2d(256))),
                                                CADBasicBlock(256, 256))
                self.net.layer4 = nn.Sequential(CADBasicBlock(256, 512, stride=2,
                                                              downsample=nn.Sequential(conv1x1(256, 512, 2),
                                                                                       nn.BatchNorm2d(512))),
                                                CADBasicBlock(512, 512))
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, 40)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, 40)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            # elif self.cnn_name == 'vgg11':
            #     self.net_1 = models.vgg11(pretrained=self.pretraining).features
            #     self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            #     self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'googlenet':
                self.net_1 = models.googlenet(pretrained=self.pretraining)
                self.net_1._modules['fc'] = nn.Linear(1024, 256)
                self.net_2 = nn.ReLU(inplace=True)
            # classification
            self.net_3 = nn.Linear(256, 30)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.pool(self.net_1(x))
            return self.net_2(y.view(y.shape[0], -1))


class MVCNN(BaseModel):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames = ['00_Gear', '01_Washer', '02_Steel', '03_Nut', '04_Screw', '05_Spring', '06_Bearing',
                           '07_Flange',
                           '08_Ball', '09_Bolt', '10_Elbow', '11_Grooved_Pin', '12_Stud', '13_Round_Nut',
                           '14_Lock_Washer',
                           '15_Bevel_Gear', '16_Helical_Gear', '17_Key', '18_BearingHouse', '19_Distributor',
                           '20_HeaveTightCouplingSleeve',
                           '21_TemplateAndPlate', '22_LiftingHook', '23_WireTensioner', '24_ForgedShackle',
                           '25_SplineEndWrenches',
                           '26_BoringBar', '27_KeylessDrillChuck', '28_HydraulicComponent', '29_BallValve']
        self.cnn_name = model.cnn_name
        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cpu()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cpu()

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net_2
            self.net_3 = model.net_3

    def forward(self, x):
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1])).squeeze()
        features = self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
        return self.net_3(features), features
