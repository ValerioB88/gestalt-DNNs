import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torchvision
import torch.nn as nn
from sty import fg, ef, rs, bg
import os
import src.ML_framework.framework_utils as framework_utils

def get_layer_from_depth_str(all_layers, depth_layer):
    if depth_layer == 'early':
        dd = 1
    if depth_layer == 'middle_early':
        dd = 2
    if depth_layer == 'middle':
        dd = 3
    if np.any([depth_layer == i for i in ['early', 'middle_early', 'middle']]):
        ll = all_layers[int(len(all_layers) / 4 * dd - 1)]

    if depth_layer == 'penultimate_l':
        ll = all_layers[-2]

    if depth_layer == 'last_l':
        ll = all_layers[-1]

    if depth_layer == 'last_conv_l':
        ll = all_layers[[idx for idx, i in enumerate(all_layers) if 'Linear' in i][0] - 1]

    if depth_layer == 'first_linear_l':
        ll = all_layers[[idx for idx, i in enumerate(all_layers) if 'Linear' in i][0]]
    return ll


def from_depth_to_string(d):
    if d == 'middle-early':
        return 'middle-early'
    if d == 'penultimate_l':
        return 'penultimate'
    if d == 'last_l':
        return 'last layer'
    if d == 'last_conv_l':
        return 'last convolutional layer'
    if d == 'first_linear':
        return 'first linear layer'



class GrabNet():
    @classmethod
    def get_net(cls, network_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        kwargs = dict(num_classes=nc) if nc is not None else dict()
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net = torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet152':
            net = torchvision.models.resnet152(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, **kwargs)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net

    @staticmethod
    def get_other_nets(network_name, num_classes, imagenet_pt, **kwargs):
        pass


def prepare_network(net, config, train=True):
    pretraining_file = 'vanilla' if config.pretraining == 'ImageNet' else config.pretraining
    net = load_pretraining(net, pretraining_file, config.use_cuda)
    net.cuda() if config.use_cuda else None
    cudnn.benchmark = True
    net.train() if train else net.eval()
    framework_utils.print_net_info(net) if config.verbose else None



def load_pretraining(net, pretraining, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if pretraining != 'vanilla':
        if os.path.isfile(pretraining):
            print(fg.red + f"Loading.. full model from {pretraining}..." + rs.fg, end="")
            ww = torch.load(pretraining, map_location='cuda' if use_cuda else 'cpu')
            if 'full' in ww:
                ww = ww['full']
            net.load_state_dict(ww)
            print(fg.red + " Done." + rs.fg)
        else:
            assert False, f"Pretraining path not found {pretraining}"

    return net

