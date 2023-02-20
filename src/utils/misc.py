import os
import cornet
import external.vonenet.vonenet as vonenet
import numpy as np
import torch
import matplotlib.pyplot as plt
from sty import fg, ef, rs
from torchvision.transforms import transforms
from external.prednet.pytorch_prednet.prednet import PredNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Human results from
# Perception of wholes and of their component parts: Some configural superiority effects  -
# James R. Pomerantz, Lawrence C. Sager, and Robert J. Stoever 1977
# This is referred as exp2.
# STRONGEST CSE: 3-4, 10-11, A-B; STRONGEST CIE:  6-9, 6-8
import re
from copy import deepcopy
exp2_1_pairs = [['arrayA', 'arrayB'],
                ['arrayA', 'arrayC'],
                ['arrayA', 'arrayD'],
                ['arrayA', 'arrayE'],
                ['arrayA', 'arrayF']]

exp2_2_pairs = [
    ['array1', 'array2'],
    ['array3', 'array4'],
    ['array3', 'array5'],
    ['array6', 'array7'],
    ['array6', 'array8'],
    ['array6', 'array9'],
    ['array10', 'array11'],
    ['array10', 'array12'],
    ['array10', 'array13'],
    ['array14', 'array15'],
    ['array14', 'array16'],
    ['array14', 'array17']
]

all_pairs = deepcopy(exp2_1_pairs)
all_pairs.extend(exp2_2_pairs)

alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
RTexp2_1 = {f'array{alph[k]}': v for k, v in enumerate([2.40, 1.45, 1.71, 2.95, 2.45, 3.52, 3.49, 2.09, 2.40, 2.50])}
SEexp2_1 = {f'array{alph[k]}': v for k, v in enumerate([0.13, 0.10, 0.09, 0.29, 0.11, 0.24, 0.55, 0.14, 0.14, 0.08])}
Nexp2_1 = 12
STDexp2_1 = {k: v * np.sqrt(Nexp2_1) for k, v in SEexp2_1.items()}
RTexp2_2 = {f'array{k + 1}': v / 1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}
RT = {}
RT.update(RTexp2_2)
RT.update(RTexp2_1)
human_CSE_exp2_perc = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: (RT[p[0]] - RT[p[1]]) / RT[p[1]] for p in all_pairs}
human_CSE_exp2_perc = {k: v for k, v in sorted(human_CSE_exp2_perc.items(), key=lambda item: item[1], reverse=True)}
human_CSE_exp2 = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: RT[p[0]] - RT[p[1]] for p in all_pairs}
human_CSE_exp2 = {k: v for k, v in sorted(human_CSE_exp2.items(), key=lambda item: item[1], reverse=True)}
all_comb_exp1 = [re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0] for p in all_pairs]
all_comb_exp1 = [i for i in all_comb_exp1 if i in human_CSE_exp2]

#####
# Human results from Experiment 1
# Grouping and Emergent Features in Vision: Toward a Theory of Basic Gestalts
# James R. Pomerantz and Mary C. Portillo Rice University

RTexp1 = {'single_dot': np.array([1455, 1400, 1389]),
       'proximity':  np.array([994, 1087, 1002]),
       'linearity': np.array([1021,  1064, 1075]),
        'orientation': np.array([1139, 1138, 1297])}

human_CSE_exp1 = {k: np.mean((RTexp1['single_dot'] - RTexp1[k]) / 1000) for k, v in RTexp1.items()}



from copy import deepcopy
brain_score_nn = {
    'resnet152': 0.432,
    'densenet201': 0.421,
    'inception_v3': 0.414,
    'vonenet-resnet50': 0.405,
    'cornet-s': 0.402,
    'vgg19bn': 0.402,
    'vonenet-cornets': 0.390,
    'alexnet': 0.381,
    'vit_b_32': 0.355,
    'vit_l_32': 0.198,
    'prednet': 0.195,
    'vit_b_16': 0.190,
    'vit_l_16': 0.161,
    'simCLR_resnet18_stl10': 0.160,
    'dino_vitb8': np.nan,
    'dino_vits8': np.nan}

transformers = ['vit_b_32', 'vit_l_32', 'vit_b_16', 'vit_l_16', 'dino_vitb8', 'dino_vits8']
recurrent = ['cornet-s', 'vonenet-cornets', 'cornet-s', 'prednet']

main_text_nets = ['vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-resnet50', 'vit_l_32', 'dino_vitb8']
appendix_nets = ['alexnet',  'vonenet-cornets', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'dino_vits8', 'simCLR_resnet18_stl10', 'prednet']
all_nets = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'] #, 'dino_vits8', 'dino_vitb8', 'simCLR_resnet18_stl10', 'prednet']

self_superv_net = ['dino_vitb8', 'dino_vits8', 'simCLR_resnet18_stl10', 'prednet']

class RandomBackground(torch.nn.Module):
    def __init__(self, color_to_randomize=0):
        super().__init__()
        self.color_to_randomize = color_to_randomize

    def forward(self, input):
        i = np.array(input)
        s = len(i[i == self.color_to_randomize])

        i[i == self.color_to_randomize] = np.repeat([np.random.randint(0, 255, 3)], s/3, axis=0).flatten()
        return transforms.ToPILImage()(i)


def save_fig_pair(path, set, n=4):
    fig, ax = plt.subplots(n, 2)
    if np.ndim(ax) == 1:
        ax = np.array([ax])
    for idx, axx in enumerate(ax):
        axx[0].imshow(set[idx][0])
        axx[1].imshow(set[idx][1])
    [x.axis('off') for x in ax.flatten()]
    plt.gcf().set_size_inches([2.4, 5])
    plt.savefig(path)



def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def print_net_info(net):
    num_trainable_params = 0
    tmp = ''
    print(fg.yellow)
    print("Params to learn:")
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            tmp += "\t" + name + "\n"
            print("\t" + name)
            num_trainable_params += len(param.flatten())
    print(f"Trainable Params: {num_trainable_params}")

    print('***Network***')
    print(net)
    print(ef.inverse + f"Network is in {('~train~' if net.training else '~eval~')} mode." + rs.inverse)
    print(rs.fg)
    print()


def make_cuda(fun, is_cuda):
    return fun.cuda() if is_cuda else fun


def from_netname_to_str(str):
    if str == 'densenet201':
        return 'DenseNet-201'
    if str == 'inception_v3':
        return 'Inception V3'
    if str == 'vgg19bn':
        return 'VGG19'
    if str == 'vonenet-resnet50':
        return 'VOneResNet50'
    if str == 'prednet-train-sup':
        return 'PredNet'
    if str == 'vonenet-cornets':
        return 'VOneCORnet-S'
    if str == 'cornet-s':
        return 'CORnet-S'
    if str == 'simCLR_resnet18_stl10':
        return 'SimCLR'
    if str == 'resnet152':
        return 'ResNet-152'
    if str == 'vit_b_16':
        return 'ViT-B/16'
    if str == 'vit_b_32':
        return 'ViT-B/32'
    if str == 'prednet':
        return 'PredNet'
    if str == 'alexnet':
        return 'AlexNet'
    if str == 'vit_l_32':
        return 'ViT-L/32'
    if str == 'vit_l_16':
        return 'ViT-L/16'
    if str == 'dino_vitb8':
        return 'DINO ViT-B/8'
    if str == 'dino_vits8':
        return 'DINO ViT-S/8'
    else:
        return str


def standard_step(data, model, loss_fn, optimizer, use_cuda, logs, logs_prefix, train, **kwargs):
    images, labels, _ = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    optimizer.zero_grad() if train else None
    if model.__class__.__name__ == 'PredNet':
        images = images.unsqueeze(1)  # prednet output is (bs, nc, nh, nw, nt)
    output_batch = model(images)
    if hasattr(output_batch, 'logits'):
        output_batch = output_batch.logits
    loss = loss_fn(output_batch,
                   labels)
    logs['output'] = output_batch
    predicted = torch.argmax(output_batch, -1)

    logs[f'{logs_prefix}ema_loss'].add(loss.item())
    logs[f'{logs_prefix}ema_acc'].add(torch.mean((predicted == labels).float()).item())
    logs[f'{logs_prefix}ca_acc'].add(torch.mean((predicted == labels).float()).item())
    logs[f'{logs_prefix}y_true'] = labels
    logs[f'{logs_prefix}y_pred'] = predicted
    if 'collect_images' in kwargs and kwargs['collect_images']:
        logs[f'{logs_prefix}images'] = images
    if train:
        loss.backward()
        optimizer.step()

    return loss, labels, predicted, logs


def config_to_path_special_par(config):
    return f"{config.distance_type}/special_par/{config.network_name}" \
        + '//' \
        + f'{config.background}' \
        + f'_{config.pretraining}' \
        + '//' \
        + f'{config.transf_code}_{config.type_ds[0]}' \
        + '//' \
        + f'{config.type_ds_args["dist"]:.2f}_'

def config_to_path_special(config):
    return f"{config.distance_type}/special/{config.network_name}" \
        + '//' \
        + f'{config.background}' \
        + f'_{config.pretraining}' \
        + '//' \
        + f'{config.transf_code}' \
        + '//' \
        + f'{config.type_ds}'

def config_to_path_shape_fam(config):
    return f"shape_fam/{config.network_name}" \
        + '//' \
        + f'{config.background}' \
        + f'_{config.pretraining}' \
        + '_' \
        + f'{config.transf_code}' \
        + '//' \
        + f'{config.type_ds}' \
        + '//' \
        + f'{config.type_ds_args}'



def config_to_path_train(config):
    return f"dataset/{config.type_ds}" \
        + '//' \
        + f'{config.network_name}'



def config_to_path_hierarchical(config, tmp_tags=None):
    return f"{config.distance_type}/hierarchical/{config.network_name}" \
        + '//' \
        + f'{config.background}' \
        + f'_{config.pretraining}' \
        + (f'_{tmp_tags}' if tmp_tags else '') \
        + '//'


from src.utils.net_utils import GrabNet
class MyGrabNet(GrabNet):

    @staticmethod
    def get_other_nets(network_name, imagenet_pt, **kwargs):
        net = None
        norm_stats = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        resize = None
        if network_name == 'prednet-train-sup':
            resize = (128, 160)
            A_channels = (3, 48, 96, 192)
            R_channels = (3, 48, 96, 192)
            net = PredNet(resize, R_channels, A_channels, output_mode='R3', gating_mode='mul',
                          peephole=False, lstm_tied_bias=False)
            norm_stats = dict(mean=[0., 0., 0.], std=[1., 1., 1.])
        if network_name == 'prednet':
            resize = (128, 160)
            A_channels = (3, 48, 96, 192)
            R_channels = (3, 48, 96, 192)
            net = PredNet(resize, R_channels, A_channels, output_mode='R3', gating_mode='mul',
                          peephole=False, lstm_tied_bias=False)
            norm_stats = dict(mean=[0., 0., 0.], std=[1., 1., 1.])
        if network_name == 'cornet-rt':
            net = cornet.cornet_rt(pretrained=True, map_location='cpu', times=5)
        elif network_name == 'cornet-s':
            net = cornet.cornet_s(pretrained=True, map_location='cpu')
        elif network_name == 'clip':
            import clip
            net, _ = clip.load("ViT-B/32", device='cpu')
        if 'vonenet' in network_name:
            norm_stats = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            os.environ['HOME'] = './'
            if network_name == 'vonenet-resnet50':
                net = vonenet.get_model(model_arch='resnet50', pretrained=imagenet_pt)
            elif network_name == 'vonenet-cornets':
                net = vonenet.get_model(model_arch='cornets', pretrained=imagenet_pt)
            elif network_name == 'vonenet-alexnet':
                net = vonenet.get_model(model_arch='alexnet', pretrained=imagenet_pt)
            elif network_name == 'vonenet-resnet50-non-stoch':
                net = vonenet.get_model(model_arch='resnet50', pretrained=imagenet_pt, noise_level=0)

        if not net:
            assert False, f"Network {network_name} not recognised"
        return net, norm_stats, resize


class RandomPixels(torch.nn.Module):
    def __init__(self, background_color=(0, 0, 0), line_color=(255, 255, 255)):
        super().__init__()
        self.background_color = background_color
        self.line_color = line_color

    def forward(self, input):
        i = np.array(input)
        i = i.astype(np.int16)
        s_line = len(i[i == self.line_color])
        i[i == self.line_color] = np.repeat([1000, 1000, 1000], s_line/3, axis=0).flatten()

        s = len(i[i == self.background_color])
        i[i == self.background_color] = np.random.randint(0, 255, s)

        s_line = len(i[i == [1000, 1000, 1000]])
        i[i == [1000, 1000, 1000]] = np.repeat([0, 0, 0], s_line / 3, axis=0).flatten()
        i = i.astype(np.uint8)

        return transforms.ToPILImage()(i)



def imshow_batch(inp, stats=None, labels=None, title_more='', maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats['mean']
        std = stats['std']
    """Imshow for Tensor."""

    cols = int(np.ceil(np.sqrt(len(inp))))
    if ax is None:
        fig, ax = plt.subplots(int(np.ceil(len(inp) / cols)), cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis('off')
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            text = str(labels[idx]) + ' ' + (title_more[idx] if title_more != '' else '')
            # ax[idx].set_title(text, size=5)
            ax[idx].text(0.5, 0.1, f'{labels[idx]:.3f}', horizontalalignment='center', transform=ax[idx].transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=1,
                        bottom=0.01,
                        left=0,
                        right=1,
                        hspace=0.2,
                        wspace=0.01)
    return ax


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image
