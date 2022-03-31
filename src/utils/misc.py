import os
import cornet
import src.external.vonenet.vonenet as vonenet
import numpy as np
from src.utils.net_utils import GrabNet
import torch
import matplotlib.pyplot as plt
from sty import fg, bg, ef, rs
from torchvision.transforms import transforms
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
    """
    Get net must be reimplemented for any non abstract base class. It returns the network and the parameters to be updated during training
    """
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

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None, ax=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    if ax is None:
        ax = plt.gca()
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)


def from_netname_to_str(str):
    if str == 'inception_v3':
        return 'inception v3'
    if str == 'vonenet-cornets':
        return 'vonenet-cornet-s'
    else:
        return str


def standard_step(data, model, loss_fn, optimizer, use_cuda, logs, train, **kwargs):
    images, labels, _ = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    optimizer.zero_grad() if train else None
    output_batch = model(images)
    if hasattr(output_batch, 'logits'):
        output_batch = output_batch.logits
    loss = loss_fn(output_batch,
                   labels)
    logs['output'] = output_batch
    predicted = torch.argmax(output_batch, -1)

    logs['ema_loss'].add(loss.item())
    logs['ema_acc'].add(torch.mean((predicted == labels).float()).item())
    logs['ca_acc'].add(torch.mean((predicted == labels).float()).item())
    logs['y_true'] = labels
    logs['y_pred'] = predicted
    if 'collect_images' in kwargs and kwargs['collect_images']:
        logs['images'] = images
    if train:
        loss.backward()
        optimizer.step()

    return loss, labels, predicted, logs


def config_to_path_special_par(config):
    return f"special_par/{config.network_name}" \
           + '//' \
           + f'{config.background}' \
           + f'_{config.pretraining}' \
           + '//' \
           + f'{config.transf_code}_{config.type_ds[0]}' \
           + '//' \
           + f'{config.type_ds_args["dist"]:.2f}_'

def config_to_path_special(config):
    return f"special/{config.network_name}" \
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


#
# def config_to_path_special_base_comp(config):
#     return f"special_base_composite/{config.network_name}" \
#            + '//' \
#            + f'{config.background}' \
#            + f'_{config.pretraining}' \
#            + '//' \
#            + f'{config.type_ds}' \
#            + '//'



def config_to_path_train(config):
    return f"dataset/{config.type_ds}" \
           + '//' \
           + f'{config.network_name}'

def config_to_path_hierarchical_multisaccades(config, tmp_tags=None):
    return f"hierarchical_multisaccades/{config.network_name}" \
           + '//' \
           + f'{config.background}' \
           + f'_{config.pretraining}' \
           + (f'_{tmp_tags}' if tmp_tags else '') \
           + '//'


def config_to_path_hierarchical(config, tmp_tags=None):
    return f"hierarchical/{config.network_name}" \
           + '//' \
           + f'{config.background}' \
           + f'_{config.pretraining}' \
           + (f'_{tmp_tags}' if tmp_tags else '') \
           + '//'

def config_to_path_subhierarchical(config):
    return f"hierarchical/{config.network_name}" \
           + '//' \
           + f'{config.background}' \
           + f'_{config.pretraining}' \
           + f'//{"+".join(config.type_ds)}_'


class MyGrabNet(GrabNet):
    @staticmethod
    def get_other_nets(network_name, imagenet_pt, **kwargs):
        net = None
        if network_name == 'ViT':
            timm.list_models(pretrained=True)
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif network_name == 'cornet-rt':
            net = cornet.cornet_rt(pretrained=True, map_location='cpu', times=5)
        elif network_name == 'cornet-s':
            net = cornet.cornet_s(pretrained=True, map_location='cpu')
        elif network_name == 'clip':
            import clip
            net, _ = clip.load("ViT-B/32", device='cpu')
        if 'vonenet' in network_name:
            os.environ['HOME'] = './'
            if network_name == 'vonenet-resnet50':
                net = vonenet.get_model(model_arch='resnet50', pretrained=True)
            elif network_name == 'vonenet-cornets':
                net = vonenet.get_model(model_arch='cornets', pretrained=True)
            elif network_name == 'vonenet-alexnet':
                net = vonenet.get_model(model_arch='alexnet', pretrained=True)
            elif network_name == 'vonenet-resnet50-non-stoch':
                net = vonenet.get_model(model_arch='resnet50', pretrained=True, noise_level=0)

        if not net:
            assert False, f"Network {network_name} not recognised"
        return net


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
        # plt.imshow(i)

        return transforms.ToPILImage()(i)
