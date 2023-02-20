import os
import pickle
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from src.utils.create_stimuli.drawing_utils import *

from src.utils.Config import Config
from src.utils.misc import *

plt.plot([1,2,3])

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def collect(network_name, background):
    img_size = np.array((224, 224), dtype=int)
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=background,
                    draw_obj=DrawShape(background='black' if background == 'black' or background == 'random' else background, img_size=img_size, width=14))
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))
    all_layers = list(cs[type_ds].keys())
    penultimate = {k: i[all_layers[-1]] for k, i in cs.items()}
    m = {k: np.mean(i) for k, i in penultimate.items() if k in type_ds}
    s = {k: np.std(i) for k, i in penultimate.items() if k in type_ds}
    return {'mean': m, 'std': s}

def plot_net_set(m, s, x):
    span = 0.6
    width = span / (len(m)+2 - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])

##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# network_names = ['vgg16', 'alexnet', 'densenet121', 'vgg16bn', 'resnet18', 'vonenet-resnet50', 'vonenet-cornets', 'vonenet-alexnet', 'cornet-s', 'cornet-rt']
main_text_nets = ['vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-resnet50', 'vit_l_32', 'dino_vitb8', ]
appendix_nets = ['alexnet', 'densenet201', 'vonenet-cornets', 'vit_b_32', 'vit_l_16', 'dino_vits8', 'simCLR_resnet18_stl10', 'prednet']

############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = main_text_nets
#################################
# type_ds = ['empty', 'empty-single', 'single', 'proximity', 'orientation', 'linearity']
type_ds = ['empty-single', 'single', 'proximity', 'orientation', 'linearity']
# background = ['random', 'black', 'white']

##
import seaborn as sns
sns.set(style="white")
a = {nn: collect(network_name=nn, background='random') for nn in network_names}


plt.close('all')

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
[plot_net_set(list(a[nn]['mean'].values()),
              list(a[nn]['std'].values()),
              idx) for idx, nn in enumerate(network_names)]

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='s', markersize=5, color=c, linewidth=1) for c in color_cycle]

leg2 = plt.legend(custom_lines, list(a['vgg19bn']['mean'].keys()), prop={'size': 12}, framealpha=1, ncol=5, facecolor='w', bbox_to_anchor=(0.0, 1.1), loc="upper left")
ax.set_xticks(range(len(network_names)))
ax.set_ylim([0.6, 1])
ax.set_xticklabels(from_netname_to_str(network_names))
plt.axhline(0, color='k', linestyle='--')
ax.set_xticklabels(network_names, rotation=30)
plt.tight_layout()
# plt.ylim(-0.3, 0.3)
# plt.xlim([-0.5, 3])
plt.ion()
plt.ylabel('Cosine Similarity')



##

