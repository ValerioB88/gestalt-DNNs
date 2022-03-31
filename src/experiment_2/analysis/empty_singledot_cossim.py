import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.utils.create_stimuli.drawing_utils import *
from src.utils.net_utils import *
from src.utils.Config import Config
from src.utils.misc import *

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def collect(network_name, pretraining, background):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=background,
                    draw_obj=DrawShape(background='black' if background == 'black' or background == 'random' else background, img_size=img_size, width=14))
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))
    all_layers = list(cs['empty'].keys())
    # diff_penultimate = np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    ll = get_layer_from_depth_str(all_layers, depth_layer)

    m = {k: np.mean(i[ll]) for k, i in cs.items() if k in type_ds}
    s = {k: np.std(i[ll]) for k, i in cs.items() if k in type_ds}
    return {'mean': m, 'std': s}

def plot_net_set(m, s, x):
    span = 0.6
    width = span / (len(m)+2 - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])


##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

# network_names = ['vgg16', 'alexnet', 'densenet121', 'vgg16bn', 'resnet18', 'vonenet-resnet50', 'vonenet-cornets', 'vonenet-alexnet', 'cornet-s', 'cornet-rt']
type_ds = ['empty', 'empty-single']
# type_ds = ['single', 'proximity', 'orientation', 'linearity']
background = ['random', 'black', 'white']
##
import seaborn as sns
sns.set(style="white")


plt.close('all')

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
depth_layer = 5
a = {nn: collect(pretraining='ImageNet', network_name=nn, background='random') for nn in network_names}
[plot_net_set(list(a[nn]['mean'].values()),
              list(a[nn]['std'].values()),
              idx) for idx, nn in enumerate(network_names)]


depth_layer = 4
a = {nn: collect(pretraining='ImageNet', network_name=nn, background='random') for nn in network_names}
# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
[plot_net_set(list(a[nn]['mean'].values()),
              list(a[nn]['std'].values()),
              idx) for idx, nn in enumerate(network_names)]


depth_layer = 3
a = {nn: collect(pretraining='ImageNet', network_name=nn, background='random') for nn in network_names}
# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
[plot_net_set(list(a[nn]['mean'].values()),
              list(a[nn]['std'].values()),
              idx) for idx, nn in enumerate(network_names)]

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='s', markersize=5, color=c, linewidth=1) for c in color_cycle]


leg2 = plt.legend(custom_lines, type_ds, prop={'size': 15}, framealpha=1, ncol=4, facecolor='w', bbox_to_anchor=(0.3, 1.1), loc="upper left")
ax.set_xticks(range(len(network_names)))
ax.set_xticklabels(from_netname_to_str(network_names), rotation=0)
plt.axhline(0, color='k', linestyle='--')
# plt.ylim(-0.3, 0.3)
# plt.xlim([-0.5, 3])
plt.ylabel('Cosine Similarity')


plt.show()
plt.savefig(f'./results/figures/single_figs/empty_single.svg')
##

