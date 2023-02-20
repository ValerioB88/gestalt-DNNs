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

def collect(network_name, background):
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type='cossim',
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
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


############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = ['vit_b_16'] # all_nets
type_ds = ['empty', 'empty-single']
# type_ds = ['single', 'proximity', 'orientation', 'linearity']
#############################################################################
import seaborn as sns
sns.set(style="white")


plt.close('all')

fig, ax = plt.subplots(1, 1, figsize=(8*1.5, 5))
depth_layer = 'last_l'

a = {nn: collect(network_name=nn, background='random') for nn in network_names}
[plot_net_set(list(a[nn]['mean'].values()),
              list(a[nn]['std'].values()),
              idx) for idx, nn in enumerate(network_names)]

#
# depth_layer = 'last_conv_l'
# a = {nn: collect(network_name=nn, background='random') for nn in network_names}
# # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# [plot_net_set(list(a[nn]['mean'].values()),
#               list(a[nn]['std'].values()),
#               idx) for idx, nn in enumerate(network_names)]
#
#
# depth_layer = 'middle'
# a = {nn: collect(network_name=nn, background='random') for nn in network_names}
# # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# [plot_net_set(list(a[nn]['mean'].values()),
#               list(a[nn]['std'].values()),
#               idx) for idx, nn in enumerate(network_names)]

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='s', markersize=5, color=c, linewidth=1) for c in color_cycle]

def convert_labels(t):
    if t == 'empty':
        return 'Empty Pair'
    if t == 'empty-single':
        return 'Single Dot'
ax.grid(axis='y')

leg2 = plt.legend(custom_lines, [convert_labels(i) for i in type_ds], prop={'size': 15}, framealpha=1, ncol=4, facecolor='w', bbox_to_anchor=(0.3, 1.1), loc="upper left")
ax.set_xticks(range(len(network_names)))
ax.set_xticklabels([from_netname_to_str(i) for i in network_names], rotation=90)
plt.yticks([0, 0.25, 0.5, 0.75, 1])

plt.axhline(0, color='k', linestyle='--')
# plt.ylim(-0.3, 0.3)
# plt.xlim([-0.5, 3])
plt.ylabel('Cosine Similarity')


plt.savefig(f'./figures/single_figs/empty_single.svg')
plt.show()
##

