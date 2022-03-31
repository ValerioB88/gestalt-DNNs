import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from src.utils.create_stimuli.drawing_utils import *

from src.utils.Config import Config
from src.utils.misc import *

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str


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

    CSE = {k: [np.array(cs[k][ll]) - np.array(cs['single'][ll]) for ll in cs['single'].keys()]for k in ['orientation', 'proximity', 'linearity']}
    CSE.update( {k: [np.array(cs[k][ll]) - np.array(cs['empty'][ll]) for ll in cs['empty'].keys()]for k in ['empty-single']})
    CE_m = {k: -np.mean(i, axis=1) for k, i in CSE.items()}
    CE_std = {k: np.std(i, axis=1) for k, i in CSE.items()}

    return CE_m, CE_std, all_layers


##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

type_ds = ['empty-single', 'empty', 'single', 'proximity', 'orientation', 'linearity']

##

bk = 'random'
import seaborn as sns
sns.set(style="white")


plt.close('all')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

fig, ax = plt.subplots(2, 4, figsize=[16.78, 6.94], sharex=False, sharey=True)
ax = ax.flatten()
for axidx, nn in enumerate(network_names):
    m, s, all_layers = collect(pretraining='ImageNet', network_name=nn, background=bk)
    ax[axidx].axhline(0, color='k', linewidth=1, linestyle='--')
    for idx, type in enumerate(['empty-single', 'proximity', 'orientation', 'linearity']):
        ax[axidx].plot(range(len(all_layers)), m[type], color=color_cycle[idx], linewidth=1)
        lin_l = [idx for idx, l in enumerate(all_layers) if 'Linear' in l]
        ax[axidx].plot(lin_l,
                     [m[type][i] for i in lin_l], 'o', linewidth=2, color=color_cycle[idx])

        ax[axidx].fill_between(range(len(all_layers)), m[type] + s[type], m[type] - s[type], alpha=0.2, color=color_cycle[idx])
        ax[axidx].set_xticks([])
        ax[axidx].set_xticklabels([], rotation=90)
        ax[axidx].axvline([idx for idx, l in enumerate(all_layers) if 'Linear' in l][0], linestyle='--', color='r')
    ax[axidx].annotate(from_netname_to_str(nn),
                     xy=(0.1, 0.95), xycoords='axes fraction',
                     textcoords='offset points',
                     size=15, weight='bold',
                     bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))

##
fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.xlabel("Network Depth")
plt.ylabel(r"Network CE")

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='-', marker='', color=color_cycle[idx]) for idx in [0, 1, 2, 3]] + \
               [Line2D([0], [0], linestyle='', marker='o', markerfacecolor='w', markeredgecolor='k')]

leg2 = plt.legend(custom_lines, ['Single Dot', 'Proximity', 'Orientation', 'Linearity'], prop={'size': 12}, ncol=4, edgecolor='k', bbox_to_anchor=(0.0, 0), loc="upper left")
leg2.get_frame().set_linewidth(1.5)

# plt.tight_layout()
plt.show()

plt.savefig(f'./results/figures/single_figs/hierarchicalCSE_multilayer_{bk}.svg')
plt.savefig(f'./results/figures/pngs/hierarchicalCSE_multilayer_{bk}.png')

##

