import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from src.utils.create_stimuli.drawing_utils import *

from src.utils.Config import Config
from src.utils.misc import *

from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def collect(network_name, pretraining, type_ds, background):
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
    # all_layers = list(cs['empty'].keys())
    # diff_penultimate = np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])

    all_layers = list(cs[type_ds].keys())
    if depth_layer < 4:
        ll = all_layers[int(len(all_layers) / 4 * depth_layer - 1)]
    if depth_layer == 4:
        ll = all_layers[-2]
    if depth_layer == 5:
        ll = all_layers[-1]
    if depth_layer == 6:
        ll = all_layers[[idx for idx, i in enumerate(all_layers) if 'Linear' in i][0] - 1]
    if depth_layer == 7:
        ll = all_layers[[idx for idx, i in enumerate(all_layers) if 'Linear' in i][0]]

    from scipy.stats import ttest_1samp
    CSE = cs['single'][ll] - np.array(cs[type_ds][ll])
    p_values = ttest_1samp(CSE, 0).pvalue
    CSE_m = np.mean(CSE)
    CSE_std = np.std(CSE)

    return CSE_m, CSE_std, p_values

def plot_net_set(m, s, x, pv):
    span = 0.6
    width = span / (len(m)+2 - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    # rect = ax.barh(x - span / 2 + width * i, m, width, xerr=s, label=network_names, color=color_cycle[:len(m)])
    # plt.axvline(0, color='k', linestyle='--', linewidth=1.5)

    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])
    [ax.text(idx - span / 2 + width * ii, m[ii] + np.sign(m[ii])*(s[ii] + 0.02), f'**' if pv[ii] <0.005 else ('*' if pv[ii] < 0.01 else ''), ha='center', va='center') for ii, _ in enumerate(m)]
    # [ax.text(idx - span / 2 + width * ii  + width * 0.15, m[ii] + s[ii] + m[ii] * 0.1, '*' if pv[ii] < 0.005 else '') for ii, _ in enumerate(m)]
    plt.axhline(0, color='k', linestyle='-', linewidth=1.5)


##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']


plt.close('all')

depth_layer = 5
fig, ax = plt.subplots(1, 1, figsize=[11.64, 4.42])
for idx, k in enumerate(['proximity', 'orientation', 'linearity']):
    m, s, pv = np.hsplit(np.array([collect(net, 'ImageNet', k, 'black') for net in network_names]), 3)
    plot_net_set(m.flatten(), s.flatten(), idx, pv.flatten())

plt.axhline(0, color='k')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='s', color=color_cycle[idx]) for idx,n in enumerate(network_names)]
# leg2 = plt.legend(custom_lines, [from_netname_to_str(n) for n in network_names], prop={'size': 10}, framealpha=1, ncol=1, facecolor='w', bbox_to_anchor=(0.8, 1.1), loc="upper left", edgecolor='k')
plt.tight_layout()
plt.xticks([0, 1, 2], ['proximity', 'orientation', 'linearity'])
plt.ylabel('Networks CSE')
# plt.yticks([-0.1, -0.05, 0, 0.05 , 0.1])
plt.savefig(f'./results/figures/single_figs/barplotCSE_dots_depth{depth_layer}.svg')

# ##


##


