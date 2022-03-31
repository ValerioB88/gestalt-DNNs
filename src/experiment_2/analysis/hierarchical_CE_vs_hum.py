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
import seaborn as sns
sns.set(style="white")


def collect(pretraining, network_name, type_ds, background, depth_layer):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    type_ds=type_ds,
                    background=background,
                    draw_obj=DrawShape(background='black' if background == 'black' or background == 'random' else background, img_size=img_size, width=10))

    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))
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
    diff_penultimate = cs[type_ds][ll] #np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    return np.array(diff_penultimate)


##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

type_ds = [f'array{i}' for i in range(1, 19)]

background = ['random', 'black', 'white']

RT = {f'array{k}': v for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}

##

bk = 'random'
depth_layer = 5


def plot_net_set(m, s, x):
    span = 0.6
    width = span / (len(m) - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])

def compute_base_comp(base, comp, net, depth_layer):
    base_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=base, background=bk, depth_layer=depth_layer)
    composite_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=comp, background=bk, depth_layer=depth_layer)

    m = -np.mean(composite_cossim - base_cossim)
    s = np.std(composite_cossim - base_cossim)
    return m, s

##
# type_ds = ['empty', 'empty-single', 'single', 'proximity', 'orientation', 'linearity']

plt.close('all')
all_pairs = [['single', 'orientation'],
             ['single', 'proximity'],
             ['single', 'linearity']]

# RT = {f'array{k + 1}': v/1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}

RT = {'single': np.mean([1407, 1455, 1400, 1389]),
      'orientation': np.mean([1430, 1139, 1138, 1297]),
      'proximity': np.mean([925, 994, 1087, 1002]),
      'linearity': np.mean([1016,  1021,  1064, 1075])}

from scipy.stats import spearmanr
human_CSE = {f'{p[0][0]} - {p[1][0]}': RT[p[0]] - RT[p[1]] for p in all_pairs}

all_comb = [f'{p[0][0]} - {p[1][0]}' for p in all_pairs]
fig, ax = plt.subplots(2, int(np.ceil(len(network_names) / 2)), figsize=[14.26,  5.09], sharex='col', sharey=True)
ax = ax.flatten()
from adjustText import adjust_text
for idx, net in enumerate(network_names):
    net_CSE = {fr'{p[0][0]} - {p[1][0]}':compute_base_comp(p[0], p[1], net, depth_layer)[0] for p in all_pairs}
    corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])

    ax[idx].plot(corr[:, 0], corr[:, 1], 'o', color=color_cycle[idx], label=net)
    ax[idx].annotate(from_netname_to_str(net),
                     xy=(0.08, 0.15), xycoords='axes fraction',
                     textcoords='offset points',
                     size=10,
                     bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))

    text = [ax[idx].text(net_CSE[txt], human_CSE[txt], txt) for txt in all_comb if txt in human_CSE]
    adjust_text(text, ax=ax[idx])
    r = spearmanr(corr[:, 0], corr[:, 1])
    # ax[idx].annotate(rf'$r_s : {r[0]:.02f}, p={r[1]:.3f}$',
    #                  xy=(0.05, 0.7), xycoords='axes fraction',
    #                  textcoords='offset points',
    #                  size=10,
    #                  bbox=dict(boxstyle="round", fc=(1, 1, 1), ec="none"))
    ax[idx].axvline(0, linestyle='--')
    ax[idx].axhline(0, linestyle='--')
    ax[idx].set_xlim([-0.5, 0.1])
    ax[idx].set_ylim([-100, 500])


fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.xlabel('networks CE')
plt.ylabel('humans CE')
plt.show()

plt.savefig(f'./results/figures/single_figs/hierarchicalCSE_{bk}.svg')
plt.savefig(f'./results/figures/pngs/hierarchicalCSE_{bk}.png')


## Agreement

agreement = {}
for idx, net in enumerate(network_names):

    net_CSE = {fr'{p[0][0]} - {p[1][0]}': compute_base_comp(p[0], p[1], net, depth_layer)[0] for p in all_pairs}
    corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])
    agreement[net] = np.mean(np.sign(corr[:,0]) == np.sign(corr[:, 1]))


#
#
#
# ## Few special stimuli
#
# m, s = np.hsplit(np.array([compute_base_comp('array1', 'array2', net, depth_layer) for net in network_names]), 2)
# fig, ax = plt.subplots(1, 1, figsize=[11.64, 4.42])
# plot_net_set(m.flatten(), s.flatten(), 1)
#
#
#
# m, s = np.hsplit(np.array([compute_base_comp('array3', 'array4', net, depth_layer) for net in network_names]), 2)
# plot_net_set(m.flatten(), s.flatten(), 2)
#
#
# m, s = np.hsplit(np.array([compute_base_comp('array6', 'array9', net, depth_layer) for net in network_names]), 2)
# plot_net_set(m.flatten(), s.flatten(), 3)
#
# m, s = np.hsplit(np.array([compute_base_comp('array10', 'array11', net, depth_layer) for net in network_names]), 2)
# plot_net_set(m.flatten(), s.flatten(), 4)
#
# #
# m, s = np.hsplit(np.array([compute_base_comp('single-parentheses', 'double-parentheses', net, transf_code, depth_layer) for net in network_names]), 2)
# plot_net_set(m.flatten(), s.flatten(), 5)

# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0],  linestyle='', marker='s', markersize=5, color=c, linewidth=1) for c in color_cycle]
#
#
# leg2 = plt.legend(custom_lines, network_names, prop={'size': 10}, framealpha=1, ncol=3, facecolor='w', bbox_to_anchor=(0.5, 0.0), loc="lower left")
# ax.set_title(f'Transformation Applied: {transf}, Layer Depth: {depth_layer}')
# plt.savefig(f'./results/figures/single_figs/special_CSE_T{transf}_bk{bk}.svg')


##

