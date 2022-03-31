import re
import os
import pickle
from itertools import product

import pandas as pd
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
from src.utils.create_stimuli.drawing_utils import *
from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str

from src.utils.Config import Config
from src.utils.misc import *
import src.ML_framework.framework_utils

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def collect(pretraining, network_name, type_ds, background, transf_code, depth_layer):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    type_ds=type_ds,
                    background=background,
                    draw_obj=DrawShape(background='black' if background == 'black' or background == 'random' else background, img_size=img_size, width=10),
                    transf_code=transf_code)

    exp_folder = f'./results//{config_to_path_special(config)}'
    cs = pickle.load(open(exp_folder + '_cossim.df', 'rb'))
    all_layers = list(cs[type_ds].keys())
    ll = get_layer_from_depth_str(all_layers, depth_layer)


    diff_penultimate = cs[type_ds][ll]  #np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    return np.array(diff_penultimate)



### Correlation all together
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
# network_names = ['alexnet', 'inception_v3', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']  # 'vonenet-resnet50-non-stoch'
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']  # 'vonenet-resnet50-non-stoch'

type_ds = [f'array{i}' for i in range(1, 19)]
# type_ds.extend(['single-parentheses', 'parentheses-crossed', 'double-parentheses'])

transf_code = ['none', 't', 's', 'r']
background = ['random', 'black', 'white']

# to_use = [2, 4, 7, 9, 11, 13, 18]; label='shapes' #2 and 18 excluded
# to_use = [1, 3, 5, 6, 8, 10, 12, 14, 15, 16, 17]; label='non_shapes'
# to_use = list(range(1, 19)); label='all'


alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
RTexp1 = {f'array{alph[k]}': v for k, v in enumerate([2.40, 1.45, 1.71, 2.95, 2.45, 3.52])}# 3.49, 2.09, 2.40, 2.50])}
RTexp2 = {f'array{k + 1}': v / 1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}
RT = {}
RT.update(RTexp2)
RT.update(RTexp1)
plt.close('all')
type_ds = [f'array{i}' for i in range(1, 19)]
type_ds.extend(['arrayA', 'arrayB', 'arrayC', 'arrayD', 'arrayE', 'arrayF'])

plt.close('all')

transf = 'none'
bk = 'random'
depth_layer = 'last_conv_l'


##
## Correlation one network at the time with Spearmans' R
from scipy.stats import spearmanr
plt.close('all')
import seaborn as sns
sns.set(style="white")
tidx = 0
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
color_cycle = np.vstack([color_cycle, color_cycle, color_cycle])
fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)), figsize=[6.78, 7.94], sharex='col', sharey=True)
ax = ax.flatten()
df = pd.DataFrame()
tt = np.array(list(type_ds))
np.random.shuffle(tt)
text_values = np.split(tt, 6)
depth_to_consider = ['last_l', 'last_conv_l']
for idx, net in enumerate(network_names):
    m = {type: [] for type in type_ds}
    for d in depth_to_consider:
        all_cossim = {type: collect(pretraining='ImageNet', network_name=net, type_ds=type, background=bk, transf_code=transf, depth_layer=d) for type in type_ds}
        [m[tt].append(np.mean(all_cossim[tt])) for tt in type_ds]
        # s = {type: np.std(all_cossim[type]) for type in RT.keys()}
        for tt in type_ds:
            df = df.add_record({'net': net, 'type': tt,  f'cossim_d{d}': m[tt][depth_to_consider.index(d)]})

    [ax[idx].plot(m[type], [RT[type], RT[type]], '-', linewidth=1, color=color_cycle[cidx]) for cidx,type in enumerate(type_ds)]
    [ax[idx].plot(m[type][1], RT[type], 'o', linestyle='', markeredgecolor='k', color=color_cycle[cidx]) for cidx, type in enumerate(type_ds)]
    [ax[idx].plot(m[type][0], RT[type], 'o', linestyle='', markerfacecolor='none', markeredgecolor=color_cycle[cidx]) for cidx, type in enumerate(type_ds)]
    ax[idx].set_ylim([0.55, 2.8])


    ax[idx].annotate(from_netname_to_str(net),
                xy=(0.05, 0.95), xycoords='axes fraction',
                textcoords='offset points',
                size=12, weight='bold',
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))


    if idx == 0:
        plt.rcParams["font.size"] = "10"
        plt.rcParams["font.weight"] = "bold"
        text = [ax[idx].text(m[txt][0], RT[txt], re.findall("array([\w]+)", txt)[0]) for txt in list(type_ds)]
        adjust_text(text, ax= ax[idx])
        plt.rcParams["font.weight"] = "normal"


fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.xlabel('CNNs cosine similarity')
plt.ylabel('human RTs (sec)')
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='o', markersize=8,  markeredgecolor='k', markerfacecolor='none'),
                Line2D([0], [0],  linestyle='', marker='o', markersize=8, markeredgecolor='k', markerfacecolor='b')]

leg2 = plt.legend(custom_lines, ['Last Conv. Layer', 'Last (Fully Conn.) Layer'], prop={'size': 12}, ncol=2, edgecolor='k', bbox_to_anchor=(0.0, 1.1), loc="upper left")
leg2.get_frame().set_linewidth(1.5)


# plt.title(f'Depth: {from_depth_to_string(depth_layer)}')
plt.tight_layout()
plt.show()
plt.savefig(f'./results/figures/single_figs/correlation_multilayer_T{transf}_bk{bk}_{label}.svg')
plt.savefig(f'./results/figures/pngs/correlation_multilayer_T{transf}_bk{bk}_{label}.png')


## Difference between layers and relative barplot
df['diff'] = df['cossim_d5'] - df['cossim_d6']
df.groupby('type')['diff'].mean().sort_values()
df.groupby('type')['diff'].std()


def plot_bars(x, m):
    span = 0.7
    width = span / (len(m) + 2 - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = plt.bar(x - span / 2 + width * i, m, width,  label=network_names, color=color_cycle[:len(m)])
plt.figure(2)
[plot_bars(idx, df[df['type'] == type]['diff']) for idx, type in enumerate(RT.keys())]

diff = {tt: m[tt][1] - m[tt][0] for tt in RT.keys()}
sorted(diff.items(), key=lambda x: x[1])


##
##

