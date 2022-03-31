import os
import pickle
from itertools import product
from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
from src.utils.create_stimuli.drawing_utils import *

from src.utils.Config import Config
from src.utils.misc import *

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns
sns.set(style="darkgrid")
from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str

def plot_net_set(m, s, x):
    span = 0.6
    width = span / (len(m) - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])

def compute_base_comp(base, comp, net, transf_code, depth_layer):
    base_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=base, background=bk, transf_code=transf_code, depth_layer=depth_layer)
    composite_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=comp, background=bk, transf_code=transf_code, depth_layer=depth_layer)

    m = -np.mean(composite_cossim - base_cossim)
    s = np.std(composite_cossim - base_cossim)
    return m, s


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

    cs_layer = cs[type_ds][ll] #np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    return np.array(cs_layer)



##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

type_ds = [f'array{i}' for i in range(1, 19)]

RT = {f'array{k}': v for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}

##
transf = 'none'
bk = 'random'
depth_layer = 'last_l'

plt.close('all')
all_pairs = [['array1', 'array2'],
             ['array3', 'array4'],
             ['array3', 'array5'],
             # ['array4', 'array5'], #
             ['array6', 'array7'],
             ['array6', 'array8'],
             ['array6', 'array9'],
             # ['array7', 'array8'],  #
             # ['array7', 'array9'],  #
             # ['array8', 'array9'],  #
             ['array10', 'array11'],
             ['array10', 'array12'],
             ['array10', 'array13'],
             # ['array11', 'array12'],  #
             # ['array11', 'array13'],  #
             # ['array12', 'array13'],  #
             ['array14', 'array15'],
             ['array14', 'array16'],
             ['array14', 'array17'],
             # ['array15', 'array16'],  #
             # ['array15', 'array17'],  #
             # ['array16', 'array17'],  #
             ]
RT = {f'array{k + 1}': v/1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}
from scipy.stats import spearmanr
import re
human_CSE = {fr'{re.search("[0-9]+", p[0]).group()} - {re.search("[0-9]+", p[1]).group()}' : RT[p[0]]- RT[p[1]] for p in all_pairs}
human_CSE = {k: h for k, h in human_CSE.items() if np.abs(h) > 0.4} ## LIMIT TO STRONG CSE

all_comb = [f'{re.search("[0-9]+", p[0]).group()} - {re.search("[0-9]+", p[1]).group()}' for p in all_pairs]
fig, ax = plt.subplots(2, int(np.ceil(len(network_names) / 2)), figsize=[14.26,  5.09], sharex='col', sharey=True)
ax = ax.flatten()

for idx, net in enumerate(network_names):

    net_CSE = {fr'{re.search("[0-9]+", p[0]).group()} - {re.search("[0-9]+", p[1]).group()}' : compute_base_comp(p[0], p[1], net, transf, depth_layer)[0] for p in all_pairs}
    corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])

    ax[idx].plot(corr[:, 0], corr[:, 1], 'o', color=color_cycle[idx], label=net)
    ax[idx].annotate(from_netname_to_str(net),
                     xy=(0.05, 0.6), xycoords='axes fraction',
                     textcoords='offset points',
                     size=10,
                     bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
    plt.rcParams["font.size"] = "10"
    text = [ax[idx].text(net_CSE[txt], human_CSE[txt], txt) for txt in all_comb if txt in human_CSE]
    adjust_text(text, ax=ax[idx])

    r = spearmanr(corr[:, 0], corr[:, 1])
    ax[idx].annotate(rf'$r_s : {r[0]:.02f}, p={r[1]:.3f}$',
                     xy=(0.05, 0.45), xycoords='axes fraction',
                     textcoords='offset points',
                     size=10,
                     bbox=dict(boxstyle="round", fc=(1, 1, 1), ec="none"))
    ax[idx].axvline(0, linestyle='--')
    ax[idx].axhline(0, linestyle='--')

fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.xlabel('networks CE')
plt.ylabel('humans CE')
plt.title(f'Depth Layer {from_depth_to_string(depth_layer)}')
plt.tight_layout()
plt.show()
plt.savefig(f'./results/figures/single_figs/spearmanCSE_T{transf}_bk{bk}_depth{depth_layer}.svg')
plt.savefig(f'./results/figures/pngs/spearmanCSE_T{transf}_bk{bk}_depth{depth_layer}.png', dpi=300)


## Agreement

agreement = {}
for idx, net in enumerate(network_names):

    net_CSE = {fr'{re.search("[0-9]+", p[0]).group()} - {re.search("[0-9]+", p[1]).group()}' : compute_base_comp(p[0], p[1], net, transf, depth_layer)[0] for p in all_pairs}
    corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])
    agreement[net] = np.mean(np.sign(corr[:,0]) == np.sign(corr[:, 1]))





ax.set_title(f'Transformation Applied: {transf}, Layer Depth: {depth_layer}')
plt.savefig(f'./results/figures/single_figs/special_CSE_T{transf}_bk{bk}_d{depth_layer}.svg')
plt.savefig(f'./results/figures/pngs/special_CSE_T{transf}_bk{bk}_d{depth_layer}.png')


##

