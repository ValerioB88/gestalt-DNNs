import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.utils.Config import Config
from src.utils.misc import main_text_nets, all_nets, appendix_nets, config_to_path_hierarchical
from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def plot(network_name='', background='rnd-pixels'):
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    all_l_m = {}
    all_l_s = {}


    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=background)
    exp_folder = f'./results///{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))
    for type, cs_t in cs.items():
        all_l_m[type] = np.array([np.mean(c) for k, c in cs_t.items()])
        all_l_s[type] = np.array([np.std(c) for k, c in cs_t.items()])
    all_layers = list(cs[type].keys())

    ll = get_layer_from_depth_str(all_layers, depth_layer)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ind = [1, 1.5, 2, 2.5, 3, 3.5]
    order = ['empty', 'empty-single', 'single', 'proximity', 'orientation', 'linearity']
    plt.bar(ind, [all_l_m[o][ll] for o in order], width=0.3,  yerr=[all_l_s[o][ll] for o in order], error_kw=dict(lw=3, capsize=6, capthick=3))
    ax.set_xticks(ind)
    ax.set_xticklabels(labels=order)
    ax.set_ylim([-0.1, 1])
    plt.title(ll)
    plt.grid(True)
    plt.ylabel('Average Cosine Similarity')
    plt.title(list(all_layers)[ll] + f'  {depth_layer}')
    plt.savefig(os.path.dirname(exp_folder) + '/barplot.png')



############ ARGUMENTS #########
transf = 't'
background = ['black', 'white', 'random']
depth_layer = 'last_l'
network_names = main_text_nets
#################################

# # 'white-on-black', ]
all_exps = (product(network_names, background))
arguments = list((dict(network_name=i[0],  background=i[1]) for i in all_exps))

[plot(**a) for a in arguments]
