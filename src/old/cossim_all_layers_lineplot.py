import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.utils.Config import Config
from src.utils.misc import main_text_nets, all_nets, appendix_nets, config_to_path_hierarchical
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
    all_layers = cs[type].keys()

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))

    for idx, type in enumerate(all_l_m.keys()):
        ax.plot(all_l_m[type], label=type, lw=2, color=color_cycle[idx])
        plt.fill_between(range(len(all_l_m[type])), all_l_m[type] - all_l_s[type], all_l_m[type] + all_l_s[type], alpha=0.2, color=color_cycle[idx])
        plt.grid(True)
        ax.set_xticks(np.arange(0, len(all_l_m[type])))
        ax.set_xticklabels(list(all_layers), rotation=90)
        plt.ylabel('Average Cosine Similarity')
        idx = [idx for idx, i in enumerate(cs[type].keys()) if 'Linear' in i][0]
        plt.axvline(idx, color='r', ls='--')
        # plt.ylim([plt.ylim()[0], 1])
    plt.ylim([0, 1])
        # plt.show()
    ax.legend()
    ax.set_title(config_to_path_hierarchical(config))
    plt.savefig(os.path.dirname(exp_folder) + '/all-nets.png')



##

############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = main_text_nets
#################################
background = ['random'] # ['black', 'white', 'random']# 'white-on-black', ]
all_exps = (product(network_names, background))
arguments = list((dict(network_name=i[0],  background=i[1]) for i in all_exps))

[plot(**a) for a in arguments]
