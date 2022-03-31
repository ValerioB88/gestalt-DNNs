import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from src.utils.Config import Config
from src.utils.misc import *
from src.utils.create_stimuli.drawing_utils import DrawShape

def run_cossim(network_name, pretraining, type_ds, background, transf_code, type_ds_args=None):
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
                    transf_code=transf_code,
                    type_ds_args=type_ds_args)

    exp_folder = f'./results//{config_to_path_special_par(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))[type_ds[0]]
    all_layers = list(cs.keys())
    return cs, all_layers

plt.close('all')
all_d = np.arange(0.15, 1.05, 0.05)
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']  # 'vonenet-resnet50-non-stoch'
fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)),  sharex=True, sharey=True)
ax = ax.flatten()
net = 'inception_v3'
pretraining = 'ImageNet'
for idx, net in enumerate(network_names):
    all_prox = []
    all_fake_prox = []

    for d in all_d:
        cs, all_l = run_cossim(net, pretraining, ('proximity', []), 'random', 'none', {'dist': d})
        all_prox.append(cs[all_l[-1]])
        cs, all_l = run_cossim(net, pretraining, ('fake_prox', []), 'random', 'none', {'dist': d})
        all_fake_prox.append(cs[all_l[-1]])
    ax[idx].plot(all_d, all_prox, 'o-', label='prox')
    ax[idx].plot(all_d, all_fake_prox, label='dist')
    ax[idx].legend()
    ax[idx].set_title(net)
    ax[idx].grid(True)
plt.show()
np.mean(np.array(all_prox).T)
np.mean(np.array(all_fake_prox).T)