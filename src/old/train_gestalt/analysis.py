import csv
import pandas as pd
import matplotlib.pyplot as plt; import numpy as np
import seaborn as sns
from src.utils.misc import *
plt.close('all')
sns.set(style="white")
import os
# all_nets = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']# 'dino_vits8', 'dino_vits8', 'simCLR_resnet18_stl10', 'prednet-train-sup']
network_names = brain_score_nn.keys()
types = ['Proximity', 'Linearity', 'Orientation']
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['olive', 'crimson', 'violet', 'magenta', 'indigo', 'turquoise'])

filename = './results/learning_EFs_dataset'
fig, all_ax = plt.subplots(1, 3, sharey=True, figsize=[8.5*2 , 4*0.5])#, figsize=[8.73,  9.47])

def plot_type(index, type, ax):
    all_nets = []
    for net in network_names:
        with open(filename + f'/{net}/{type}.csv') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=','))
            try:
                all_nets.append([net, *[float(i) for i in reader[-1]]])
            except ValueError:
                print(f"error with {net} and {type}")
                all_nets.append([net, 0, 0, 0])
                continue

    df = pd.DataFrame(all_nets, columns=['network_name', 'last_epoch', 'train_acc', 'test_acc'])
    df['color'] = color_cycle[:len(all_nets)]
    # df = df.sort_values(['test_acc'])
    df['network_name'] = [from_netname_to_str(n) for n in df['network_name']]
    idx = list(range(len((df['network_name']))))
    values = list(df['test_acc']*100)
    values_tr = list(df['train_acc']*100)
    if index < 2:
        ax.axhline(33, ls='--', color='red')
    else:
        ax.axhline(50, ls='--', color='red')

    for i in idx:
        ax.bar(idx[i], values[i], color=list(df['color'])[i])
        ax.plot(idx[i], values_tr[i], 'ko')
    ax.set_ylabel('Test Set Accuracy%', fontsize=15) if index == 0 else None
    ax.set_xticks(range(len((df['network_name']))))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, 25, 50, 75, 100], size=15)
    ax.set_xticklabels(df['network_name'], rotation=90, size=13)
    ax.set_title(type, fontsize=20)
    # plt.xlabel('Network Names')
    ax.legend()
    # ax.grid(axis='y')

[plot_type(idx, t, a) for idx, (t, a) in enumerate(zip(types, all_ax))]

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0],  linestyle='', marker='o', markersize=5, color='k', linewidth=1),
                Line2D([0], [0],  linestyle='--', marker='', markersize=5, color='r', markerfacecolor='w', markeredgecolor='k', linewidth=1)]
leg2 = plt.legend(custom_lines, ['Train set \naccuracy', 'Chance Level'], prop={'size': 15}, ncol=1, edgecolor='k', bbox_to_anchor=(0.0, 1.1), loc="best", framealpha=1)
plt.savefig(f'./figures/single_figs/train_gestalt.svg')
plt.show()
