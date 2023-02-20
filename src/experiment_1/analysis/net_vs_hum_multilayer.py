import pickle
from src.utils.Config import Config
from src.utils.misc import *

from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str
import seaborn as sns
sns.set(style="white")


def collect(network_name, type_ds, background, depth_layer):
    img_size = np.array((224, 224), dtype=int)
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type='cossim',
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    type_ds=type_ds,
                    background=background)
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + 'cossim.df', 'rb'))
    all_layers = list(cs[type_ds].keys())
    ll = get_layer_from_depth_str(all_layers, depth_layer)

    diff_penultimate = cs[type_ds][ll]
    return np.array(diff_penultimate)



############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = main_text_nets
#################################


def plot_net_set(m, s, x):
    span = 0.6
    width = span / (len(m) - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])

def compute_base_comp(base, comp, net, depth_layer):
    base_cossim = collect(network_name=net, type_ds=base, background=bk, depth_layer=depth_layer)
    composite_cossim = collect(network_name=net, type_ds=comp, background=bk, depth_layer=depth_layer)

    m = -np.mean(composite_cossim - base_cossim)
    s = np.std(composite_cossim - base_cossim)
    return m, s

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
plt.close('all')
all_pairs = [
             ['single', 'orientation'],
             ['single', 'proximity'],
             ['single', 'linearity'],
             ['empty', 'single']]

# RT = {f'array{k + 1}': v/1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}

RT = {'single': np.mean([1407, 1455, 1400, 1389])/1000,
      'orientation': np.mean([1430, 1139, 1138, 1297])/1000,
      'proximity': np.mean([925, 994, 1087, 1002])/1000,
      'linearity': np.mean([1016,  1021,  1064, 1075])/1000,
      'empty': np.mean([1])}
(RT['single'] - RT['orientation'])/RT['single']
(RT['single'] - RT['proximity'])/RT['single']
(RT['single'] - RT['linearity'])/RT['single']
(RT['single'] - RT['orientation'])
(RT['single'] - RT['proximity'])
(RT['single'] - RT['linearity'])

human_CSE = {f'{p[0][0]} - {p[1][0]}': RT[p[0]] - RT[p[1]] for p in all_pairs}
from matplotlib import patches
all_comb = [f'{p[0][0]} - {p[1][0]}' for p in all_pairs]
fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)), figsize=[7.38, 8.23], sharex=True, sharey=True)
ax = ax.flatten()


d_considered = ['last_conv_l', 'last_l']
for idx, net in enumerate(network_names):
    sel_net = []
    sel_net_std = []

    empty_single = []
    empty_single_std =  []

    for depth_layer in d_considered:
        net_CSE = {fr'{p[0][0]} - {p[1][0]}': compute_base_comp(p[0], p[1], net, depth_layer)[0] for p in all_pairs}
        net_CSE_std = {fr'{p[0][0]} - {p[1][0]}': compute_base_comp(p[0], p[1], net, depth_layer)[1] for p in all_pairs}

        selected = [(net_CSE[type], human_CSE[type], net_CSE_std[type]) for type in all_comb if type in human_CSE]
        sel_net.append(np.array([i[0] for i in selected]))
        sel_net_std.append(np.array([i[2] for i in selected]))
        sel_human = [i[1] for i in selected]

        corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])

        net_CSE = {fr'{p[0][0]} - {p[1][0]}': compute_base_comp(p[0], p[1], net, depth_layer)[0] for p in [['empty','empty-single']]}
        net_CSE_std = {fr'{p[0][0]} - {p[1][0]}': compute_base_comp(p[0], p[1], net, depth_layer)[1] for p in [['empty','empty-single']]}

        empty_single.append(net_CSE['e - e'])
        empty_single_std.append(net_CSE_std['e - e'])


    sel_net = np.array(sel_net)
    sel_net_std = np.array(sel_net_std)

    ax[idx].axvline(0, linestyle='--', color='k')
    ax[idx].axhline(0, linestyle='--', color='k')

    lims = [ax[idx].get_xlim(), ax[idx].get_ylim()]
    plt.rcParams.update({'hatch.color': 'k'})
    p = patches.Rectangle((0, 0), 3, 3, facecolor='none', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.2), hatch='///')
    ax[idx].add_patch(p)

    p = patches.Rectangle((-3, -3), 3, 3, facecolor='k', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.2), hatch='///')
    ax[idx].add_patch(p)
    ax[idx].set_xlim(*lims[0])
    ax[idx].set_ylim(*lims[1])
    plt.autoscale(True)

fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='o', markersize=8,  markeredgecolor='k', markerfacecolor='none'),
                Line2D([0], [0],  linestyle='', marker='o', markersize=8, markeredgecolor='k', markerfacecolor='b'),
                patches.Rectangle((0, 0), 3, 3, facecolor='none', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.4), hatch='////')]

leg2 = plt.legend(custom_lines, ['Last Conv. Layer', 'Last (Fully Conn.) Layer', 'Agreement Area'], prop={'size': 12}, ncol=3, edgecolor='k', bbox_to_anchor=(0.5, 1.1), loc="center")
plt.xlabel('Networks CE')
plt.ylabel('Humans CE (sec)')
plt.show()

plt.savefig(f'./results/figures/single_figs/hierarchical_multil_CSE_{bk}.svg')


