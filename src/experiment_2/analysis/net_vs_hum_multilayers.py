import pickle
import matplotlib.patches as patches
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str

color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns

sns.set(style="white")
from scipy.stats import ttest_1samp
from scipy.stats import spearmanr, kendalltau


def compute_base_comp(base, comp, net):
    img_size = np.array((224, 224), dtype=int)
    pt = 'kitti' if net == 'prednet' else pretraining

    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type=distance_type,
                    network_name=net,
                    pretraining=pt,
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,

                    background=bk,
                    draw_obj=DrawShape(background='black' if bk == 'black' or bk == 'random' else bk, img_size=img_size, width=10),
                    transf_code=transf)

    exp_folder_norm = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder_norm + f'{distance_type}.df', 'rb'))
    all_layers = list(cs['empty'].keys())
    ll = get_layer_from_depth_str(all_layers, depth_layer)
    normalization_factor = np.mean(cs['empty'][ll] - np.array(cs['empty-single'][ll]))

    base_cossim = collect(config, type_ds=base)
    composite_cossim = collect(config, type_ds=comp)
    if distance_type == 'cossim':
        CE = base_cossim - composite_cossim
    elif distance_type == 'euclidean':
        CE = (composite_cossim - base_cossim) / (composite_cossim + base_cossim)
    if normalize:
        CE = CE / normalization_factor
    ttest = ttest_1samp(CE, 0)
    m = np.mean(CE)
    s = np.std(CE)
    return m, s, ttest.pvalue


def collect(config, type_ds):
    config.type_ds = type_ds

    #

    exp_folder = f'./results//{config_to_path_special(config)}'
    cs = pickle.load(open(exp_folder + f'_{distance_type}.df', 'rb'))
    all_layers = list(cs[type_ds].keys())

    ll = get_layer_from_depth_str(all_layers, depth_layer)

    cs_layer = cs[type_ds][ll]
    return np.array(cs_layer)


############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = brain_score_nn.keys()
normalize = False
distance_type = 'euclidean'
#################################

type_ds = [f'array{i}' for i in range(1, 25)]

fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)), sharex='col', sharey=True, figsize=[7.3, 9.62])  # , figsize=[8.73,  9.47])
ax = ax.flatten()


for idx, net in enumerate(network_names):
    sel_net = []
    sel_net_std = []
    if network_names == 'prednet':
        d_considered = ['last_conv_l']
    else:
        d_considered = ['last_l']
    for depth_layer in d_considered:
        net_CSE = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: compute_base_comp(p[0], p[1], net)[0] for p in all_pairs}
        net_CSE_std = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: compute_base_comp(p[0], p[1], net)[1] for p in all_pairs}

        {k: v for k, v in sorted(net_CSE.items(), key=lambda item: item[1], reverse=True)}
        selected = [(net_CSE[type], human_CSE_exp2[type], net_CSE_std[type]) for type in human_CSE_exp2]
        sel_net.append(np.array([i[0] for i in selected]))
        sel_net_std.append(np.array([i[2] for i in selected]))
        sel_human = [i[1] for i in selected]

    sel_net = np.array(sel_net)
    sel_net_std = np.array(sel_net_std)

    col = ['orange', 'dodgerblue', ]
    markers = ['o', '^']

    for ll, i in enumerate(d_considered):
        [ax[idx].plot([sel_net[ll][iidx] - sel_net_std[ll][iidx], sel_net[ll][iidx] + sel_net_std[ll][iidx]],
                      [sel_human[iidx], sel_human[iidx]], '-', markerfacecolor='none', color=col[ll]) for iidx, _ in enumerate(sel_human)]
        [ax[idx].plot(sel_net[ll][iidx], sel_human[iidx], markers[ll], color=col[ll], markeredgecolor='k') for iidx, _ in enumerate(sel_human)]

    agreement = np.mean(np.sign(sel_net[0]) == np.sign(sel_human)) * 100

    ax[idx].annotate(from_netname_to_str(net),
                     color='orangered' if net in recurrent else 'k',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     textcoords='offset points',
                     size=10, fontweight='heavy' if net in self_superv_net else 'normal',
                     # fontstyle='italic' if nn in recurrent else 'normal',
                     bbox=dict(boxstyle="round", fc='lightblue' if net in transformers else 'wheat', ec="none"))

    r = kendalltau(sel_net[0], sel_human)

    ax[idx].annotate(rf'$r_s = {r.correlation:.02f}$' + '\n' + rf'$p={r.pvalue:.3f}$',
                     xy=(0.05, 0.7), xycoords='axes fraction',
                     textcoords='offset points',
                     size=8,
                     bbox=dict(boxstyle="round",
                               fc=(1, 1, 1, 0.7), ec="none"))

    ax[idx].axvline(0, linestyle='--', color='k')
    ax[idx].axhline(0, linestyle='--', color='k')

    lims = [ax[idx].get_xlim(), ax[idx].get_ylim()]
    plt.rcParams.update({'hatch.color': 'k'})
    p = patches.Rectangle((0, 0), 3, 3, facecolor='none', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.2), hatch='///')
    ax[idx].add_patch(p)

    p = patches.Rectangle((-3, -3), 3, 3, facecolor='k', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.2), hatch='///')
    ax[idx].add_patch(p)
    plt.autoscale(True)
if normalize == False:
    ax[idx].set_ylim([-1.5, 1.8])
ax[idx].set_xlim([-0.4, 1])
ax[0].set_xlim([-0.5, 0.5])
ax[1].set_xlim([-0.5, 0.5])
ax[2].set_xlim([-0.5, 0.5])
ax[3].set_xlim([-0.5, 0.5])

fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], linestyle='', marker='o', markersize=8, markeredgecolor='k', markerfacecolor='none'),
                Line2D([0], [0], linestyle='', marker='o', markersize=8, markeredgecolor='k', markerfacecolor='b'),
                patches.Rectangle((0, 0), 3, 3, facecolor='none', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.4), hatch='////')]

leg2 = plt.legend(custom_lines, ['Middle-Stage Layer', 'Last (Fully Conn.) Layer', 'Agreement Area'], prop={'size': 12}, ncol=3, edgecolor='k', bbox_to_anchor=(0.5, 1.1), loc="center")
leg2.get_frame().set_linewidth(1.5)
plt.xlabel('Networks CE (Euclidean)')
plt.ylabel('Humans CE')
plt.savefig(f'./figures/single_figs/CE_multilayer_T{transf}_bk{bk}.svg')
plt.show()
