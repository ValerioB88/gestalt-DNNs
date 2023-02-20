import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
import seaborn as sns
from matplotlib.lines import Line2D


def collect(network_name):
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type=distance_type,
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=bk,
                    draw_obj=DrawShape(background='black' if bk == 'black' or bk == 'random' else bk, img_size=img_size, width=14))
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + f'{distance_type}.df', 'rb'))
    all_layers = list(cs['empty'].keys())

    normalization_factor = np.mean([np.array(cs['empty'][ll]) - np.array(cs['empty-single'][ll]) for ll in all_layers], axis=1)

    if distance_type == 'cossim':
        CE = {k: [np.array(cs['single'][ll]) - np.array(cs[k][ll])  for ll in all_layers]for k in ['orientation', 'proximity', 'linearity']}
        CE.update({k: [np.array(cs['empty'][ll] - np.array(cs[k][ll])) for ll in all_layers] for k in ['empty-single']})
    elif distance_type == 'euclidean':
        CE = {k: [(np.array(cs[k][ll]) - np.array(cs['single'][ll]))/(np.array(cs[k][ll]) + np.array(cs['single'][ll])) for ll in all_layers] for k in ['orientation', 'proximity', 'linearity']}
        CE.update({k: [(np.array(cs[k][ll]) - np.array(cs['empty'][ll]))/(np.array(cs[k][ll]) + np.array(cs['empty'][ll])) for ll in all_layers] for k in ['empty-single']})

    if normalize:
        CE_m = {k: np.mean(i, axis=1)/normalization_factor for k, i in CE.items()}
        CE_std = {k: 0 for k, i in CE.items()}

    else:
        CE_m = {k: np.array([j for j in np.mean(i, axis=1) if not np.isnan(j)]) for k, i in CE.items()}
        idx = {k: [idx for idx, j in enumerate(np.mean(i, axis=1)) if not np.isnan(j)] for k, i in CE.items()}
        CE_std = {k: np.array([j for j in np.std(i, axis=1) if not np.isnan(j)]) for k, i in CE.items()}
    all_layers = [all_layers[i] for i in list(idx.values())[0]]
    return CE_m, CE_std, all_layers


############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = list(brain_score_nn.keys())
normalize = False
distance_type = 'euclidean'
#################################

##
sns.set(style="white")
plt.close('all')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])


type_ds = ['empty-single', 'empty', 'single', 'proximity', 'orientation', 'linearity']
axx = plt.figure(constrained_layout=True).subplot_mosaic(
    """
    AAAABBBBBCCC
    DDDEEEFFGGHH
    IIIJJJKKKLLL
    MMMNNNOOOPPP
    """
)
# if distance_type == 'cossim':
axx['A'].set_ylim([-0.3, 0.8])
axx['B'].set_ylim([-0.3, 0.8])
axx['C'].set_ylim([-0.3, 0.8])

axx['D'].set_ylim([-0.3, 0.82])
axx['E'].set_ylim([-0.3, 0.82])
axx['F'].set_ylim([-0.3, 0.82])
axx['G'].set_ylim([-0.3, 0.82])
axx['H'].set_ylim([-0.3, 0.82])

axx['I'].set_ylim([-0.1, 0.7])
axx['J'].set_ylim([-0.1, 0.7])
axx['K'].set_ylim([-0.1, 0.7])
axx['L'].set_ylim([-0.1, 0.7])

axx['M'].set_ylim([-0.15, 0.7])
axx['N'].set_ylim([-0.15, 0.7])
axx['O'].set_ylim([-0.15, 0.7])
axx['P'].set_ylim([-0.15, 0.7])

[x.set_yticks([]) for x in list(axx.values())]
axx['A'].set_yticks([-0.25, 0.5])
axx['D'].set_yticks([-0.25, 0.5])
axx['I'].set_yticks([-0.1, 0.5])
axx['M'].set_yticks([-0.1, 0.5])

from matplotlib.pyplot import *
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'STIXGeneral:italic'
rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'

plt.gcf().set_size_inches(8.5*1.6, 5*1.6)
# ax = ax.flatten()
ax = list(axx.values())
[x.yaxis.set_tick_params(labelsize=13) for x in ax]
for axidx, nn in enumerate(network_names):
    m, s, all_layers = collect(network_name=nn)
    ax[axidx].axhline(0, color='k', linewidth=1, linestyle='--')
    for idx, type in enumerate(['empty-single', 'proximity', 'linearity', 'orientation']):
        ax[axidx].plot(range(len(all_layers)), m[type], color=color_cycle[idx], linewidth=1)
        lin_l = [idx for idx, l in enumerate(all_layers) if 'Linear' in l]

        ax[axidx].fill_between(range(len(all_layers)), m[type] + s[type], m[type] - s[type], alpha=0.1, color=color_cycle[idx])
        ax[axidx].set_xticks([])
        ax[axidx].set_xticklabels([], rotation=90)
        ax[axidx].axvline(lin_l[0], ls='--', color='r') if len(lin_l) > 0 else None
        ax[axidx].plot(len(m[type]) -1, m[type][-1], 'o', linewidth=2, color=color_cycle[idx])


    ax[axidx].annotate(from_netname_to_str(nn),
                       color='orangered' if nn in recurrent else 'k',
                       xy=(0.1, 0.95), xycoords='axes fraction',
                       textcoords='offset points',
                       size=15, fontweight='heavy' if nn in self_superv_net else 'normal',
                       # fontstyle='italic' if nn in recurrent else 'normal',
                       bbox=dict(boxstyle="round", fc='lightblue' if nn in transformers else 'wheat', ec="none"))

fig = plt.gcf().add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Network Depth", fontsize=18)
plt.ylabel(r"Network CE", fontsize=18)
[x.set_xlim([0, None]) for x in ax]

custom_lines = [Line2D([0], [0],  linestyle='-', marker='', color=color_cycle[idx]) for idx in [0, 1, 2, 3]] + \
               [Line2D([0], [0], linestyle='', marker='o', markerfacecolor='w', markeredgecolor='k')]

leg2 = plt.legend(custom_lines, ['Control', 'Proximity', 'Orientation', 'Linearity', 'Output Layer'], prop={'size': 12}, ncol=4, edgecolor='k', bbox_to_anchor=(0.0, 0), loc="upper left")
leg2.get_frame().set_linewidth(1.5)

plt.tight_layout()

plt.savefig(f'.//figures/single_figs/hierarchicalCSE_multilayer_{bk}.svg')
plt.show()

