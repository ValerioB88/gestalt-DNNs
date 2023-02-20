import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
import seaborn as sns


def compute_base_comp(type_ds, network_name):
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    distance_type=distance_type,
                    verbose=False,
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=bk,
                    draw_obj=DrawShape(background='black' if bk == 'black' or bk == 'random' else bk, img_size=img_size, width=10),
                    transf_code=transf_code)

    base_str, comp_str = type_ds.split('-')
    base = collect(config, 'array' + base_str)
    composite = collect(config, 'array' + comp_str)
    all_layers = list(base.keys())

    if distance_type == 'cossim':
        CE = [np.array(k)-np.array(v) for k, v in zip(base.values(), composite.values())]
    elif distance_type == 'euclidean':
        CE = [(np.array(k)-np.array(v))/(np.array(k)+np.array(v)) for k, v in zip(composite.values(), base.values())]


    ## Get normalizing factors
    if normalize:
        exp_folder = f'./results//{config_to_path_hierarchical(config)}'
        cs = pickle.load(open(exp_folder + f'{distance_type}.df', 'rb'))
        all_layers = list(cs['empty'].keys())
        normalization_factor = np.mean([np.array(cs['empty'][ll]) - np.array(cs['empty-single'][ll]) for ll in all_layers], axis=1)

        CE_m = np.array([np.mean(i)/normalization_factor for i in CE])
        CE_std = [0 for i in CE]  # not sure what's the right std to return here

    else:
        CE_m = np.array([np.mean(i) for i in CE])
        idx = [idx for idx, i in enumerate(CE_m) if not np.isnan(i)]
        CE_m = np.array([i for i in CE_m if not np.isnan(i)])
        CE_std = np.array([i for i in [np.std(i) for i in CE] if not np.isnan(i)])
    all_layers = [all_layers[i] for i in idx]

    return CE_m, CE_std, all_layers


def collect(config, type_ds):
    config.type_ds = type_ds
    exp_folder = f'./results//{config_to_path_special(config)}'
    cs = pickle.load(open(exp_folder + f'_{distance_type}.df', 'rb'))
    cs_layer = cs[type_ds]
    return cs_layer




############ ARGUMENTS #########
transf_code = 't'
bk = 'random'
network_names = brain_score_nn.keys()
normalize = False
type_ds = list(human_CSE_exp2.keys())[:5]
distance_type = 'euclidean'
#################################


##
sns.set(style="white")
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.close('all')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

axx = plt.figure(constrained_layout=False).subplot_mosaic(
    """
    AAAABBBBBCCC
    DDDEEEFFGGHH
    IIIJJJKKKLLL
    MMMNNNOOOPPP
    """, sharey=True
)

if distance_type == 'cossim':
    axx['I'].set_ylim([-0.25, 0.3])
    axx['I'].set_yticks([-0.1, 0.1])
elif distance_type == 'euclidean':
    axx['I'].set_ylim([-0.35, 0.35])
    axx['I'].set_yticks([-0.2, 0.2])

plt.gcf().set_size_inches(8.5*1.6, 5*1.6)
# ax = ax.flatten()
ax = list(axx.values())
[x.yaxis.set_tick_params(labelsize=13) for x in ax]
for axidx, nn in enumerate(network_names):
    m, s = {}, {}

    ax[axidx].axhline(0, color='k', linewidth=1, linestyle='--')
    for idx, type in enumerate(type_ds):
        m[type], s[type], all_layers = compute_base_comp(type_ds=type, network_name=nn)

        ax[axidx].plot(range(len(all_layers)), m[type], color=color_cycle[idx], linewidth=1)
        lin_l = [idx for idx, l in enumerate(all_layers) if 'Linear' in l]
        ax[axidx].axvline(lin_l[0], ls='--', color='r') if len(lin_l) > 0 else None
        ax[axidx].plot(len(m[type]) -1, m[type][-1], 'o', linewidth=2, color=color_cycle[idx])

        ax[axidx].fill_between(range(len(all_layers)), m[type] + s[type], m[type] - s[type], alpha=0.1, color=color_cycle[idx])
        ax[axidx].set_xticks([])
        ax[axidx].set_xticklabels([], rotation=90)

    ax[axidx].annotate(from_netname_to_str(nn),
                       color='orangered' if nn in recurrent else 'k',
                     xy=(0.1, 0.95), xycoords='axes fraction',
                     textcoords='offset points',
                     size=15, fontweight='heavy' if nn in self_superv_net else 'normal',
                       # fontstyle='italic' if nn in recurrent else 'normal',
                     bbox=dict(boxstyle="round", fc='lightblue' if nn in transformers else 'wheat', ec="none"))


plt.tight_layout()

plt.savefig(f'./figures/single_figs/specialCSE_multilayer_{bk}.svg')
plt.show()

##

