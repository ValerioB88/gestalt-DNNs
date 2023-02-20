import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
import seaborn as sns
from src.utils.net_utils import get_layer_from_depth_str

sns.set(style="white")

def compute_base_comp(base, comp, net):
    pt = 'kitti' if net == 'prednet' else 'ImageNet'
    base_dist = collect(pretraining=pt, network_name=net, type_ds=base)
    composite_dist = collect(pretraining=pt, network_name=net, type_ds=comp)
    if distance_type == 'cossim':
        CSE = base_dist - composite_dist
    elif distance_type == 'euclidean':
        CSE = (composite_dist - base_dist)/(composite_dist + base_dist)
    m = np.mean(CSE)
    s = np.std(CSE)
    return m, s, CSE

def collect(pretraining, network_name, type_ds):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type=distance_type,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    type_ds=type_ds,
                    background=bk,
                    draw_obj=DrawShape(background='black' if bk == 'black' or bk == 'random' else bk, img_size=img_size, width=10),
                    transf_code=transf)

    exp_folder = f'./results//{config_to_path_special(config)}'
    cs = pickle.load(open(exp_folder + f'_{distance_type}.df', 'rb'))
    all_layers = list(cs[type_ds].keys())

    ll = get_layer_from_depth_str(all_layers, depth_layer)

    cs_layer = cs[type_ds][ll] #np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    return np.array(cs_layer)


##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = [(0/255, 204/255, 255/255), (255/255, 127/255, 42/255)]
pretraining = ['ImageNet']

############ ARGUMENTS #########
transf = 't'
bk = 'random'
network_names = brain_score_nn.keys()
distance_type = 'euclidean'
depth_layer = 'last_l'
#################################

type_ds = [f'array{i}' for i in range(1, 19)]


def plot_sets(cse_pairs):
    fig, ax = plt.subplots(1, 1, figsize=[8.5*1.7,  5.5*0.9], sharex='col', sharey=True)
    ax.grid(axis='y')

    def plot_bar(m, s, distr, x):
        span = 0.4
        space = 0.03
        width = span / (len(m))
        i = np.arange(0, len(m))
        # plt.figure()
        for iidx, n in enumerate(range(len(distr))):
            bp = ax.boxplot(distr[iidx], patch_artist=True, showfliers=False, positions=[x - span / 2 + width * i[iidx]], widths=width - space, labels=[n], boxprops={'fill': (0, 0, 0, 1), 'lw': 0.5, 'facecolor': color_cycle[iidx], 'alpha': 1}, medianprops={'color': 'k', 'linestyle': '-', 'linewidth': 0.5, 'zorder': 2})
            if bp['medians'][0].get_data()[1][0] < 0:
                bp['boxes'][0].set(hatch='////', facecolor='w', edgecolor=color_cycle[iidx])
                bp['medians'][0].set(linewidth=1)

    net_plotted = []
    net_CSE_m = {}
    net_CSE_s = {}
    net_CSE_distr = {}
    for idx, net in enumerate(network_names):
        net_CSE_m[net] = {f'{p[0]} - {p[1]}': compute_base_comp(p[0], p[1], net)[0] for p in cse_pairs}
        net_CSE_s[net] = {fr'{p[0]} - {p[1]}': compute_base_comp(p[0], p[1], net)[1] for p in cse_pairs}
        net_CSE_distr[net] = {fr'{p[0]} - {p[1]}': compute_base_comp(p[0], p[1], net)[2] for p in cse_pairs}

        if np.median(list(net_CSE_distr[net].values())[0]) > 0:
            net_plotted.append(net)

    for idx, net in enumerate(net_plotted):
        plot_bar(list(net_CSE_m[net].values()), list(net_CSE_s[net].values()), list(net_CSE_distr[net].values()), idx)
        plt.gca().annotate(from_netname_to_str(net),
                 color='orangered' if net in recurrent else 'k',
                 xy=(idx/len(net_plotted)+0.025, 0.1), xycoords='axes fraction',
                 textcoords='offset points',
                 size=15, fontweight='heavy' if net in self_superv_net else 'normal',
                 # fontstyle='italic' if nn in recurrent else 'normal',
                 bbox=dict(boxstyle="round", alpha=1, fc='lightblue' if net in transformers else 'wheat', ec="none"))


    plt.xticks(range(len(net_plotted)), ['']*len(net_plotted))

    plt.yticks([-0.2, 0, 0.2, 0.4], fontsize=20)
    plt.ylabel('Networks CE (Euclidedan)', fontsize=20)

    plt.tight_layout()

plt.close('all')
cse_pairs = [
    ['array3', 'array4'],
    ['array3', 'array4_curly'],
    ]

plot_sets(cse_pairs)
# plt.ylim([-0.05, 0.35])
plt.savefig(f'./figures/single_figs/barplot_set3-4.svg')
plt.show()

##
cse_pairs = [
    ['array10', 'array11'],
    ['array10', 'array11_curly'],
    ]

plot_sets(cse_pairs)
# plt.ylim([-0.05, 0.35])
plt.savefig(f'./figures/single_figs/barplot_set10-11.svg')
plt.show()
##
cse_pairs = [
    ['arrayA', 'arrayB'],
    ['arrayA', 'curly_composite_with_space'],
    ]
plot_sets(cse_pairs)
# plt.ylim([-0.05, 0.2])
plt.savefig(f'./figures/single_figs/barplot_setA-Bcurly.svg')
plt.show()
