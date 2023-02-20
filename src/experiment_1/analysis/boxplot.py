import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.misc import *
from src.utils.Config import Config
from src.utils.misc import main_text_nets, all_nets, appendix_nets, self_superv_net, config_to_path_hierarchical

from src.utils.net_utils import from_depth_to_string, get_layer_from_depth_str
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'] + ['olive', 'crimson', 'violet', 'magenta', 'indigo', 'turquoise'])
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

def collect(network_name, type_ds, dist_type):
    img_size = np.array((224, 224), dtype=int)
    pt = 'kitti' if network_name == 'prednet' else 'ImageNet'

    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type=dist_type,
                    network_name=network_name,
                    pretraining=pt,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    background=bk,
                    draw_obj=DrawShape(background='black' if bk == 'black' or bk == 'random' else bk, img_size=img_size, width=14))
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    cs = pickle.load(open(exp_folder + f'{dist_type}.df', 'rb'))
    # all_layers = list(cs['empty'].keys())
    # diff_penultimate = np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])

    all_layers = list(cs['empty'].keys())
    ll = get_layer_from_depth_str(all_layers, depth_layer)

    from scipy.stats import ttest_1samp
    normalization_factor = np.mean(cs['empty'][ll] - np.array(cs['empty-single'][ll]))
    if type_ds == 'control':
        if dist_type == 'cossim':
            CE = cs['empty'][ll] - np.array(cs['empty-single'][ll])
        elif dist_type == 'euclidean':
            CE = (np.array(cs['empty-single'][ll]) - cs['empty'][ll])/(np.array(cs['empty-single'][ll]) + cs['empty'][ll])

    else:
        if dist_type == 'cossim':
            CE = cs['single'][ll] - np.array(cs[type_ds][ll])
        elif dist_type == 'euclidean':
            CE = (np.array(cs[type_ds][ll]) - cs['single'][ll])/(np.array(cs[type_ds][ll]) + cs['single'][ll])

    if normalize:
        CE = CE/normalization_factor
    ttest_pv = ttest_1samp(CE, 0).pvalue
    CE_m = np.mean(CE)
    CSE_std = np.std(CE)
    print(f'{network_name} {type_ds}, % < 0: {np.sum(CE < 0) / len(CE)}')
    return CE_m, CSE_std, ttest_pv, CE


def plot_net_set(m, s, x, pv, distr, ax):
    span = 0.35
    space = 0.03
    width = span / (len(m) - 1)
    i = np.arange(0, len(m))

    for iidx, n in  enumerate(range(len(distr))):
        bp = ax.boxplot(distr[iidx], patch_artist=True, showfliers=False, positions=[x - span / 2 + width * i[iidx]], widths=width - space, labels=[n], boxprops={'fill': (0, 0, 0, 1), 'lw': 0.5, 'facecolor': color_cycle[iidx], 'alpha': 1}, medianprops={'color':  'k', 'linestyle': '-', 'linewidth': 0.5, 'zorder': 2})
        if bp['medians'][0].get_data()[1][0] < 0:
            bp['boxes'][0].set(hatch='////', facecolor='w', edgecolor=color_cycle[iidx])
            bp['medians'][0].set(linewidth=1)


split = 8
def set_boxplot():
    plt.close('all')
    all_types_ds = ['proximity',  'linearity', 'orientation']
    axx = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        AAAAAAA
        BBBBBCC
        """
    )
    plt.gcf().set_size_inches(8.5*1.1, 5*1.1)

    for idx, net in enumerate(network_names[:split]):
        m, s, pv, distr = np.hsplit(np.array([collect(net, k, distance_type) for k in all_types_ds]), 4)
        plot_net_set(m.flatten(), s.flatten(), idx, pv.flatten(), distr.flatten(), axx['A'])

    for idx, net in enumerate(network_names[split+1:]):
        m, s, pv, distr = np.hsplit(np.array([collect(net, k, distance_type) for k in all_types_ds]), 4)
        plot_net_set(m.flatten(), s.flatten(), idx, pv.flatten(), distr.flatten(), axx['B'])

    axx['A'].set_xticks(range(0, len(network_names[:split])))
    axx['A'].set_xticklabels(['' for n in network_names[:split]])
    axx['B'].set_xticks(range(0, len(network_names[split+1:])))
    axx['B'].set_xticklabels(['' for n in network_names[split+1:]])

    for idx, nn in enumerate(network_names[:split]):
        axx['A'].annotate(from_netname_to_str(nn),
                          color='orangered' if nn in recurrent else 'k',
                          xy=(idx/len(network_names[:split]), 0.1 if idx%2 == 0 else -0.05), xycoords='axes fraction',
                          textcoords='offset points',
                          size=12, fontweight='heavy' if nn in self_superv_net else 'normal',
                          # fontstyle='italic' if nn in recurrent else 'normal',
                          bbox=dict(boxstyle="round", alpha=0.7,  fc='lightblue' if nn in transformers else 'wheat', ec="none"))
    for idx, nn in enumerate(network_names[split+1:]):
        axx['B'].annotate(from_netname_to_str(nn),
                          color='orangered' if nn in recurrent else 'k',
                          xy=(idx / len(network_names[split+1:]), 0.1 if idx % 2 == 0 else -0.05), xycoords='axes fraction',
                          textcoords='offset points',
                          size=12, fontweight='heavy' if nn in self_superv_net else 'normal',
                          # fontstyle='italic' if nn in recurrent else 'normal',
                          bbox=dict(boxstyle="round", alpha=0.7, fc='lightblue' if nn in transformers else 'wheat', ec="none"))

    axx['A'].set_xlim([-0.55, len(network_names[:split])-0.5])
    axx['B'].set_xlim([-0.55, len(network_names[split+1:])-0.5])

    if distance_type == 'euclidean':
        axx['A'].set_ylabel('Networks CE (Euclidean)')
        axx['A'].set_yticks([-0.3, 0, 0.3, 0.6])
        axx['B'].set_ylabel('Networks CE (Euclidean)')
        axx['B'].set_yticks([-0.3, 0, 0.3, 0.6])

    elif distance_type == 'cossim':
        axx['A'].set_ylabel('Networks CE (Cosine Sim.)}')
        axx['B'].set_ylabel('Networks CE (Cosine Sim.)}')
        axx['A'].set_yticks([-0.1, 0, 0.1, 0.2])
        axx['B'].set_yticks([-0.05, 0, 0.05])
        axx['B'].set_ylim([-0.1, 0.1])

    axx['A'].grid(axis='y')
    axx['B'].grid(axis='y')

    human_CSE = [human_CSE_exp1[i] for i in all_types_ds]
    axx['C'].plot(human_CSE, 'k-')
    axx['C'].plot(np.array([np.arange(0, 3)]), np.array([human_CSE]), 'o')
    axx['C'].set_ylabel('Humans CE (RTs)')

    axx['C'].set_ylim([-0.1, 0.5])
    axx['C'].set_xticks(range(3))
    axx['C'].set_xticklabels(all_types_ds, size=12)
    axx['C'].set_yticks([0, 0.2,  0.4])
    axx['C'].grid(axis='y')
    plt.tight_layout()

    def set_thickness(ax, x=0, thickness=2.6):
        horizontal_gridlines = ax.get_ygridlines()

        for line in horizontal_gridlines:
            if line.get_ydata()[0] == x:
                # Set the thickness of the gridline to 2 points
                line.set_linewidth(thickness)

    set_thickness(axx['A'])
    set_thickness(axx['B'])

    plt.savefig(f'./figures/single_figs/barplotCSE_dots_depth_{depth_layer}.svg')

    plt.show()

############ EUCLIDEAN FIGURE 3 #########
transf = 't'
bk = 'random'
network_names = list(brain_score_nn.keys())
depth_layer = 'last_l'
normalize = False
distance_type = 'euclidean'
set_boxplot()
#################################

############ COSINE SIMILARITY FIGURE A.8 #########
# transf = 't'
# bk = 'random'
# network_names = list(brain_score_nn.keys())
# depth_layer = 'last_l'
# normalize = False
# distance_type = 'cossim'
# set_boxplot()
#################################



