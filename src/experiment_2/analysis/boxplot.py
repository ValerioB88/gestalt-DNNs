import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
from src.utils.net_utils import get_layer_from_depth_str
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns
sns.set(style="white")
from scipy.stats import ttest_1samp
from src.utils.misc import main_text_nets
plt.rcParams['svg.fonttype'] = 'none'


def compute_base_comp(base, comp, net):
    pt = 'kitti' if net == 'prednet' else 'ImageNet'
    base_cossim = collect(pretraining=pt, network_name=net, type_ds=base)
    composite_cossim = collect(pretraining=pt, network_name=net, type_ds=comp)


    if distance_type == 'cossim':
        CE = base_cossim - composite_cossim
    elif distance_type == 'euclidean':
        CE = (composite_cossim - base_cossim) / (composite_cossim + base_cossim)

    ttest_pv = ttest_1samp(CE, 0)
    CE_m = np.mean(CE)
    CE_std = np.std(CE)
    return CE_m, CE_std, ttest_pv.pvalue, CE

def collect(pretraining, network_name, type_ds):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    distance_type=distance_type,
                    verbose=False,
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

    cs_layer = cs[type_ds][ll]
    return np.array(cs_layer)


##

def set_boxplot():
    plt.close('all')


    def plot_net_set(m, s, x, distr, ax, **kwargs):
        span = 0.5
        width = span / (len(m) - 1)
        space = 0.02
        i = np.arange(0, len(m))
        # plt.figure()
        for iidx, n in enumerate(range(len(distr))):
            bp = ax.boxplot([distr[iidx]], patch_artist=True, showfliers=False, positions=[x - span / 2 + width * i[iidx]], widths=width - space, labels=[n], boxprops={'fill': (0, 0, 0, 1), 'lw': 0.5, 'facecolor': color_cycle[iidx], 'alpha': 1}, medianprops={'color': 'k', 'linestyle': '-', 'linewidth': 0.5, 'zorder': 2})
            # rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)], **kwargs)
            if bp['medians'][0].get_data()[1][0] < 0:
                bp['boxes'][0].set(hatch='////', facecolor='w', edgecolor=color_cycle[iidx])
                bp['medians'][0].set(linewidth=1)


    axx = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        AAAAAAA
        BBBBBCC
        """
    )


    plt.gcf().set_size_inches(9.5*1.1, 4*1.1)

    for idx, net in enumerate(network_names[:split]):
        iii = [[i for i in re.findall('([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', k)[0]] for k in interesting_sets]
        m, s, pv, distr = np.hsplit(np.array([compute_base_comp(f'array{ii[0]}', f'array{ii[1]}', net) for ii in iii]), 4)
        plot_net_set(m.flatten(), s.flatten(), idx, distr.flatten(), ax=axx['A'])

    for idx, net in enumerate(network_names[split+1:]):
        iii = [[i for i in re.findall('([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', k)[0]] for k in interesting_sets]
        m, s, pv, distr = np.hsplit(np.array([compute_base_comp(f'array{ii[0]}', f'array{ii[1]}', net) for ii in iii]), 4)
        plot_net_set(m.flatten(), s.flatten(), idx, distr.flatten(), ax=axx['B'])

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0],  linestyle='', marker='s', color=color_cycle[idx]) for idx,n in enumerate(network_names)]

    humanRT_interesting = [human_CSE_exp2[i] for i in interesting_sets]
    axx['A'].set_xticks(range(0, len(network_names[:split])))
    axx['B'].set_xticks(range(0, len(network_names[split+1:])))
    # axx['A'].set_xticklabels([from_netname_to_str(n) for n in network_names])
    axx['A'].set_xticklabels(['' for n in network_names[:split]],  horizontalalignment='center')
    axx['B'].set_xticklabels(['' for n in network_names[split+1:]],  horizontalalignment='center')

    for idx, nn in enumerate(network_names[:split]):
        axx['A'].annotate(from_netname_to_str(nn),
                          color='orangered' if nn in recurrent else 'k',
                          xy=(idx/len(network_names[:split]), 0.1 if idx%2 == 0 else -0.05), xycoords='axes fraction',
                          textcoords='offset points',
                          size=10, fontweight='heavy' if nn in self_superv_net else 'normal',
                          # fontstyle='italic' if nn in recurrent else 'normal',
                          bbox=dict(boxstyle="round", alpha=0.7,  fc='lightblue' if nn in transformers else 'wheat'))
    for idx, nn in enumerate(network_names[split+1:]):
        axx['B'].annotate(from_netname_to_str(nn),
                          color='orangered' if nn in recurrent else 'k',
                          xy=(idx / len(network_names[split+1:]), 0.1 if idx % 2 == 0 else -0.05), xycoords='axes fraction',
                          textcoords='offset points',
                          size=10, fontweight='heavy' if nn in self_superv_net else 'normal',
                          # fontstyle='italic' if nn in recurrent else 'normal',
                          bbox=dict(boxstyle="round", alpha=0.7, fc='lightblue' if nn in transformers else 'wheat'))

    axx['C'].plot(humanRT_interesting, 'k-')
    axx['C'].plot(np.array([np.arange(0, len(interesting_sets))]), np.array([humanRT_interesting]), 'o')
    axx['C'].set_ylabel('Humans CE (RTs)')

    if 'first5' in save_name:
        axx['A'].set_yticks([-0.3, 0, 0.3, 0.6])
        axx['B'].set_yticks([-0.3, 0, 0.3, 0.6])

        axx['C'].set_xticks(range(len(interesting_sets)))
        axx['C'].set_ylim([-0.2, None])
        axx['C'].set_xticklabels(['set 1'] + [f'{i}' for i in range(2, len(interesting_sets) + 1)] )  # <---
        axx['C'].set_yticks([0, 0.5, 1, 1.5])
    if 'last5' in save_name:
        axx['A'].set_yticks([-0.3, 0, 0.3])
        axx['B'].set_yticks([-0.3, 0, 0.3])

        axx['C'].set_ylim(-1.5, 0.2)
        axx['C'].set_yticks([0, -0.5, -1])
        axx['C'].set_xticks(range(len(interesting_sets)))
        axx['C'].set_xticklabels(['set 13'] + [f'{i}' for i in range(14, 17 + 1)])  # <---


    if distance_type == 'euclidean':
        axx['A'].set_ylabel('Networks CE (Euclidean)')
        axx['A'].set_xticks(range(0, len(network_names[:split])))
        axx['A'].set_xlim([-0.55, len(network_names[:split])-0.5])
        axx['B'].set_ylabel('Networks CE (Euclidean)')
        axx['B'].set_xticks(range(0, len(network_names[:split])))
        axx['B'].set_xlim([-0.55, len(network_names[split+1:]) - 0.5])


    if distance_type == 'cossim':
        if 'first5' in save_name:
            axx['A'].set_ylabel('Networks CE (Cosine Sim.)')
            axx['A'].set_yticks([0, 0.2, 0.4])
            axx['B'].set_ylabel('Networks CE (Cosine Sim.)')
            axx['B'].set_yticks([0, 0.1])

            axx['A'].set_ylim([-0.1, 0.5])
            axx['B'].set_ylim([-0.1, 0.15])

        if 'last5' in save_name:
            axx['A'].set_ylabel('Networks CE (Cosine Sim.)')
            axx['B'].set_ylabel('Networks CE (Cosine Sim.)')
            axx['A'].set_ylim([-0.35, 0.25])
            axx['B'].set_ylim([-0.15, 0.08])
            axx['A'].set_yticks([-0.2, 0, 0.2])
            axx['B'].set_yticks([-0.1, 0])

    axx['A'].grid(axis='y')
    axx['B'].grid(axis='y')
    axx['C'].grid(axis='y')


    plt.savefig(f'./figures/single_figs/{save_name}.svg')
    plt.show()

split = 8

############ EUCLIDEAN FIGURE 5 #########
# distance_type = 'euclidean'
# ### first 5 sets
# transf = 't'
# bk = 'random'
# network_names = list(brain_score_nn.keys())
# depth_layer = 'last_l'
# interesting_sets = list(human_CSE_exp2.keys())[:5]
# save_name = f'boxplot_{depth_layer}_{distance_type}_first5.svg'
# set_boxplot()
#
# ### last 5 sets
# interesting_sets = list(human_CSE_exp2.keys())[-5:]
# save_name = f'boxplot_{depth_layer}_{distance_type}_last5.svg'
# set_boxplot()

############ COSINE SIMILARITY FIGURE A.9 #########
distance_type = 'cossim'
### first 5 sets
transf = 't'
bk = 'random'
network_names = list(brain_score_nn.keys())
depth_layer = 'last_l'
interesting_sets = list(human_CSE_exp2.keys())[:5]
save_name = f'boxplot_{depth_layer}_{distance_type}_first5.svg'
set_boxplot()

### last 5 sets
interesting_sets = list(human_CSE_exp2.keys())[-5:]
save_name = f'boxplot_{depth_layer}_{distance_type}_last5.svg'
set_boxplot()

##

