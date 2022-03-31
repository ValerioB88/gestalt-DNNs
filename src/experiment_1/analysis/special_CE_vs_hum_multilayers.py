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
def compute_base_comp(base, comp, net, transf_code, depth_layer):
    base_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=base, background=bk, transf_code=transf_code, depth_layer=depth_layer)
    composite_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=comp, background=bk, transf_code=transf_code, depth_layer=depth_layer)

    CE = base_cossim - composite_cossim
    ttest = ttest_1samp(CE, 0)
    m = np.mean(CE)
    s = np.std(CE)
    return m, s, ttest.pvalue

def collect(pretraining, network_name, type_ds, background, transf_code, depth_layer):
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
                    transf_code=transf_code)

    exp_folder = f'./results//{config_to_path_special(config)}'
    cs = pickle.load(open(exp_folder + '_cossim.df', 'rb'))
    all_layers = list(cs[type_ds].keys())

    ll = get_layer_from_depth_str(all_layers, depth_layer)

    cs_layer = cs[type_ds][ll] #np.array(cs['base'][all_layers[-2]]) - np.array(cs['composite'][all_layers[-2]])
    return np.array(cs_layer)


##
from adjustText import adjust_text
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
# network_names = ['alexnet', 'inception_v3', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']  # 'vonenet-resnet50-non-stoch'
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']  # 'vonenet-resnet50-non-stoch'
# network_names = ['cornet-s']

type_ds = [f'array{i}' for i in range(1, 25)]
# type_ds.extend(['single-parentheses', 'parentheses-crossed', 'double-parentheses'])

transf_code = ['t', 's', 'r']
background = ['random', 'black', 'white']

##
transf = 't'
bk = 'random'
## STRONGEST CSE: 3-4, 10-11, A-B; STRONGEST CIE:  6-9, 6-8

plt.close('all')
all_pairs = [
             ['arrayA', 'arrayB'],
             ['arrayA', 'arrayC'],
             ['arrayA', 'arrayD'],
             ['arrayA', 'arrayE'],
             ['arrayA', 'arrayF'],
             ['array1', 'array2'],
             ['array3', 'array4'],
             ['array3', 'array5'],
             ['array6', 'array7'],
             ['array6', 'array8'],
             ['array6', 'array9'],
             ['array10', 'array11'],
             ['array10', 'array12'],
             ['array10', 'array13'],
             ['array14', 'array15'],
             ['array14', 'array16'],
             ['array14', 'array17']
             ]

#
# all_pairs = [
#              # ['arrayA', 'arrayB'],
#              ['arrayA', 'arrayC'],
#              ['arrayA', 'arrayD'],
#              ['arrayA', 'arrayE'],
#              ['arrayA', 'arrayF'],
#              ['array1', 'array2'],
#              # ['array3', 'array4'],
#              ['array3', 'array5'],
#              ['array6', 'array7'],
#              ['array6', 'array8'], #
#              ['array6', 'array9'], #
#              # ['array10', 'array11'],
#              ['array10', 'array12'],
#              ['array10', 'array13'],
#              ['array14', 'array15'],
#              ['array14', 'array16'],
#              ['array14', 'array17']
#              ]

alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
RTexp1 = {f'array{alph[k]}': v for k, v in enumerate([2.40, 1.45, 1.71, 2.95, 2.45, 3.52, 3.49, 2.09, 2.40, 2.50])}
RTexp2 = {f'array{k + 1}': v / 1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}
RT = {}
RT.update(RTexp2)
RT.update(RTexp1)
from scipy.stats import spearmanr
import re
human_CSE_perc = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: (RT[p[0]]-RT[p[1]])/RT[p[1]] for p in all_pairs}
human_CSE_perc = {k: v for k, v in sorted(human_CSE_perc.items(), key=lambda item: item[1], reverse=True)}


human_CSE = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: RT[p[0]] - RT[p[1]] for p in all_pairs}
# human_CSE = {k: h for k, h in human_CSE.items() if np.abs(h) > 0.4} ## LIMIT TO STRONG CSE
human_CSE = {k: v for k, v in sorted(human_CSE.items(), key=lambda item: item[1], reverse=True)}
all_comb = [re.findall("array([\w]+)", p[0])[0]  + '-' + re.findall("array([\w]+)", p[1])[0]  for p in all_pairs]
all_comb = [i for i in all_comb if i in human_CSE]
fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)),  sharex=True, sharey=True, figsize=[7.3 , 9.62])#, figsize=[8.73,  9.47])
ax = ax.flatten()



color_cycle = np.tile(np.array(plt.rcParams['axes.prop_cycle'].by_key()['color']), (3, 1))


for idx, net in enumerate(network_names):
    sel_net = []
    sel_net_std = []
    d_considered = ['last_conv_l', 'last_l']
    for depth_layer in d_considered:
        net_CSE = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0] : compute_base_comp(p[0], p[1], net, transf, depth_layer)[0] for p in all_pairs}
        net_CSE_std = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0] : compute_base_comp(p[0], p[1], net, transf, depth_layer)[1] for p in all_pairs}

        {k: v for k, v in sorted(net_CSE.items(), key=lambda item: item[1], reverse=True)}
        selected = [(net_CSE[type], human_CSE[type], net_CSE_std[type]) for type in human_CSE]
        sel_net.append(np.array([i[0] for i in selected]))
        sel_net_std.append(np.array([i[2] for i in selected]))
        sel_human = [i[1] for i in selected]

    sel_net = np.array(sel_net)
    sel_net_std = np.array(sel_net_std)

    col1 = 'dodgerblue'
    col2 = 'orange'


    ## Conv Layers

    [ax[idx].plot([sel_net[0][iidx] - sel_net_std[0][iidx], sel_net[0][iidx] + sel_net_std[0][iidx]],
                  [sel_human[iidx], sel_human[iidx]], '-', markerfacecolor='none', color=col1) for iidx, _ in enumerate(sel_human)]

    [ax[idx].plot(sel_net[0][iidx], sel_human[iidx], '^', color=col1, markeredgecolor='k') for iidx, _ in enumerate(sel_human)]

    ## FC Layers
    [ax[idx].plot([sel_net[1][iidx] - sel_net_std[1][iidx], sel_net[1][iidx] + sel_net_std[1][iidx]],
                  [sel_human[iidx], sel_human[iidx]], '-', markerfacecolor='none', color=col2) for iidx, _ in enumerate(sel_human)]
    [ax[idx].plot(sel_net[1][iidx], sel_human[iidx], 'o', color=col2, markeredgecolor='k') for iidx, _ in enumerate(sel_human)]

    # assert False


    plt.show()


    agreement = np.mean(np.sign(sel_net[1]) == np.sign(sel_human)) * 100
    ax[idx].annotate(f'A: {int(agreement)}%',
                     xy=(0.8, 0.1), xycoords='axes fraction',
                     textcoords='offset points',
                     size=8,
                     bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.7), ec="none"))
    ax[idx].annotate(from_netname_to_str(net),
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     textcoords='offset points',
                     size=12, weight='bold',
                     bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"))
    # if idx == 3:
    #     plt.rcParams["font.size"] = "10"
    #     plt.rcParams["font.weight"] = "bold"
    #     text = [ax[idx].text(sel_net[1][ii], sel_human[ii], txt) for ii, txt in enumerate(human_CSE.keys())]
    #     plt.rcParams["font.weight"] = "normal"
        # adjust_text(text, ax=ax[idx])
    from sklearn.metrics import r2_score
    r = spearmanr(sel_net[1], sel_human)
    ax[idx].annotate(rf'$r_s : {r[0]:.02f}, p={r[1]:.3f}$',
                     xy=(0.05, 0.7), xycoords='axes fraction',
                     textcoords='offset points',
                     size=8,
                     bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.7), ec="none"))

    # r = r2_score(sel_net[0], sel_human)
    # ax[idx].annotate(rf'$R^2 : {r:.02f}$',
    #                  xy=(0.05, 0.5), xycoords='axes fraction',
    #                  textcoords='offset points',
    #                  size=10,
    #                  bbox=dict(boxstyle="round", fc=(1, 1, 1), ec="none"))
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
ax[idx].set_xlim([-0.4, 0.35])
ax[idx].set_ylim([-1.5, 1.8])

fig = fig.add_subplot(111, frameon=False)
fig.grid(False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0],  linestyle='', marker='o', markersize=8,  markeredgecolor='k', markerfacecolor='none'),
                Line2D([0], [0],  linestyle='', marker='o', markersize=8, markeredgecolor='k', markerfacecolor='b'),
                patches.Rectangle((0, 0), 3, 3, facecolor='none', linewidth=0, fill=None, edgecolor=(0, 0, 1, 0.4), hatch='////')]

leg2 = plt.legend(custom_lines, ['Last Conv. Layer', 'Last (Fully Conn.) Layer', 'Agreement Area'], prop={'size': 12}, ncol=3, edgecolor='k', bbox_to_anchor=(0.5, 1.1), loc="center")
leg2.get_frame().set_linewidth(1.5)
plt.xlabel('Networks CE')
plt.ylabel('Humans CE')
# plt.title(f'Depth Layer {from_depth_to_string(depth_layer)}')
plt.tight_layout()
plt.show()
plt.savefig(f'./results/figures/single_figs/CE_multilayer_T{transf}_bk{bk}.svg')
plt.savefig(f'./results/figures/pngs/CE_multilayer_T{transf}_bk{bk}.png')


## Agreement
#
agreement = {}
for idx, net in enumerate(network_names):
    net_CSE = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: compute_base_comp(p[0], p[1], net, transf, depth_layer)[0] for p in all_pairs}
    corr = np.array([(net_CSE[type], human_CSE[type]) for type in all_comb if type in human_CSE])
    agreement[net] = np.mean(np.sign(corr[:,0]) == np.sign(corr[:, 1]))



