import pickle
import matplotlib.patches as patches
from src.utils.create_stimuli.drawing_utils import *
from src.utils.net_utils import get_layer_from_depth_str
from src.utils.Config import Config
from src.utils.misc import *

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

    # DEPTH LAYER is 1: early, 2: early-middle, 3:middle. 4 is penultimate, 5 is last.
    # 6 is the last convolutional layer
    # 7 is the first linear layer
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
#
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


alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
RTexp1 = {f'array{alph[k]}': v for k, v in enumerate([2.40, 1.45, 1.71, 2.95, 2.45, 3.52, 3.49, 2.09, 2.40, 2.50])}
RTexp2 = {f'array{k + 1}': v / 1000 for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}
RT = {}
RT.update(RTexp2)
RT.update(RTexp1)
import re
human_CSE_perc = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: (RT[p[0]]-RT[p[1]])/RT[p[1]] for p in all_pairs}
human_CSE_perc = {k: v for k, v in sorted(human_CSE_perc.items(), key=lambda item: item[1], reverse=True)}


human_CSE = {re.findall("array([\w]+)", p[0])[0] + '-' + re.findall("array([\w]+)", p[1])[0]: RT[p[0]] - RT[p[1]] for p in all_pairs}
# human_CSE = {k: h for k, h in human_CSE.items() if np.abs(h) > 0.4} ## LIMIT TO STRONG CSE
human_CSE = {k: v for k, v in sorted(human_CSE.items(), key=lambda item: item[1], reverse=True)}
all_comb = [re.findall("array([\w]+)", p[0])[0]  + '-' + re.findall("array([\w]+)", p[1])[0]  for p in all_pairs]
all_comb = [i for i in all_comb if i in human_CSE]
fig, ax = plt.subplots(4, int(np.ceil(len(network_names) / 4)),  sharex=True, sharey=True, figsize=[7.3 , 9.62])
ax = ax.flatten()
# USE THIS FOR IMG IN SUPPL MAT.
fig, ax = plt.subplots(2, int(np.ceil(len(network_names) / 2)),  sharex='col', sharey='row', figsize=[11.64 , 4.42])
ax = ax.flatten()


color_cycle = np.tile(np.array(plt.rcParams['axes.prop_cycle'].by_key()['color']), (3, 1))


for idx, net in enumerate(network_names):
    sel_net = []
    sel_net_std = []
    d_considered = ['early', 'middle_early', 'middle']
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

    col = ['dodgerblue', 'orange', 'green']
    markers = ['^', 'o', 's']
    from matplotlib import colors

    ## Conv Layers
    for ll in range(len(sel_net)):
        [ax[idx].plot(sel_net[ll][iidx], sel_human[iidx], markers[ll], color=(*colors.to_rgb(col[ll]), 0.5), markeredgecolor='k') for iidx, _ in enumerate(sel_human)]


    plt.show()
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
ax[idx].set_xlim([-0.38, 0.05])
ax[idx].set_ylim([-1.5, 2])
ax[0].set_xlim([-0.21, 0.05])
ax[0].set_ylim([-1.41, 1.6])

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
plt.tight_layout()
plt.show()
plt.savefig(f'./results/figures/single_figs/suppl_mat_CE_multilayer_T{transf}_bk{bk}2.svg')
plt.savefig(f'./results/figures/pngs/suppl_mat_CE_multilayer_T{transf}_bk{bk}.png')


