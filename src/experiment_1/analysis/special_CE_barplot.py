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

    cs_layer = cs[type_ds][ll]
    return np.array(cs_layer)


##
from adjustText import adjust_text
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

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
human_CSE = {k: v for k, v in sorted(human_CSE.items(), key=lambda item: item[1], reverse=True)}
all_comb = [re.findall("array([\w]+)", p[0])[0]  + '-' + re.findall("array([\w]+)", p[1])[0]  for p in all_pairs]
all_comb = [i for i in all_comb if i in human_CSE]

def plot_net_set(m, s, x, **kwargs):
    span = 0.6
    width = span / (len(m) - 1)
    i = np.arange(0, len(m))
    # plt.figure()
    rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)], **kwargs)


depth_layer = 'last_l'
fig, ax = plt.subplots(1, 1, figsize=[11.64, 4.42])
for idx, k in enumerate(list(human_CSE.keys())[:5]):
    ii = [i for i in re.findall('([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', k)[0]]
    m, s, pv = np.hsplit(np.array([compute_base_comp(f'array{ii[0]}', f'array{ii[1]}', net, transf, depth_layer) for net in network_names]), 3)
    plot_net_set(m.flatten(), s.flatten(), idx)

plt.axhline(0, color='k')
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0],  linestyle='', marker='s', color=color_cycle[idx]) for idx,n in enumerate(network_names)]
leg2 = plt.legend(custom_lines, [from_netname_to_str(n) for n in network_names], prop={'size': 10}, framealpha=1, ncol=1, facecolor='w', bbox_to_anchor=(0.0, 1.1), loc="upper left")
plt.tight_layout()
plt.xticks([])
plt.ylabel('Networks CSE')
plt.savefig(f'./results/figures/single_figs/barplotCSE_d{depth_layer}.svg')


