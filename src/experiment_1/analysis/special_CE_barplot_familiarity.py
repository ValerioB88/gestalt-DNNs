import pickle
from src.utils.create_stimuli.drawing_utils import *
from src.utils.Config import Config
from src.utils.misc import *
color_cycle = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns
from src.utils.net_utils import get_layer_from_depth_str

sns.set(style="white")

def compute_base_comp(base, comp, net, transf_code, depth_layer):
    base_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=base, background=bk, transf_code=transf_code, depth_layer=depth_layer)
    composite_cossim = collect(pretraining='ImageNet', network_name=net, type_ds=comp, background=bk, transf_code=transf_code, depth_layer=depth_layer)

    m = -np.mean(composite_cossim - base_cossim)
    s = np.std(composite_cossim - base_cossim)
    return m, s

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

pretraining = ['ImageNet']
network_names = ['alexnet', 'vgg19bn', 'resnet152', 'inception_v3', 'densenet201', 'cornet-s', 'vonenet-cornets', 'vonenet-resnet50']

type_ds = [f'array{i}' for i in range(1, 19)]

RT = {f'array{k}': v for k, v in enumerate([1775, 896, 2139, 759, 2582, 793, 733, 1228, 1583, 1884, 749, 2020, 2022, 914, 908, 993, 989, 724])}

##
transf = 't'
bk = 'random'
depth_layer = 'last_conv_l'

def plot_sets(cse_pairs):
    fig, ax = plt.subplots(1, 1, figsize=[10.16,  5.92], sharex='col', sharey=True)
    def plot_bar(m, s, x):
        span = 0.7
        width = span / (len(m))
        i = np.arange(0, len(m))
        # plt.figure()
        rect = ax.bar(x - span / 2 + width * i, m, width, yerr=s, label=network_names, color=color_cycle[:len(m)])


    for idx, net in enumerate(network_names):
        net_CSE_m = {f'{p[0]} - {p[1]}': compute_base_comp(p[0], p[1], net, transf, depth_layer)[0] for p in cse_pairs}
        net_CSE_s = {fr'{p[0]} - {p[1]}': compute_base_comp(p[0], p[1], net, transf, depth_layer)[1] for p in cse_pairs}
        plot_bar(list(net_CSE_m.values()), list(net_CSE_s.values()), idx)

    plt.axhline(0, linestyle='-', linewidth=1.5, color='k')
    # plt.title(f'Depth Layer {from_depth_to_string(depth_layer)}')
    plt.ylim([-0.05, 0.35])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [from_netname_to_str(nn) for nn in network_names], fontsize=13)
    plt.yticks([0, 0.1, 0.2, 0.3], fontsize=20)
    plt.ylabel('Networks CE', fontsize=20)
    plt.tight_layout()
    plt.show()

plt.close('all')
cse_pairs = [
    ['array3', 'array4'],
    ['array3', 'array4_curly'],
    ]

plot_sets(cse_pairs)
plt.ylim([-0.05, 0.35])
plt.savefig(f'./results/figures/single_figs/barplot_set3-4.svg')

##
cse_pairs = [
    ['array10', 'array11'],
    ['array10', 'array11_curly'],
    ]

plot_sets(cse_pairs)
plt.ylim([-0.05, 0.35])
plt.savefig(f'./results/figures/single_figs/barplot_set10-11.svg')
##
cse_pairs = [
    ['arrayA', 'arrayB'],
    ['arrayA', 'curly_composite_with_space'],
    ]
plot_sets(cse_pairs)
plt.ylim([-0.05, 0.2])
plt.savefig(f'./results/figures/single_figs/barplot_setA-Bcurly.svg')



## Agreement
