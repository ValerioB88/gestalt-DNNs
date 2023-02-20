import torch.random

from src.utils.Config import Config
# from src.compute_cossim_dots_hierarchical import generate_dataset_rndset
from src.experiment_1.distance_helper import compute_distance_set
from itertools import product
from src.utils.create_stimuli.drawing_utils import *
from src.utils.misc import *
from src.utils.misc import main_text_nets, all_nets, appendix_nets, brain_score_nn
import sty
import numpy as np
import random
def run_distance(network_name,
                 pretraining,
                 background,
                 distance_type):
    img_size = np.array((224, 224), dtype=int)
    print(sty.fg.red + f"******{network_name}*****" + sty.rs.fg)
    config = Config(project_name='Pomerantz',
                    verbose=False,
                    distance_type=distance_type,
                    background=background,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    rep=500,
                    draw_obj=DrawShape(background=background, img_size=img_size, width=10, min_dist_bw_points=20, min_dist_borders=40))
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    plt.close('all')
    if network_name == 'prednet':
        config.pretraining = 'kitti'
    exp_folder = f'./results//{config_to_path_hierarchical(config)}'
    if network_name == 'prednet':
        config.pretraining = './models/prednet-L_0-mul-peepFalse-tbiasFalse-best.pt'

    compute_distance_set(config, out_path=exp_folder + f'{distance_type}.df')  # if not os.path.exists(exp_folder + 'cossim.df')



pretraining = ['ImageNet']
background = ['random']#     ['black', 'white'] #, 'random' 'black', 'white']
distance_type = ['euclidean', 'cossim'] # cossim
all_exps = (product(pretraining,
                    list(brain_score_nn.keys()), #all_nets + self_superv_net,
                    background,
                    distance_type
                    ))
arguments = list((dict(pretraining=i[0],
                       network_name=i[1],
                       background=i[2],
                       distance_type=i[3]) for i in all_exps))
[run_distance(**a) for a in arguments]
plt.close('all')
