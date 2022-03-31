from src.utils.Config import Config
# from src.compute_cossim_dots_hierarchical import generate_dataset_rndset
from src.experiment_2.compute_cossim_dots_hierarchical import generate_dataset_rndset
from itertools import product
from src.utils.create_stimuli.drawing_utils import *
from src.utils.misc import *

def run_cossim(network_name,
               pretraining,
               background):
    img_size = np.array((224, 224), dtype=int)

    config = Config(project_name='Pomerantz',
                    verbose=False,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    rep=50,
                    draw_obj=DrawShape(background=background, img_size=img_size, width=10, min_dist_bw_points=20, min_dist_borders=40))

    exp_folder = f'./results/{config_to_path_hierarchical(config)}'
    generate_dataset_rndset(config, out_path=exp_folder + 'cossim.df') # if not os.path.exists(exp_folder + 'cossim.df') else None


pretraining = ['ImageNet']
network_names = ['inception_v3', 'alexnet', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s']  # 'vonenet-resnet50-non-stoch'
background = ['black', 'random', 'white']
all_exps = (product(pretraining,
                    network_names,
                    background
                    ))
arguments = list((dict(pretraining=i[0],
                       network_name=i[1],
                       background=i[2]) for i in all_exps))
[run_cossim(**a) for a in arguments]
plt.close('all')
