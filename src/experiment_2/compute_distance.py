#%% 
from src.utils.Config import Config
from src.experiment_2.distance_helper import generate_dataset_rnd
from src.utils.create_stimuli.drawing_utils import *
from itertools import product
from src.utils.misc import *
from src.utils.misc import main_text_nets, all_nets, appendix_nets


def run_cossim(network_name, pretraining, type_ds, background, distance_type, transf_code, type_ds_args=None):
    img_size = np.array((224, 224), dtype=int)
    config = Config(project_name='Pomerantz',
                    distance_type=distance_type,
                    verbose=False,
                    network_name=network_name,
                    pretraining=pretraining,  # get_model_path(config, resume=True)
                    weblogger=0,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False,
                    type_ds=type_ds,
                    background=background,
                    draw_obj=DrawShape(background='black' if background == 'black' or background == 'random' else background, img_size=img_size, width=10),
                    rep=500,
                    transf_code=transf_code,
                    type_ds_args=type_ds_args)


    plt.close('all')
    a=plt.figure(1)
    if network_name == 'prednet':
        config.pretraining = 'kitti'
    exp_folder = f'./results//{config_to_path_special(config)}'
    if network_name == 'prednet':
        config.pretraining = './models/prednet-L_0-mul-peepFalse-tbiasFalse-best.pt'

    generate_dataset_rnd(config, out_path=exp_folder)  # if not os.path.exists(exp_folder + '_cossim.df') else None


pretraining = ['ImageNet']
background = ['random'] #, 'white', 'black']
distance_type = ['euclidean', 'cossim'] #,'cossim'
type_ds = [f'array{i}' for i in range(1, 19)]
type_ds.extend(['arrayF', 'arrayA', 'arrayB', 'arrayC', 'arrayD', 'arrayE'])
type_ds.extend(['curly_composite_with_space', 'array4_curly', 'array11_curly'])
transf_code = ['t'] # 'none', 't', 's', 'r']
nets = list(brain_score_nn.keys())  # all_nets + self_superv_net,
all_exps = (product(pretraining, nets, type_ds, background, distance_type, transf_code))
arguments = list((dict(pretraining=i[0], network_name=i[1], type_ds=i[2], background=i[3], distance_type=i[4], transf_code=i[5]) for i in all_exps))
[run_cossim(**a) for a in arguments]
plt.close('all')

# %%
