from src.utils.Config import Config
from src.experiment_1.compute_cossim_special import generate_dataset_rnd
from src.utils.create_stimuli.drawing_utils import *
from itertools import product
from src.utils.misc import *

def run_cossim(network_name, pretraining, type_ds, background, transf_code, type_ds_args=None):
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
                    rep=50,
                    transf_code=transf_code,
                    type_ds_args=type_ds_args)

    plt.close('all')
    exp_folder = f'./results//{config_to_path_special(config)}'

    generate_dataset_rnd(config, out_path=exp_folder) #if not os.path.exists(exp_folder + 'cossim.df') else None


pretraining = ['ImageNet']
network_names = ['inception_v3', 'alexnet', 'densenet201', 'vgg19bn', 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets']  #

background = ['random']#, 'black', 'white']

type_ds = [f'array{i}' for i in range(1, 19)]
type_ds.extend(['arrayA', 'arrayB', 'arrayC', 'arrayD', 'arrayE', 'arrayF'])
type_ds.extend(['curly_composite_with_space', 'array4_curly', 'array11_curly'])
transf_code = ['none', 't', 's', 'r']
all_exps = (product(pretraining, network_names, type_ds, background, transf_code))
arguments = list((dict(pretraining=i[0], network_name=i[1], type_ds=i[2], background=i[3], transf_code=i[4]) for i in all_exps))
[run_cossim(**a) for a in arguments]
plt.close('all')
