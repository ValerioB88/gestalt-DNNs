import torch.nn
from src.experiment_2.train_gestalt.train_utils import run
from src.utils.Config import Config
from src.utils.net_utils import GrabNet, prepare_network
from src.experiment_2.train_gestalt.datasets import add_compute_stats, MyImageFolder
from src.experiment_2.train_gestalt.train_utils import ExpMovingAverage, CumulativeAverage #weblog_dataset_info
from torch.utils.data import DataLoader
from src.experiment_2.train_gestalt.callbacks import *
from src.utils.misc import *
from itertools import product
from torchvision.transforms import transforms

def train(type_ds, network_name):
    config = Config(stop_when_train_acc_is=95,
                    patience_stagnation=500,
                    project_name='Train-Gestalt',
                    network_name=network_name,
                    batch_size=8 if not torch.cuda.is_available() else 64,
                    weblogger=False,
                    pretraining='ImageNet',  # get_model_path(config, resume=True)
                    learning_rate=0.0005,
                    clip_max_norm=None,
                    type_ds=type_ds,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)


    if config.type_ds == 'orientation' or config.type_ds == 'proximity':
        num_classes = 3
    elif config.type_ds == 'linearity':
        num_classes = 2

    if config.network_name == 'inception_v3':
        pil_t = [transforms.Resize(299)]
    else:
        pil_t = None

    config.net = MyGrabNet().get_net(config.network_name, imagenet_pt=True if config.pretraining == 'ImageNet' else False, num_classes=num_classes)
    prepare_network(config.net, config)
    config.additional_tags = f'{config.type_ds}_{config.network_name}'

    config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'
    # config.run_id = get_run_id(config)
    config.loss_fn = torch.nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(config.net.parameters(),
                                        lr=config.learning_rate)

    train_dataset = add_compute_stats(MyImageFolder)(root=f'./data/learning_EFs_dataset/{config.type_ds}/train', name_generator='train', add_PIL_transforms=pil_t)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                              timeout=0 if config.use_cuda and not config.is_pycharm else 0)

    test_dataset = add_compute_stats(MyImageFolder)(root=f'./data/learning_EFs_dataset/{config.type_ds}/test', name_generator='test', add_PIL_transforms=pil_t)
    test_loaders = [DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size,
                               num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                               timeout=0 if config.use_cuda and not config.is_pycharm else 0)]

    config.step = standard_step


    def call_run(loader, train, callbacks, **kwargs):
        logs = {'ema_loss': ExpMovingAverage(0.2),
                'ema_acc': ExpMovingAverage(0.2),
                'ca_acc': CumulativeAverage()}

        return run(loader,
                   use_cuda=config.use_cuda,
                   net=config.net,
                   callbacks=callbacks,
                   loss_fn=config.loss_fn,
                   optimizer=config.optimizer,
                   iteration_step=config.step,
                   train=train,
                   logs=logs,
                   collect_images=kwargs['collect_images'] if 'collect_images' in kwargs else False)


    def stop(logs, cb):
        logs['stop'] = True
        print('Early Stopping')

    all_cb = [
        StopFromUserInput(),
        ProgressBar(l=len(train_loader), batch_size=config.batch_size, logs_keys=['ema_loss', 'ema_acc']),

        TriggerActionWhenReachingValue(mode='max', patience=20, value_to_reach=0.90, check_every=10, metric_name='ema_acc', action=stop, action_name='goal90%'),

        TriggerActionWithPatience(mode='min', min_delta=0.01,
                                  patience=config.patience_stagnation,
                                  min_delta_is_percentage=False,
                                  metric_name='ema_loss',
                                  check_every=10,
                                  triggered_action=stop,
                                  action_name='Early Stopping',
                                  weblogger=config.weblogger),


        PlateauLossLrScheduler(config.optimizer, patience=1000, check_batch=True, loss_metric='ema_loss'),

        DuringTrainingTest(testing_loaders=test_loaders, auto_increase=True, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run, plot_samples_corr_incorr=True,  callbacks=[
            PrintConsole(id='ca_acc', endln=" / ", plot_every=np.inf, plot_at_end=True)]),

    ]

    all_cb.append(SaveModel(config.net, config.model_output_filename, epsilon_loss=0.01, loss_metric_name='ema_loss', max_iter=100)) if not config.is_pycharm else None

    net, logs = call_run(train_loader, True, all_cb)
    config.weblogger.stop()

network_names = ['alexnet', 'vgg16bn', 'vonenet-cornets', 'inception_v3' ]
type_ds = ['proximity', 'linearity', 'orientation']
all_exps = (product(network_names, type_ds))
arguments = list((dict(network_name=i[0],
                       type_ds=i[1]) for i in all_exps))
[train(**a) for a in arguments]
