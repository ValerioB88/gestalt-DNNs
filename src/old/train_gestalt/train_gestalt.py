import torch.nn
from src.old.train_gestalt.train_utils import run
from src.utils.Config import Config
from src.utils.net_utils import prepare_network
from src.old.train_gestalt.datasets import add_compute_stats, MyImageFolder
from src.old.train_gestalt.train_utils import ExpMovingAverage, CumulativeAverage #weblog_dataset_info
from torch.utils.data import DataLoader
from src.old.train_gestalt.callbacks import *
from src.utils.misc import *
from itertools import product
from torchvision.transforms import transforms

def train(type_ds, network_name, pt):
    config = Config(stop_when_train_acc_is=95,
                    patience_stagnation=500,
                    project_name='Train-Gestalt',
                    network_name=network_name,
                    batch_size=8 if not torch.cuda.is_available() else (8 if 'vit_l' in network_name else 32),
                    weblogger=False,  #set to "2" if you want to log into neptune client
                    pretraining=pt,
                    learning_rate=0.0005,
                    clip_max_norm=None,
                    type_ds=type_ds,
                    is_pycharm=True if 'PYCHARM_HOSTED' in os.environ else False)


    if config.type_ds == 'orientation' or config.type_ds == 'proximity':
        num_classes = 3
    elif config.type_ds == 'linearity':
        num_classes = 2
    config.verbose = False
    print(sty.fg.red + f"~~~~~~~ {config.network_name} ~~~~~~~" + sty.rs.fg)
    config.net, stats, resize = MyGrabNet().get_net(config.network_name, imagenet_pt=True if config.pretraining == 'ImageNet' else False, num_classes=num_classes)
    prepare_network(config.net, config)
    pil_t = [transforms.Resize(resize)] if resize else None

    parameters = config.net.parameters()
    if 'simCLR' in config.network_name:
        for name, param in config.net.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, config.net.parameters()))
        assert len(parameters) == 2
        config.net.fc = torch.nn.Linear(512, num_classes)

    if 'prednet' in config.network_name:
        # pass
        config.net = torch.nn.Sequential(config.net,
                                         torch.nn.Flatten(),
                                         torch.nn.Linear(61440, num_classes).cuda())
        config.net.__class__.__name__ = 'PredNet'



    if 'dino' in config.network_name:
        for name, param in config.net.named_parameters():
            param.requires_grad = False
        if config.network_name == 'dino_vits8':
            config.net = torch.nn.Sequential(config.net,
                                torch.nn.Linear(384, num_classes).cuda())
        elif config.network_name == 'dino_vitb8':
            config.net = torch.nn.Sequential(config.net,
                                             torch.nn.Linear(768, num_classes).cuda())

    config.additional_tags = f'{config.type_ds}_{config.network_name}'

    config.model_output_filename = './models/' + config_to_path_train(config) + '.pt'
    # config.run_id = get_run_id(config)
    config.loss_fn = torch.nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(parameters,
                                        lr=config.learning_rate)
    config.results_folder = f'./results/learning_EFs_dataset/{config.network_name}/'
    train_dataset = add_compute_stats(MyImageFolder)(root=f'./data/learning_EFs_dataset/{config.type_ds}/train', name_ds='train', add_PIL_transforms=pil_t, stats=stats)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                              num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                              timeout=0 if config.use_cuda and not config.is_pycharm else 0)
    train_loader.name_ds = 'train'

    test_dataset = add_compute_stats(MyImageFolder)(root=f'./data/learning_EFs_dataset/{config.type_ds}/test', name_ds='test', add_PIL_transforms=pil_t, stats=train_dataset.stats)

    test_loaders = [DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size,
                               num_workers=8 if config.use_cuda and not config.is_pycharm else 0,
                               timeout=0 if config.use_cuda and not config.is_pycharm else 0)]

    config.step = standard_step


    def call_run(loader, train, callbacks, logs_prefix='', logs=None, **kwargs):
        if logs is None:
            logs = {}
        logs.update({f'{logs_prefix}ema_loss': ExpMovingAverage(0.2),
                     f'{logs_prefix}ema_acc': ExpMovingAverage(0.2),
                     f'{logs_prefix}ca_acc': CumulativeAverage()})

        return run(loader,
                   use_cuda=config.use_cuda,
                   net=config.net,
                   callbacks=callbacks,
                   loss_fn=config.loss_fn,
                   optimizer=config.optimizer,
                   logs_prefix=logs_prefix,
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

        # Either train for 10 epochs (which is more than enough for convergence):
        TriggerActionWhenReachingValue(mode='max', patience=1, value_to_reach=50, check_after_batch=False, check_every=1, metric_name='epoch', action=stop, action_name='10epochs'),

        # Or explicitely traing until 90% accuracy or convergence: 
        TriggerActionWhenReachingValue(mode='max', patience=20, value_to_reach=0.90, check_every=10, metric_name='ema_acc', action=stop, action_name='goal90%'),
        #
        TriggerActionWithPatience(mode='min', min_delta=0.01,
                                  patience=config.patience_stagnation,
                                  min_delta_is_percentage=False,
                                  metric_name='ema_loss',
                                  check_every=10,
                                  triggered_action=stop,
                                  action_name='Early Stopping',
                                  weblogger=config.weblogger),


        PlateauLossLrScheduler(config.optimizer, patience=1000, check_batch=True, loss_metric='ema_loss'),

        *[DuringTrainingTest(testing_loaders=tl, eval_mode=False, every_x_epochs=1, auto_increase=False, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda,  logs_prefix='test/', call_run=call_run, plot_samples_corr_incorr=True,  callbacks=[
            PrintConsole(id='test/ca_acc', endln=" / ", log_prefix='test/', plot_every=np.inf, plot_at_end=True),
            SaveInfoCsv(log_names=[f'epoch',  'ema_acc', 'test/ca_acc'], path=config.results_folder + f'{config.type_ds}.csv')]) for tl in test_loaders]

    ]

    # all_cb.append(SaveModel(config.net, config.model_output_filename, epsilon_loss=0.01, loss_metric_name='ema_loss', max_iter=100)) if not config.is_pycharm else None

    net, logs = call_run(train_loader, True, all_cb)
    if config.weblogger:
        config.weblogger.stop()

network_names =  ['vgg19bn']#  ['prednet-v-sup'] #'simCLR_resnet18_stl10',  'dino_vitb8', 'dino_vits8', 'densenet201', , 'vit_l_32', 'vit_b_32','vgg19bn', , 'resnet152', 'vonenet-resnet50', 'cornet-s', 'vonenet-cornets', 'alexnet', 'inception_v3',


type_ds = ['proximity', 'linearity', 'orientation']
pt = ['ImageNet'] # 'vanilla']
all_exps = (product(network_names, type_ds, pt))
arguments = list((dict(network_name=i[0],
                       type_ds=i[1],
                       pt=i[2]) for i in all_exps))
[train(**a) for a in arguments]
