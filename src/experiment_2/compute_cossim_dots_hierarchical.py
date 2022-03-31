from sty import fg, bg, rs, ef
import pickle
import torch
from src.utils.distance_activation import RecordActivations
import os
import pathlib
from tqdm import tqdm
import torchvision.transforms as transforms
from src.utils.misc import make_cuda
from src.utils.net_utils import prepare_network, load_pretraining, GrabNet
from src.utils.misc import MyGrabNet, conver_tensor_to_plot, save_fig_pair
from copy import deepcopy
from src.utils.create_stimuli.drawing_utils import *
import torchvision


class GenerateDataset(RecordActivations):
    def get_images_for_each_category(self, dataset, N, **kwargs):
        selected_class = dataset.samples
        correct_paths = selected_class
        correct_paths = [correct_paths[i] for i in np.random.choice(range(len(correct_paths)), np.min([N, len(correct_paths)]), replace=False)]
        return correct_paths

    def compute_cosine_set(self, set, transform, path_save_fig, stats):
        cossim = {}

        images = [[transform(i[0]), transform(i[1])] for i in set]
        image_plt = [[conver_tensor_to_plot(i, stats['mean'], stats['std']) for i in j] for j in images]
        save_fig_pair(path_save_fig, image_plt, n=4)

        for (image0, image1) in tqdm(images):
            self.net(make_cuda(image0.unsqueeze(0), torch.cuda.is_available()))
            first_image_act = {}
            activation_image1 = deepcopy(self.activation)
            for name, features1 in self.activation.items():
                if not np.any([i in name for i in self.only_save]):
                    continue
                first_image_act[name] = features1.flatten()

            self.net(make_cuda(image1.unsqueeze(0), torch.cuda.is_available()))
            activation_image2 = deepcopy(self.activation)

            second_image_act = {}
            for name, features2 in self.activation.items():
                if not np.any([i in name for i in self.only_save]):
                    continue
                second_image_act[name] = features2.flatten()
                if name not in cossim:
                    cossim[name] = []
                cossim[name].append(torch.nn.CosineSimilarity(dim=0)(first_image_act[name], second_image_act[name]).item())

        return cossim


    def compute_random_set(self, transform, N=5, path_save_fig=None, stats=None, draw_obj=None):
        cossim_all = {}
        im_types = {}
        for i in range(N):
            [im_types.setdefault(k, []).append(v) for k, v in draw_obj.get_all_sets()[0].items()]

        for idx, (type, im_samples) in enumerate(im_types.items()):
            cossim_all[type] = self.compute_cosine_set(im_samples, transform, path_save_fig + f'/{type}.png', stats)

        return cossim_all


def generate_dataset_rndset(config, out_path):
    config.model = MyGrabNet().get_net(config.network_name,
                                       imagenet_pt=True if config.pretraining == 'ImageNet' else False)
    prepare_network(config.model, config, train=False)
    if 'vonenet' in config.network_name:
        s = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif config.network_name == 'clip':
        s = dict(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711])
    else:
        s = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    transf_list = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(s['mean'], s['std'])]
    transf_list.insert(0, transforms.Resize(299)) if config.network_name == 'inception_v3' else None
    transform = torchvision.transforms.Compose(transf_list)

    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    recorder = GenerateDataset(net=config.model, use_cuda=False, only_save=['Conv2d', 'Linear'])
    cossim = recorder.compute_random_set(transform=transform, N=config.rep, path_save_fig=os.path.dirname(out_path), stats=s, draw_obj=config.draw_obj)


    print(fg.red + f'Saved in {out_path}' + rs.fg)
    pickle.dump(cossim, open(out_path, 'wb'))

