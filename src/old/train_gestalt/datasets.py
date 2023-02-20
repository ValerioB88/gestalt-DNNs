import random
import os
import pathlib
import pickle
from pathlib import Path
from time import time
import numpy as np
import torch
import torchvision
from PIL import ImageStat
from sty import fg, rs, ef
from torchvision import transforms as tf
from torchvision.datasets import ImageFolder

class MyImageFolder(ImageFolder):
    def finalize_getitem(self, path, sample, labels, info=None):
        if info is None:
            info = {}
        return sample, labels, info

    def __init__(self, name_classes=None, verbose=True, *args, **kwargs):
        print(fg.red + ef.inverse + "ROOT:  " + kwargs['root'] + rs.inverse + rs.fg)
        self.name_classes = np.sort(name_classes) if name_classes is not None else None
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _find_classes(self, dir: str):
        if self.name_classes is None:
            return super()._find_classes(dir)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and (d.name in self.name_classes)]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        output = self.finalize_getitem(path=path, sample=sample, labels=target)
        return output


def add_return_path(class_obj):
    class ImageFolderWithPath(class_obj):
        def __getitem__(self, index):
            img, target = super().__getitem__(index)
            path = self.samples[index][0]
            return img, target, {'path': path}
    return ImageFolderWithPath


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        ## This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(self, name_ds='dataset', add_PIL_transforms=None, add_tensor_transforms=None, num_image_calculate_mean_std=70, stats=None, save_stats_file=None, **kwargs):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u], a path to a pickle file, or None
            @param save_stats_file:
            @param kwargs:
            """
            self.verbose = True
            print(fg.yellow + f"\n**Creating Dataset [" + fg.cyan + f"{name_ds}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_ds = name_ds
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(fg.red + f"Using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, 'rb'))
                    print(fg.red + f"Using stats from file [{Path(stats).name}]: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(fg.red + f"File [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(parents=True, exist_ok=True)
                pickle.dump(self.stats, open(save_stats_file, 'wb'))

            normalize = torchvision.transforms.Normalize(mean=self.stats['mean'],
                                                         std=self.stats['std'])
            # self.stats = {}
            # self.stats['mean'] = [0.491, 0.482, 0.44]
            # self.stats['std'] = [0.247, 0.243, 0.262]
            # normalize = torchvision.transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            self.transform.transforms += [normalize]

            print(f'Map class_name -> labels: {self.class_to_idx}\n{len(self)} samples.') if self.verbose else None
            self.finalize_init()

        def finalize_init(self):
            pass

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

        def __getitem__(self, idx, class_name=None):
            image, *rest = super().__getitem__(idx)
            label = rest[0]
            return image, label, rest[1] if len(rest) > 1 else {}

    return ComputeStatsUpdateTransform



class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, np.array(self.h)/255, np.array(other.h)/255)))

def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    if max_iteration < 30:
        print(f'Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!') if verbose else None
    start = time()
    idxs = random.choices(range(len(dataset)), k=np.min((max_iteration, len(dataset))))
    imgs = [dataset[i][0] for i in idxs]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()
    means = [imgs[:, i, :, :].mean() for i in range(3)]
    stds = [imgs[:, i, :, :].std(ddof=0) for i in range(3)]
    stats = {'mean': means, 'std': stds, 'time_one_iter': (time() - start)/max_iteration, 'iter': max_iteration}

    print(fg.cyan + f'mean={np.around(stats["mean"],4)}, std={np.around(stats["std"], 4)}, time1it: {np.around(stats["time_one_iter"], 4)}s' + rs.fg) if verbose else None
    if dataset_path is not None:
        print('Saving in {}'.format(dataset_path))
        with open(dataset_path, 'wb') as f:
            pickle.dump(stats, f)

    return stats