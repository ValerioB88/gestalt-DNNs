import numpy as np
import shutil
import os
import pathlib
from src.utils.create_stimuli.drawing_utils import *
img_size = np.array((224, 224), dtype=int)
width = int(img_size[0] * 0.06)

dr = DrawShape(background='random', img_size=img_size, width=10)
N_train = 2000
N_test = N_train / 10


def generate_orientation_in_folder(name_folder, n):
    counter = {i: 0 for i in range(3)}

    while np.any([i < n for i in counter.values()]):
        # pp = np.array(dr.get_orientation_points(pp=(((112, 112),), ((200, 200),))))[0]
        repeat = True
        while repeat:
            try:
                pp = np.array(dr.get_orientation_points()[0])
                repeat = False
            except ConstrainedError:
                pass
        im = dr.draw_all_dots(pp)
        # pp = pp[np.argsort(np.linalg.norm(pp, axis=0))]
        w = (pp[0]-pp[1])/2
        wuni = w/np.linalg.norm(w)
        dd = np.rad2deg(np.arccos(np.dot(wuni, np.array([0, 1]))))
        orientation = np.min([dd, 90 - (dd - 90)])
        if orientation < 25:
            oriclass = 0
        elif orientation > 35 and orientation < 60:
            oriclass = 1
        elif orientation > 70:
            oriclass = 2
        else:
            continue
        # oriclass = int(np.floor(orientation/10))
        print(orientation)
        if counter[oriclass] >= n:
            continue
        counter[oriclass] += 1
        pathlib.Path(name_folder + f'/{oriclass}/').mkdir(parents=True, exist_ok=True)

        im.save(name_folder + f'/{oriclass}/{counter[oriclass]}.png')



def generate_proximity_in_folder(name_folder, n):
    counter = {i: 0 for i in range(3)}

    while np.any([i < n for i in counter.values()]):
        # pp = np.array(dr.get_orientation_points(pp=(((112, 112),), ((200, 200),))))[0]
        repeat = True
        while repeat:
            try:
                pp = np.array(dr.get_proximity_points()[0])
                repeat = False
            except ConstrainedError:
                pass
        im = dr.draw_all_dots(pp)
        # pp = pp[np.argsort(np.linalg.norm(pp, axis=0))]
        prox = np.linalg.norm(pp[0] - pp[1])
        if prox < 50:
            proxclass = 0
        elif prox >= 60 and prox < 110:
            proxclass = 1
        elif prox >= 120:
            proxclass = 2
        else:
            continue
        if counter[proxclass] >= n:
            continue
        counter[proxclass] += 1
        pathlib.Path(name_folder + f'/{proxclass}/').mkdir(parents=True, exist_ok=True)

        im.save(name_folder + f'/{proxclass}/{counter[proxclass]}.png')


def generate_linearity_in_folder(name_folder, n):
    count = 0
    while count < n:
        # pp = np.array(dr.get_orientation_points(pp=(((112, 112),), ((200, 200),))))[0]
        repeat = True
        while repeat:
            try:
                pp = np.array(dr.get_linearity_o_points())
                repeat = False
            except ConstrainedError:
                pass
        im1, im2 = dr.draw_all_dots(pp[0]),  dr.draw_all_dots(pp[1])
        # pp = pp[np.argsort(np.linalg.norm(pp, axis=0))]
        count += 1
        pathlib.Path(name_folder + f'/0/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(name_folder + f'/1/').mkdir(parents=True, exist_ok=True)

        im1.save(name_folder + f'/0/{count}.png')
        im2.save(name_folder + f'/1/{count}.png')


# folder = './data/learning_EFs_dataset/orientation'
# shutil.rmtree(folder) if os.path.exists(folder) else None
# generate_orientation_in_folder(folder + '/train', N_train)
# generate_orientation_in_folder(folder + '/test', N_test)
#
#
# folder = './data/learning_EFs_dataset/proximity'
# shutil.rmtree(folder) if os.path.exists(folder) else None
# generate_proximity_in_folder(folder + '/train', N_train)
# generate_proximity_in_folder(folder + '/test', N_test)
#

folder = './data/learning_EFs_dataset/linearity'
shutil.rmtree(folder) if os.path.exists(folder) else None
generate_linearity_in_folder(folder + '/train', N_train)
generate_linearity_in_folder(folder + '/test', N_test)
