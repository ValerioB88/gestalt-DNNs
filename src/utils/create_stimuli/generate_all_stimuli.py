"""
Generate stimuli used in the paper. Here we only generate stimuli with a randomly pixellated background (can be changed) and without transformation (generally applied at runtime).

Notice that when computing cosine similarity we generated these images on the fly (see for example `run_cossim_dots_hierarchical`. However, having the dataset here in this format could be useful anyway.
"""

from src.utils.create_stimuli.drawing_utils import *
import pathlib

### Generate Stimuli for Experiment 1
background = 'random'
dr = DrawShape(background=background, img_size=(224, 224), width=10)

sample = []
# sample = [f'array{i}' for i in range(1, 19)]
# sample.extend(['arrayA', 'arrayB', 'arrayC', 'arrayD', 'arrayE', 'arrayF'])
sample.extend(['curly_composite_with_space', 'array4_curly', 'array11_curly'])

folder = f'./data/all_stimuli_exp1/'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

# dr.get_curly_composite_with_space()[1].show()
dr.get_array4_curly()[1].show()
dr.get_array11_curly()[0].show()
dr.get_array11()[0].show()

for s in sample:
    im_pair = dr.__getattribute__(f'get_{s}')()
    # im_pair[0].save(f'{folder}/{s}_0.png')
    # im_pair[0].save(f'{folder}/{s}_1.png')

### Generate Stimuli for Experiment 2. Notice that due to the hierarchical nature of this dataset, each set is produced together.
background = 'random'
dr = DrawShape(background=background, img_size=(224, 224), width=10, min_dist_bw_points=20, min_dist_borders=40)
folder = f'./data/all_stimuli_exp2/'
num_sets = 200


for idx_s in range(num_sets):
    set_img, points = dr.get_all_sets()
    for k in set_img.keys():
        pathlib.Path(folder + f'{k}').mkdir(parents=True, exist_ok=True)

        set_img[k][0].save(folder + f'{k}/set{idx_s}_0.png')
        set_img[k][1].save(folder + f'{k}/set{idx_s}_1.png')

