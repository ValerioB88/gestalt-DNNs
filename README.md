# gestalt-DNNs
Code for reproducing the experiments in _"Do DNNs trained on Natural Images acquire Gestalt Properties?"_ by Valerio Biscione and Jeffrey S. Bowers.

The work is currently hosted on [arxiv](https://arxiv.org/abs/2203.07302). You don't need any dataset to run this as the images are generated on the fly, but if you want to see the images, you can check this [link](https://valeriobiscione.com/PomerantzDataset).

# General Organization
You can compute the cosine similarity for all networks/transformations/stimulus sets/background conditions by running `src/experiment{X}/run_cossim_special_stimuli.py`, where `{X}` is either 1 or 2.  This script will save a pickle file with the cosine similarity for all Convolutional and Linear layers. In this file we also compute the additional test such as unfamiliar shapes, or the "empty vs single dot" canvas (the sanity-check in Experimetn 2). 

In the `../analysis` folder within each experiment folder you will find various scripts to do several types of analysis on these files. These scripts also generate the related figures. Some of these are in the paper, other didn't make the cut but could potentially be interesting.

In a subsection of Experiment 2 we trained directly for orientation, proximity, and linearity. The scripts for replicating this are in `src/experiment2/train_gestalt/`. First, you need to generate the dataset using the script `src/experiment2/train_gestalt/create_dataset.py`. Then you can run the trainnig with `src/experiment2/train_gestalt/train_gestalt.py`. 
