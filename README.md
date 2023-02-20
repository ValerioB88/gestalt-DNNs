# Mixed Evidence for Gestalt Grouping in Deep Neural Networks
Valerio Biscione & Jeffrey S. Bowers, 2023

Computational Brain & Behaviour 

*Gestalt psychologists have identified a range of conditions in which humans organize elements of a scene into a group or whole, and perceptual grouping principles play an essential role in scene perception and object identification. Recently, Deep Neural Networks (DNNs) trained on natural images (ImageNet) have been proposed as compelling models of human vision based on reports that they perform well on various brain and behavioral benchmarks.  Here we test a total of 16 networks covering a  variety of architectures and learning paradigms (convolutional, attention-based, supervised and self-supervised, feed-forward and recurrent) on dots (Experiment 1) and more complex shapes (Experiment 2) stimuli that produce strong Gestalts effects in humans. In Experiment 1 we found that convolutional networks were indeed sensitive in a human-like fashion to the principles of proximity, linearity, and orientation, but only at the output layer. In Experiment 2, we found that most networks exhibited Gestalt effects only for a few sets, and again only at the latest stage of processing. Overall, self-supervised and Vision-Transformer appeared to perform worse than convolutional networks in terms of human similarity. Remarkably, no model presented a grouping effect at the early or intermediate stages of processing. This is at odds with the widespread assumption that Gestalts occur prior to object recognition, and indeed, serve to organize the visual scene for the sake of object recognition. Our overall conclusion is that, albeit noteworthy that networks trained on simple 2D images support a form of Gestalt grouping for some stimuli at the output layer, this ability does not seem to transfer to more complex features. Additionally, the fact that this grouping only occurs at the last layer suggests that networks learn fundamentally different perceptual properties than humans.*

<img src="https://github.com/ValerioB88/gestalt-DNNs/blob/main/FigureGitHubWB.png">
The work is currently hosted on [arxiv](https://arxiv.org/abs/2203.07302). You don't need any dataset to run this as the images are generated on the fly, but if you want to see the images, you can check this [link](https://valeriobiscione.com/PomerantzDataset).

## Dependencies 
A part from standard libraries, you need to install CORnet:
`pip install git+https://github.com/dicarlolab/CORnet
`
after that you should only need sty:
```
pip install sty;  
```
Other external networks such as VOneNet and PredNet are included as file in `external` within the library.
 

# General Organization
You can compute the cosine similarity for all networks/transformations/stimulus sets/background conditions by running `src/experiment_{X}/compute_distance.py`, where `{X}` is either 1 or 2.  This script will save a pickle file with the cosine similarity for all Convolutional and Linear layers. In this file we also compute the additional test such as unfamiliar shapes, or the "empty vs single dot" canvas (the control condition in Experiment 1). 

In the `analysis` folder within each experiment folder you will find various scripts to do several types of analysis on these files. These scripts also generate the related figures. Some of these are in the paper, other didn't make the cut but could potentially be interesting.

In a previous version of Experiment 2 we trained directly for orientation, proximity, and linearity. We left the script here you want to try that. The scripts for replicating this are in `src/experiment_1/train_gestalt/`. First, you need to generate the dataset using the script `src/experiment_1/train_gestalt/create_dataset.py`. Then you can run the trainnig with `src/experiment_1/train_gestalt/train_gestalt.py`.

## Figures
- **Figure 3** and **Figure A.8**: `src/experiment_1/analysis/boxplot.py` un/comment the correct set of arguemnts at the bottom of thescript to generate either figures.
- **Figure 4**: `src/experiment_1/analysis/all_layers_lineplot.py`
- **Figure 5** and **Figure A.9**: `src/experiment_2/analysis/boxplot.py` un/comment the correct set of arguemnts at the bottom of thescript to generate either figures.
- **Figure 6**: `src/experiment_2/analysis/all_layers_lineplot.py`
- **Figure 7**: `src/experiment_2/analysis/boxplot_familiarity.py`
- **Figure B.10**: `src/experiment_2/analysis/net_vs_hum_multilayers.py`
