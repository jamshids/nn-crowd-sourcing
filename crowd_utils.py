from itertools import tee
import tensorflow as tf
import numpy as np
import pdb
import sys
import os

curr_path = os.getcwd()
NN_model_path = os.path.join(os.path.dirname(curr_path), 'nn-active-learning')
sys.path.insert(0, NN_model_path)
from datasets import utils

def gens_complete_dat_missing_annots(X,Z, batch_size):
    """Building eternal and non-eternal data generators
    from missing annotation matrices

    The function returns one generator per annotator, 
    which generates only those samples that the annotator
    has sampled. However, for each batch, it returns labels
    of all annotators (not only the corresponding k-th
    labeler), because in some EM-based approaches we
    need to know labels from all other annotators too.
    More accurately, for each x_i, we need to know labels
    of all annotators in L_i too, but it is easier to 
    return all other annotations.
    """

    K = len(Z)
    batch_size = 100

    eternal_gens = []
    non_eternal_gens = []
    for k in range(K):
        # indices of samples in A_k
        annot_idx = np.sum(Z[k], axis=0)>0
        annot_X = X[:, annot_idx]
        # get all available annotatins for A_k
        annot_Z = []
        for kk in range(K):
            annot_Z += [Z[kk][:,annot_idx]]

        # eternal generators
        eternal_gens += [utils.generator_complete_data(annot_X,
                                                       annot_Z,
                                                       batch_size,
                                                       True)]
        
        # non-eternal generators 
        # (for initiating the confusion matrix) 
        non_eternal_gens += [utils.generator_complete_data(annot_X,
                                                           annot_Z,
                                                           batch_size)]

    return eternal_gens, non_eternal_gens

