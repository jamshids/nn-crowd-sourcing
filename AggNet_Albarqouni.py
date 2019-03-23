from itertools import tee
import numpy as np
import sys
import os

curr_path = os.getcwd()
NN_model_path = os.path.join(os.path.dirname(curr_path), 'nn-active-learning')
sys.path.insert(0, NN_model_path)

import NN_extended
from utils import compute_confusion_matrices, compute_posteriors_Estep



class model(object):

    def __init__(self, model, sess, K):

        self.K = K
        self.c = model.class_num
        self.model = model
        self.sess  = sess
        self.t = 0

        # confusion matrices are data-independent
        self.conf_mats = np.zeros((self.c, self.c, self.K))

    def compute_conf_mats(self, non_eternal_gen):

        conf_mats = np.zeros((self.c, self.c, self.K))
        for Xb, Zb, _ in non_eternal_gen:
            if self.t==0:
                posts_b = self.compute_init_posteriors(Zb)
            else:
                posts_b = self.compute_Estep_posteriors(Xb,Zb)

            for k in range(self.K):
                for ell in range(self.c):
                    conf_mats[:,ell,k] = np.sum(
                        posts_b[ell,:]*Zb[k], axis=1)

        # normalizing each column of conf_mats[:,ell,k], which
        # is equivalent to the division to  sum_i[ p_ell(xi) ]
        for k in range(self.K):
            conf_mats[:,:,k] = conf_mats[:,:,k] / np.sum(conf_mats[:,:,k],axis=0)

        return conf_mats

    def compute_init_posteriors(self, Zb):
        """Initializing posterior probabilities of latent ground
        truths before starting the EM step, based on the crowd-votes
        over the training data
        """

        hists = np.sum(Zb ,axis=0)
        return hists / np.sum(hists, axis=0)

    def compute_Estep_posteriors(self, Xb, Zb):

        pies = self.sess.run(self.aux_model.posteriors, 
                             feed_dict = {self.model.x:Xb})

        # initialize the joint with prior
        joints = pies * 1
        for ell in range(self.c):
            # initializing the likelihood of P(Z|y=ell)
            # by ones, and multiply the inner products
            # one-by-one to get the joints P(Z, y=ell)
            likelihoods_ell = np.ones(pies.shape[1])
            for k in range(self.K):
                pi_ell_k = np.expand_dims(self.conf_mats[:,ell,k],axis=1)
                likelihoods_ell = likelihoods_ell * np.prod(
                    np.repeat(pi_ell_k, pies.shape[1], axis=1) ** Zb[k],
                    axis=0)
            joints[ell,:] = joints[ell,:] * likelihoods_ell

        # matrix/vec = each column will be divided by
        # one element of the vector
        E_posts = joints / np.sum(joints,axis=0)

        return E_posts


    def prepare_for_training(self, prev_weights_path):
        # creating the auxiliary model, if necessary
        if not(hasattr(self, 'aux_model')):
            self.aux_model = NN_extended.replicate_model(self.model, '_aux', True)
            self.aux_model.add_assign_ops()
        self.prev_weights_path = prev_weights_path

    
    def run_M_step(self, 
                   eternal_gen, 
                   non_eternal_gen, 
                   M_iter):
        """Running the M-step, which includes compting the E-step
        posteriors for the minibatches, and then maximizing the likelihood
        with respect to the prior parameters, and the confusion elements
        """

        # maximizing wrt prior parameters 
        # i.e., maximizing wrt parameters of the prior network
        # by back-propagation for "M_iter" iteration
        for _ in range(M_iter):
            Xb, Zb, _ = next(eternal_gen)
            if self.t==0:
                posts_b = self.compute_init_posteriors(Zb)
            else:
                posts_b = self.compute_Estep_posteriors(Xb, Zb)
            feed_dict = {self.model.x: Xb, self.model.y_: posts_b}
            if self.model.dropout_rate is not None:
                feed_dict.udapte({self.model.dropout_rate: 
                                  1 - self.model.keep_prob})
            self.sess.run(self.model.train_step, feed_dict=feed_dict)
            

        # maximizing wrt confusion elements (closed form)
        self.conf_mats = self.compute_conf_mats(non_eternal_gen)

    def iterate_EM(self, 
                   M_iter,
                   EM_iter,
                   eternal_gen,
                   non_eternal_gen,
                   test_non_eternal_gen=None):

        # replicating non-eternal generators
        rep_non_eternal_gen = tee(non_eternal_gen, EM_iter)
        if test_non_eternal_gen is not None:
            rep_test_non_eternal_gen = tee(
                test_non_eternal_gen, EM_iter)

        t0 = self.t
        eval_accs = []
        for t in range(t0, t0+EM_iter):
            # E-step
            # we do not do this step explicitly, but save
            # the weights of the current model in an auxiliary
            # model so that it can be used to compute the E-step
            # posteriors of the selected mini-batch samples in
            # the M-step objective.
            self.aux_model.perform_assign_ops(
                self.prev_weights_path, self.sess)

            # M-step
            self.run_M_step(eternal_gen,
                            rep_non_eternal_gen[t-t0],
                            M_iter)

            # saving weights
            self.model.save_weights(self.prev_weights_path)

            if test_non_eternal_gen is not None:
                eval_accs += [eval_model(self.model, self.sess,
                                         rep_test_non_eternal_gen[t-t0])[0]]
                print(eval_accs)

        return eval_accs


def eval_model(model,sess,dat_gen):

    preds = []
    grounds = []
    for Xb, Yb,_ in dat_gen:
        preds += [sess.run(model.prediction, feed_dict={model.x:Xb})]
        grounds += [np.argmax(Yb, axis=0)]
    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    acc = np.sum(preds==grounds) / len(preds)

    return acc, preds
            
