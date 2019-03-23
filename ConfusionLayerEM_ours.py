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

    def __init__(self, model, sess):

        self.K = int(len(model.branches) / model.class_num)
        self.c = model.class_num
        self.t = 0      # EM iteration index
        self.model = model
        self.sess  = sess

    def compute_init_conf_mats(self, train_dat_gen):
        """Computing initial confusion matrices for all the 
        labelers

        The data is provided through a data generator `train_dat_gen`
        which generates the training data. The eternality flag is set to 
        `False`, hence the generator stops once all the samples are visited.
        """

        # get the prediction of the current model
        Z = [[] for i in range(self.K)]
        train_preds = []
        for Xb, Zb, _ in train_dat_gen:
            for i in range(self.K):
                Z[i] += [Zb[i]]
            train_preds += [self.sess.run(self.model.prediction, 
                                    feed_dict={self.model.x: Xb})]
        for i in range(self.K):
            Z[i] = np.concatenate(Z[i], axis=1)
        train_preds = np.concatenate(train_preds)

        # Z and preds are shuffled here (the order is randomly
        # determined based on random generation of the batches),
        # but it does not matter in the esimation of the confusion 
        # matrix, as it only uses the counts. The only thing that matters
        # here is that the order of columns in Z[i]'s and train_preds
        # do match.
        init_conf_mats = np.zeros((self.c,self.c,self.K))
        for k in range(self.K):
            for j in range(self.c):
                for ell in range(self.c):
                    restricted_Z = Z[k][:, train_preds==ell]
                    init_conf_mats[j,ell,k] = np.sum(restricted_Z[j,:]) / np.sum(train_preds==ell)

        return init_conf_mats

    def compute_conf_mats(self, single_x):
        """Computing confusion matrix for a single data point
        """

        conf_mats = np.zeros((self.c, self.c, self.K))
        for k in range(self.K):
            for ell in range(self.c):
                head_k_ell = self.aux_model.branches['labeler_{}{}'.format(k,ell)]
                pi_jell_k = self.sess.run(head_k_ell.posteriors, 
                                          feed_dict={head_k_ell.x: single_x})
                conf_mats[:,ell,k] = np.squeeze(pi_jell_k)

        return conf_mats

    def compute_Estep_posteriors(self, Xb, Zb, conf_mats=None):
        
        # priors
        pies = self.sess.run(self.aux_model.posteriors, 
                             feed_dict={self.model.x:Xb})
        E_posts = np.zeros((self.c, pies.shape[1]))
        for i in range(pies.shape[1]):

            if Xb.ndim==2:
                single_x = Xb[:,[i]]
            else:
                single_x = Xb[[i],:,:]
            
            # if a confusion matrix is already given,
            # just use that for all samples, otherwise,
            # compute it separately for each sample
            if conf_mats is None:
                conf_mats = self.compute_conf_mats(single_x)

            # for this sample, initialize the joint with prior
            joint = pies[:,i]
            for ell in range(self.c):
                for k in range(self.K):
                    joint[ell] = joint[ell] * np.prod(conf_mats[:,ell,k]**Zb[k][:,i])

            # normalize the joint
            E_posts[:,i] = joint / np.sum(joint)

        return E_posts
                

    def prepare_for_training(self, prev_weights_path):

        # collecting all the train ops and the training feed_dict
        feed_dict = {self.model.x: [], self.model.y_: []}
        train_ops = [self.model.train_step]
        for k in range(self.K):
            for ell in range(self.c):
                head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]
                feed_dict.update({head_k_ell.y_: [], 
                                  head_k_ell.labeled_loss_weights: []})
                train_ops += [head_k_ell.train_step]
        self.feed_dict = feed_dict
        self.train_ops = train_ops

        # creating the auxiliary model, if necessary
        if not(hasattr(self, 'aux_model')):
            self.aux_model = NN_extended.replicate_model(self.model, '_aux', True)
            self.aux_model.add_assign_ops()
        self.prev_weights_path = prev_weights_path

    def run_M_step(self,
                   dat_gen, 
                   M_iter):
        """ Performing one step of the M-step

        The input data generator `dat_gen` is assumed to output three
        values: `(Xb, Yb, inds)`, where
        
            * `Xb`: batch of samples
            * `Zb`: observed labels for the samples (list of `K` label matrices)  

        The eternality flag of this data generator should be set to `True` so
        that it does not stop iterations even when all the samples are visited.
        """

        # compute the initial confusion matrix, if it's the 
        # first step
        if self.t==0:
            conf_mats = self.init_conf_mats
        else:
            conf_mats = None  # should be comptued for each sample

        for _ in range(M_iter):
            Xb, Zb, inds = next(dat_gen)
            posts_b = self.compute_Estep_posteriors(Xb, Zb, conf_mats)
    
            self.feed_dict[self.model.x] = Xb
            self.feed_dict[self.model.y_] = posts_b
            for k in range(self.K):
                # observed annotations from the k-th labeler
                for ell in range(self.c):
                    head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]
            
                    self.feed_dict[head_k_ell.y_] = Zb[k]
                    self.feed_dict[head_k_ell.labeled_loss_weights] = posts_b[ell,:]
                
            self.sess.run(self.train_ops, feed_dict=self.feed_dict)

    def iterate_EM(self, 
                   EM_iter,
                   M_iter,
                   dat_gen,
                   test_dat_gen=None):

        if test_dat_gen is not None:
            rep_test_gen = tee(test_dat_gen, EM_iter)
        eval_accs = []
        t0 = self.t
        for t in range(self.t, self.t+EM_iter):
            # E-step
            # we do not do this step explicitly, but save
            # the weights of the current model in an auxiliary
            # model so that it can be used to compute the E-step
            # posteriors of the selected mini-batch samples in
            # the M-step objective.
            self.aux_model.perform_assign_ops(
                self.prev_weights_path, self.sess)

            # M-step
            self.run_M_step(dat_gen, M_iter)
            self.t += 1

            # saving the weights
            self.model.save_weights(self.prev_weights_path)

            if test_dat_gen is not None:
                eval_accs += [eval_model(self.model,self.sess,
                                         rep_test_gen[t-t0])[0]]
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
            
