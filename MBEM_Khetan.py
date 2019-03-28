from itertools import tee
import tensorflow as tf
import numpy as np
import sys
import os

curr_path = os.getcwd()
NN_model_path = os.path.join(os.path.dirname(curr_path), 'nn-active-learning')
sys.path.insert(0, NN_model_path)
import NN_extended

class model(object):

    def __init__(self, model, sess, K):

        self.K = K
        self.c = model.class_num
        self.t = 0      # EM iteration index
        self.model = model
        self.sess  = sess

    def compute_init_posteriors(self, Zb):
        """Initializing posterior probabilities of ground
        truths before starting the EM step, based on the crowd-votes
        over the training data
        """

        hists = np.sum(Zb ,axis=0)
        return hists / np.sum(hists, axis=0)

    def compute_posteriors(self, 
                           Zb,
                           conf_mats, 
                           priors):
        """Computing the posteriors given the current estimate
        of confusion matrices and priors
        """
        
        # initialize the posteriors with priors
        b = Zb[0].shape[1]
        posts = np.repeat(np.expand_dims(priors,axis=1),b,axis=1)
        # A=matrix, z=vector, then A**z replaces each
        # column with A[:,j]**z[j] 
        # Here, we want ther way, i.e. A[j,:]**z[j]
        # Hence, make it (A.T**z).T
        for k in range(self.K):
            posts = posts * np.prod(
                (conf_mats[:,:,k].T**Zb[k]).T, axis=0)

        return posts / np.sum(posts)

    def compute_conf_mats_priors(self, non_eternal_dat_gen):
        """Computing confusion matrices and priors assuming that the 
        ground truths are equal to the model predictions
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
        conf_mats = np.zeros((self.c,self.c,self.K))
        for k in range(self.K):
            for j in range(self.c):
                for ell in range(self.c):
                    restricted_Z = Z[k][:, train_preds==ell]
                    conf_mats[j,ell,k] = np.sum(restricted_Z[j,:]) / np.sum(train_preds==ell)

        priors = np.sum(train_preds, axis=1) / np.sum(train_preds)

        return conf_mats, priors



    def run_M_step(self, 
                   eternal_dat_gen, 
                   non_eternal_dat_gen,
                   M_iter):
        
        if self.t>0:
            conf_mats, priors = self.compute_conf_mats_priors(
                non_eternal_dat_gen)

        for _ in range(M_iter):
            Xb,Zb,_ = next(eternal_dat_gen)

            # compute the posteriors
            if self.t==0:
                posts_b = self.compute_init_posteriors(Zb)
            else:
                conf_mats = self.compute_conf_mats_priors
                posts_b = self.compute_posteriors(Zb, conf_mats, priors)

            feed_dict={self.model.x:Xb, self.model.y_:posts_b}
            if self.model.dropout_rate is not None:
                feed_dict[self.model.dropout_rate: 1-self.model.keep_prob]
            
            self.sess.run(self.model.train_step, feed_dict=feed_dict)


    def iterate_MBEM(self, 
                     M_iter,
                     EM_iter,
                     eternal_dat_gen, 
                     non_eternal_dat_gen,
                     test_non_eternal_gen=None):

        # replicating non-eternal generators
        rep_non_eternal_gen = tee(non_eternal_dat_gen, EM_iter)
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

            # M-step
            self.run_M_step(eternal_dat_gen,
                            rep_non_eternal_gen[t-t0],
                            M_iter)

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
