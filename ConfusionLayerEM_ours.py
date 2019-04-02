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

    def __init__(self, model, sess, name=None):

        self.K = int(len(model.branches) / model.class_num)
        self.c = model.class_num
        self.t = 0      # EM iteration index
        self.model = model
        self.sess  = sess
        self.name=name
        if name is None:
            self.name = model.name + '_ConfLayer'

    def initialize_optimizer(self):
        
        # this var_list does not contain the mode variables
        var_list = [var for var in tf.global_variables()
                    if self.name in var.name]
        self.sess.run(tf.variables_initializer(var_list))

    def set_aux_model(self, prev_weights_path):

        # creating the auxiliary model, if necessary
        if not(hasattr(self, 'aux_model')):
            self.aux_model = NN_extended.replicate_model(self.model, '_aux', True)
            self.aux_model.add_assign_ops()
        self.prev_weights_path = prev_weights_path


    def compute_init_conf_mats(self, non_eternal_dat_gens):
        """Computing initial confusion matrices for all the 
        labelers, using the prediction of the current PN model
        """

        # get the prediction of the current model
        Z = [[] for i in range(self.K)]
        train_preds = [[] for i in range(self.K)]

        for k, dat_gen in enumerate(non_eternal_dat_gens):
            for Xb, Zb, _ in dat_gen:
                # here, for samples coming from A_k, only labels
                # of the k-th labeler will be necessary
                Z[k] += [Zb[k]]

                feed_dict = {self.model.x:Xb, self.model.keep_prob:1.}
                train_preds[k] += [self.sess.run(self.model.prediction, 
                                                 feed_dict=feed_dict)]
        for k in range(self.K):
            Z[k] = np.concatenate(Z[k], axis=1)
            train_preds[k] = np.concatenate(train_preds[k])

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
                    restricted_Z = Z[k][:, train_preds[k]==ell]
                    init_conf_mats[j,ell,k] = np.sum(restricted_Z[j,:]) / np.sum(train_preds[k]==ell)

        return init_conf_mats

    def compute_conf_mats(self, single_x):
        """Computing confusion matrix for a single data point
        """

        conf_mats = np.zeros((self.c, self.c, self.K))
        feed_dict = {}
        if len(self.model.dropout_layers)>0:
            feed_dict[self.model.keep_prob] = 1.
        for k in range(self.K):
            for ell in range(self.c):
                head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]
                if len(head_k_ell.dropout_layers)>0:
                    feed_dict[head_k_ell.keep_prob] = 1.
                feed_dict[self.model.x] = single_x
                pi_jell_k = self.sess.run(head_k_ell.posteriors, 
                                          feed_dict=feed_dict)
                conf_mats[:,ell,k] = np.squeeze(pi_jell_k)

        return conf_mats

    def compute_Estep_posteriors(self, Xb, Zb, conf_mats=None):
        
        # priors
        pies = self.sess.run(self.model.posteriors, 
                             feed_dict={self.model.x:Xb,
                                        self.model.keep_prob:1.})
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
        
    def run_M_step(self,
                   eternal_gens, 
                   M_iter):
        """ Performing the M-step for few iterations

        The input `eternal_gens` is a list of generators with length K+1.
        The first K generators only generate samples from A_k (to be used
        in fine-tuning heads associated with the k-th labeler), whereas the
        last generator generate samples generally from [n]. Each generator
        outputs the following in each call:
        
            * `Xb`: batch of samples
            * `Zb`: observed labels for the samples (list of `K` label matrices)  
            *  the third element is not important (will be deprecated soon)

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

            """ fine-tuning heads of the k-th labeler """
            """ ------------------------------------- """
            for k, dat_gen in enumerate(eternal_gens[:-1]):
                Xb, Zb, _ = next(dat_gen)
                posts_b = self.compute_Estep_posteriors(Xb, Zb, conf_mats)

                # train-step ops of heads (k,0),...,(k,c)
                train_steps = []
                feed_dict = {self.model.x: Xb,
                             self.model.keep_prob: 1.}
                for ell in range(self.c):
                    head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]

                    # filling the corresponding feed_dict
                    feed_dict[head_k_ell.y_] = Zb[k]
                    feed_dict[head_k_ell.labeled_loss_weights] = posts_b[ell,:]
                    feed_dict[head_k_ell.keep_prob] = 1-head_k_ell.dropout_rate
                    
                    # adding the train step
                    train_steps += [head_k_ell.train_step]
                    
                self.sess.run(train_steps, feed_dict=feed_dict)

            """ fine-tuning the prior-net """
            """ ------------------------- """
            Xb,Zb,_ = next(eternal_gens[-1])
            posts_b = self.compute_Estep_posteriors(Xb, Zb, conf_mats)
    
            feed_dict = {self.model.x: Xb, 
                         self.model.y_: posts_b,
                         self.model.keep_prob: 1-self.model.dropout_rate}
            self.sess.run(self.model.train_step, feed_dict=feed_dict)

    def iterate_EM(self, 
                   EM_iter,
                   M_iter,
                   eternal_gens,
                   test_non_eternal_gen=None):
        """This function takes a list of eternal generators (one
        per annotator) and one non-eternal generator for test data.

        Having multiple eternal generators allows the algorithm
        to run for missing data as well, because the generators will
        be built such that they generate only those samples that are
        labeled by each annotator.
        """

        if test_non_eternal_gen is not None:
            rep_test_gen = tee(test_non_eternal_gen, EM_iter)
        eval_accs = []
        t0 = self.t
        for t in range(self.t, self.t+EM_iter):
            # E-step
            # we do not do this step explicitly, but save
            # the weights of the current model in an auxiliary
            # model so that it can be used to compute the E-step
            # posteriors of the selected mini-batch samples in
            # the M-step objective.
            #self.aux_model.perform_assign_ops(
            #    self.prev_weights_path, self.sess)

            # M-step
            self.run_M_step(eternal_gens, M_iter)
            self.t += 1

            # saving the weights
            self.model.save_weights(self.prev_weights_path)

            if test_dat_gen is not None:
                eval_accs += [eval_model(self.model,self.sess,
                                         rep_test_gen[t-t0])[0]]
                print({'0:.4f'}.format(eval_accs[-1]), end=', ')

        return eval_accs
