from itertools import tee
import tensorflow as tf
import numpy as np
import pdb
import sys
import os

curr_path = os.getcwd()
NN_model_path = os.path.join(os.path.dirname(curr_path), 'nn-active-learning')
sys.path.insert(0, NN_model_path)

import NN_extended
from eval_utils import simple_eval_model


class model(object):


    def __init__(self, model, sess, name=None):

        self.K = len(model.branches)
        self.c = model.class_num
        self.t = 0      # EM iteration index
        self.model = model
        self.sess  = sess
        self.name=name
        if name is None:
            self.name = model.name + '_OneCoinLayer'

    def get_optimizers_for_heads(self, eps=1e-3, optimizer_name='SGD'):

        self.eps = eps
        self.heads_clipped_posts = []
        for k in range(self.K):

            # loss
            # -----------------
            head_k = self.model.branches['labeler_{}'.format(k)]
            self.heads_clipped_posts += [tf.clip_by_value(head_k.posteriors,eps,1-eps)]
            head_k.loss = -tf.reduce_mean(tf.reduce_sum(
                head_k.y_ * tf.log(self.heads_clipped_posts[k]), axis=0))

            # optimizer
            # -----------------
            if optimizer_name=='SGD':
                head_k.optimizer = tf.train.GradientDescentOptimizer(
                    head_k.learning_rate, name='av_logits_SGD')
            elif optimizer_name=='Adam':
                head_k.optimizer = tf.train_AdamOptimizer(
                    head_k.learning_rate, name='av_logits_Adam')

            var_list = []
            # ASSUMPTION: all the variables in the heads' layers are assumed
            # to be in list of trainable variables (hence no BN in the heads)
            for _,V in head_k.var_dict.items():
                if 'labeler_{}'.format(k) in V[0].name:
                    var_list += V
            head_k.grads_vars = head_k.optimizer.compute_gradients(head_k.loss, var_list)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                head_k.train_step = head_k.optimizer.apply_gradients(
                    head_k.grads_vars, global_step=head_k.global_step)


    def initialize_optimizer(self):
        
        # this var_list does not contain the mode variables
        var_list = [var for var in tf.global_variables()
                    if self.name in var.name]
        self.sess.run(tf.variables_initializer(var_list))

    def compute_init_succ_probs(self, non_eternal_dat_gens):
        """Computing initial confusion matrices for all the 
        labelers, using the prediction of the current PN model

        The output is a PMF for each annotator with 2 elements;
        first element, `succ_probs[0,k]`, is the probability that
        the `k`-th annotator is correct, and `succ_prob[1,k]` is
        the probability the she is wrong.
        """

        succ_probs = np.zeros((2, self.K))

        for k, dat_gen in enumerate(non_eternal_dat_gens):

            # get the prediction of the current model
            Z_k = []
            Yhats_k = []
            for Xb, Zb, _ in dat_gen:
                # here, for samples coming from A_k, only labels
                # of the k-th labeler will be necessary
                Z_k += [Zb[k]]

                feed_dict = {self.model.x:Xb, self.model.keep_prob:1.}
                Yhats_k += [self.sess.run(self.model.prediction, 
                                          feed_dict=feed_dict)]
            Z_k = np.concatenate(Z_k, axis=1)
            Yhats_k = np.concatenate(Yhats_k)

            succ_probs[0,k] = np.sum(Yhats_k == np.argmax(Z_k, axis=0))/len(Yhats_k)
            succ_probs[1,k] = 1 - succ_probs[0,k]

        self.init_succ_probs = succ_probs


    def compute_init_dat_dep_succ_probs(self, Xb, Zb):
        """Data-dependent initialization of the success probabilities

        Output size:   K x 2 x b
        """

        self.init_succ_probs = np.zeros((self.K, 2, Zb[0].shape[1]))
        if not(hasattr(self, 'eps')):
            self.eps = 1e-3
            print('The epsilon value is set to 1e-3.')
        
        # the predictions
        feed_dict = {self.model.x:Xb, self.model.keep_prob:1.}
        Yhats = self.sess.run(self.model.prediction, feed_dict=feed_dict)
        Yhats_idx = np.argmax(Yhats, axis=0)
        
        for k in range(self.K):
            true_indics = Yhats_idx == np.argmax(Zb[k], axis=0)
            self.init_succ_probs[k,0,true_indics] = 1 - self.eps
            self.init_succ_probs[k,0,~(true_indics)] = self.eps
            self.init_succ_probs[k,1,:] = 1 - self.init_succ_probs[k,0,:]


    def compute_succ_probs(self, Xb):
        """Computing success probability of all annotators at
        a given batch of samples

        size of input `Xb`:  d x b
        size of output    :  K x 2 x b
        """
        
        succ_probs = []

        feed_dict = {self.model.branches['labeler_{}'.format(k)].keep_prob: 1.
                     for k in range(self.K)}
        feed_dict[self.model.x] = Xb 
        feed_dict[self.model.keep_prob] = 1.
        for k in range(self.K):
            head_k = self.model.branches['labeler_{}'.format(k)]
            succ_prob_k = self.sess.run(head_k.posteriors, feed_dict=feed_dict)
            succ_probs += [np.expand_dims(succ_prob_k, axis=0)]

        return np.concatenate(succ_probs, axis=0)


    def compute_Estep_posteriors(self, Xb, Zb, succ_probs=None):
        
        # priors  (c x b)
        pies = self.sess.run(self.model.posteriors, 
                             feed_dict={self.model.x:Xb,
                                        self.model.keep_prob:1.})
        # success probabilities for all annotators
        # (K x 2 x b)
        if succ_probs is None:
            succ_probs = self.compute_succ_probs(Xb)
        else:
            succ_probs = np.repeat(np.expand_dims(succ_probs.T, axis=2),
                                   pies.shape[1], axis=2)

        # joints (initialized by the priors)
        joints = pies*1
        # Zb tensor (K x b)
        Zb_tns = []
        for k in range(self.K):
            alabels = np.argmax(Zb[k], axis=0)
            # no label for annotator k ==> make it -1
            # (or anything different than 0,...,c-1)
            alabels[np.max(Zb[k], axis=0)==0] = -1
            Zb_tns += [np.expand_dims(alabels, axis=0)]
        Zb_tns = np.concatenate(Zb_tns, axis=0)

        for ell in range(self.c):
            # z_i^k==ell  (K x b)
            eq_indic = np.float32(Zb_tns == ell)
            non_eq_indic = np.float32(Zb_tns != ell) * np.float32(Zb_tns>-1)
            joints[ell,:] = joints[ell,:] * \
                            np.prod(succ_probs[:,0,:]**eq_indic, axis=0) * \
                            np.prod(succ_probs[:,1,:]**non_eq_indic, axis=0)

        return joints / np.sum(joints,axis=0)
        

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
        if self.t==1:
            succ_probs = self.init_succ_probs
        else:
            succ_probs = None  # should be comptued for each sample

        for _ in range(M_iter):

            """ fine-tuning the k-th labeler's head """
            """ ----------------------------------- """
            for k, dat_gen in enumerate(eternal_gens[:-1]):
                Xb, Zb, _ = next(dat_gen)
                posts_b = self.compute_Estep_posteriors(Xb, Zb, succ_probs)

                head_k = self.model.branches['labeler_{}'.format(k)]

                # the target for the i-th sample:  -p^t(xi) * [1,-1]
                alabels = np.argmax(Zb[k], axis=0)
                target = posts_b[alabels,np.arange(posts_b.shape[1])] 
                target = np.outer(np.array([[1],[-1]]), target)

                feed_dict = {self.model.x: Xb,
                             self.model.keep_prob: 1.,
                             head_k.y_: target,
                             head_k.keep_prob: 1-head_k.dropout_rate}

                self.sess.run(head_k.train_step, feed_dict=feed_dict)

            """ fine-tuning the prior-net """
            """ ------------------------- """
            Xb,Zb,_ = next(eternal_gens[-1])
            posts_b = self.compute_Estep_posteriors(Xb, Zb, succ_probs)
    
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

            if test_non_eternal_gen is not None:
                eval_accs += [simple_eval_model(self.model,self.sess,
                                                rep_test_gen[t-t0])[0]]
                print('{0:.4f}'.format(eval_accs[-1]), end=', ')

        return eval_accs
