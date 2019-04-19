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


    def __init__(self, model, sess, K, name=None):

        self.K = K
        self.c = model.class_num
        self.t = 0      # EM iteration index
        self.model = model
        self.sess  = sess
        self.name=name
        if name is None:
            self.name = model.name + '_OneCoinLayer'

    def build_crowd_layers(self, clip_posts=0.):

        # dimensionality of the input to the crowd layers
        probed_inputs = list(self.model.probes[0].keys())
        assert len(probed_inputs)==1, \
            'There is ambiguity in which input probes is '+\
            'the input to the crowd layers.'
        U = self.model.probes[0][probed_inputs[0]]
        du = U.shape[0].value

        shape_W = [self.K, 2, du]
        shape_b = [self.K, 2, 1]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.crowd_global_step = tf.Variable(
                0, trainable=False, name='crowd_global_step')
            # parameters of the crowd layers
            init_W = np.zeros(shape_W, dtype=np.float32)
            #init_W[:,0,:] = 1.#tf.constant(0., shape=shape_W)
            self.crowd_W = tf.get_variable('crowd_W', initializer=init_W)
            init_b = tf.constant(0., shape=shape_b)
            self.crowd_b = tf.get_variable('crowd_b', initializer=init_b)

            # output of the crowd layers 
            # (K x 2 x b)
            self.crowd_output = tf.keras.backend.dot(self.crowd_W,U) + self.crowd_b
            self.crowd_posteriors = tf.nn.softmax(self.crowd_output,
                                                  axis=1)

            if clip_posts > 0.:
                self.crowd_clipped_posteriors = tf.clip_by_value(
                    self.crowd_posteriors,clip_posts,1-clip_posts)
                self.clip_posts_thr = clip_posts
            
            # get the target placeholder for the crowds
            self.crowd_target = tf.placeholder(tf.float32, 
                                               self.crowd_output.shape)

    def get_crowd_optimizer(self, learning_rate, optimizer_name='SGD'):
        
        #if hasattr(self, 'crowd_clipped_posteriors'):
        #    vec = -tf.reduce_sum(
        #        self.crowd_target * tf.log(self.crowd_clipped_posteriors), axis=1)
        #else:
        #    vec = -tf.reduce_sum(
        #        self.crowd_target * tf.log(self.crowd_posteriors), axis=1)

        vec = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.crowd_output,
            labels=self.crowd_target,
            dim=1)

        # filtering missing data
        mask = tf.equal(self.crowd_target[:,0,:], -1)
        zer = tf.zeros_like(vec)
        self.loss = tf.where(mask,  x=zer, y=vec)

        # optimizer
        # -----------------
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            if optimizer_name=='SGD':
                self.crowd_optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate)
            elif optimizer_name=='Adam':
                self.crowd_optimizer = tf.train.AdamOptimizer(
                    learning_rate)

            self.crowd_grads_vars = self.crowd_optimizer.compute_gradients(
                self.loss, [self.crowd_W, self.crowd_b])

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.crowd_train_step = self.crowd_optimizer.apply_gradients(
                    self.crowd_grads_vars, global_step=self.crowd_global_step)

    def initialize_optimizer(self):
        
        # this var_list does not contain the mode variables
        var_list = [var for var in tf.global_variables()
                    if self.name in var.name]
        self.sess.run(tf.variables_initializer(var_list))

    def compute_init_succ_probs(self, non_eternal_gen, Z):
        """Computing initial confusion matrices for all the 
        labelers, using the prediction of the current PN model

        The output is a PMF for each annotator with 2 elements;
        first element, `succ_probs[0,k]`, is the probability that
        the `k`-th annotator is correct, and `succ_prob[1,k]` is
        the probability the she is wrong.
        """

        succ_probs = np.zeros((2, self.K))

        # get the prediction of the current model
        Yhats = []
        all_inds = []
        for Xb,_,inds in non_eternal_gen:
            feed_dict = {self.model.x:Xb, self.model.keep_prob:1.}
            Yhats += [self.sess.run(self.model.prediction, 
                                    feed_dict=feed_dict)]
            all_inds += [inds]
        
        Yhats = np.concatenate(Yhats)
        all_Yhats = np.zeros(len(Yhats))
        all_inds = np.concatenate(all_inds)
        all_Yhats[all_inds] = Yhats

        for k in range(self.K):
            annot_inds = np.sum(Z[k],axis=0)>0
            annots = np.argmax(Z[k][:,annot_inds], axis=0)
            annot_Yhats = all_Yhats[annot_inds]
            succ_probs[0,k] = np.sum(annot_Yhats == annots)/np.sum(annot_inds)
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

        feed_dict = {self.model.x: Xb, self.model.keep_prob: 1.}
        return self.sess.run(self.crowd_posteriors, feed_dict=feed_dict)

    def compute_clipped_succ_probs(self, Xb):
        """Computing success probability of all annotators at
        a given batch of samples

        size of input `Xb`:  d x b
        size of output    :  K x 2 x b
        """

        feed_dict = {self.model.x: Xb, self.model.keep_prob: 1.}
        return self.sess.run(self.crowd_clipped_posteriors, feed_dict=feed_dict)

    def compute_Estep_posteriors(self, Xb, Zb, succ_probs=None):
        """Computing E-step posteriors of the ground truth

        Output size:   c x b
        """
        
        # priors  (c x b)
        pies = self.sess.run(self.model.posteriors, 
                             feed_dict={self.model.x:Xb,
                                        self.model.keep_prob:1.})
        # success probabilities for all annotators
        # (K x 2 x b)
        if succ_probs is None:
            succ_probs = self.compute_clipped_succ_probs(Xb)
        
        else:
            succ_probs = np.repeat(np.expand_dims(succ_probs.T, axis=2),
                                   pies.shape[1], axis=2)
            

        # joints (initialized by the priors)
        joints = pies*1
        # Labels tensor (K x b)
        Zb_tns = np.stack(Zb, axis=0)
        Lb_tns = np.argmax(Zb_tns, axis=1)
        # no label for annotator k ==> make it -1
        # (or anything different than 0,...,c-1)
        Lb_tns[np.max(Zb_tns, axis=1)==0] = -1

        for ell in range(self.c):
            # z_i^k==ell  (K x b)
            eq_indic = np.float32(Lb_tns == ell)
            non_eq_indic = np.float32(Lb_tns != ell) * np.float32(Lb_tns>-1)
            joints[ell,:] = joints[ell,:] * \
                            np.prod(succ_probs[:,0,:]**eq_indic, axis=0) * \
                            np.prod((succ_probs[:,1,:]/(self.c-1))**non_eq_indic, axis=0)
        
        if np.any(np.sum(joints,axis=0)==0):
            pdb.set_trace()
        return joints / np.sum(joints,axis=0)
        

    def run_M_step(self,
                   eternal_gen, 
                   M_iter,
                   E_posts):
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
        #if self.t==0:
        #    succ_probs = self.init_succ_probs
        #else:
        #    succ_probs = None  # should be comptued for each sample

        for _ in range(M_iter):

            """ fine-tuning the k-th labeler's head """
            """ ----------------------------------- """
            Xb, Zb, inds = next(eternal_gen)
            #posts_b = self.compute_Estep_posteriors(Xb, Zb, succ_probs)
            posts_b = E_posts[:,inds]
            rep_posts_b = np.repeat(np.expand_dims(posts_b,axis=0),
                                    self.K, axis=0)

            # the target
            target = np.zeros((self.K, 2, posts_b.shape[1]))
            Zb_tns = np.stack(Zb, axis=0)
            rep_posts_zeroed = rep_posts_b * Zb_tns
            max_posts = np.max(rep_posts_zeroed, axis=1)
            max_posts[np.max(Zb_tns, axis=1)==0] = -1
            target[:,0,:] = max_posts
            max_posts = 1 - max_posts
            max_posts[np.max(Zb_tns, axis=1)==0] = -1
            target[:,1,:] = max_posts
            
            feed_dict = {self.model.x: Xb,
                         self.model.keep_prob: 1.,
                         self.crowd_target: target}

            self.sess.run(self.crowd_train_step, feed_dict=feed_dict)

            """ fine-tuning the prior-net """
            """ ------------------------- """
            feed_dict = {self.model.x: Xb, 
                         self.model.y_: posts_b,
                         self.model.keep_prob: 1-self.model.dropout_rate}
            self.sess.run(self.model.train_step, feed_dict=feed_dict)


    def iterate_EM(self, 
                   EM_iter,
                   M_iter,
                   eternal_gen,
                   non_eternal_gen,
                   test_non_eternal_gen=None,
                   validation=False,
                   silent=False):
        """This function takes a list of eternal generators (one
        per annotator) and one non-eternal generator for test data.

        Having multiple eternal generators allows the algorithm
        to run for missing data as well, because the generators will
        be built such that they generate only those samples that are
        labeled by each annotator.
        """

        if test_non_eternal_gen is not None:
            rep_test_gen = tee(test_non_eternal_gen, EM_iter)
        rep_non_eternal_gen = tee(non_eternal_gen, EM_iter)

        eval_accs = []
        t0 = self.t
        for t in range(self.t, self.t+EM_iter):
            # E-step
            if self.t==0:
                succ_probs = self.init_succ_probs
            else:
                succ_probs = None  # should be comptued for each sample

            unordered_E_posts = []
            all_inds = []
            for Xb,Zb,inds in rep_non_eternal_gen[t-t0]:
                posts_b = self.compute_Estep_posteriors(Xb,Zb,succ_probs)
                unordered_E_posts += [posts_b]
                all_inds += [inds]
            all_inds = np.concatenate(all_inds)
            unordered_E_posts = np.concatenate(unordered_E_posts,axis=1)
            E_posts = np.zeros(unordered_E_posts.shape)
            E_posts[:,all_inds] = unordered_E_posts

            # M-step
            self.run_M_step(eternal_gen, M_iter, E_posts)
            self.t += 1

            # saving the weights
            self.model.save_weights(self.prev_weights_path)

            if test_non_eternal_gen is not None:
                acc_t = simple_eval_model(self.model,self.sess,
                                          rep_test_gen[t-t0])[0]
                eval_accs += [acc_t]

                if validation:
                    if np.max(eval_accs)==acc_t:
                        valid_path = os.path.join(
                            os.path.dirname(self.prev_weights_path),
                            'valid_weights.h5')
                        self.model.save_weights(valid_path)

                if not(silent):
                    print('{0:.4f}'.format(eval_accs[-1]), end=', ')

        return eval_accs
