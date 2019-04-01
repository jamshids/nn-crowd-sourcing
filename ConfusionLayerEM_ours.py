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


    def get_optimizer(self, PN_start_layer=None):


        # start by collecting parameters of the prior-net
        # based on the layer from which multiple annotator branches
        # are extracted from
        if PN_start_layer is None:
            # assuming that there is only one probe at the
            # input of the prior-net
            probed_inputs = self.model.probes[0]
            assert len(probed_inputs)==1, 'There is ambiguity in'+\
                ' detecting the first layer of PN.'
            PN_start_layer = list(probed_inputs.keys())[0]
        
        # take all the layers after the starting layer as the 
        # PN ---  for now assume we want to add ALL these layers
        # into the training part
        self.train_vars = []
        layers_w_pars = list(self.model.var_dict.keys())
        for i, layer in enumerate(self.model.layer_dict):
            # we still have not reached the beginning of PN,
            # just continue
            if PN_start_layer not in layers_w_pars[:i+1]:
                continue


            if hasattr(self.model, 'grads_vars'):
                self.train_vars = [GV[1] for GV in self.model.grads_vars]

        # collect all the head losses and sum them up
        # also collect parameters of all the heads
        self.loss = [self.model.loss]
        for k in range(self.K):
            for ell in range(self.c):
                head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]
                # each head loss is equal to the weighted cross-entropy of
                # the second term in ther M-step's objective
                self.loss += [head_k_ell.loss]
                self.train_vars += [GV[1] for GV in head_k_ell.grads_vars]
        self.loss = tf.reduce_sum(self.loss)

        # now construct the train step and optimizer using parameters 
        # set in the main body model
        with tf.variable_scope(self.name):
            self.global_step = tf.Variable(0, trainable=False, 
                                           name='global_step')

            if self.model.optimizer_name=='SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.model.learning_rate)
            elif self.model.optimizer_name=='Adam':
                self.optimizer = tf.train.AdamOptimizer(
                    self.model.learning_rate, 
                    self.model.beta1, 
                    self.model.beta2)
            elif self.model.optimizer_name=='RMSProp':
                self.optimizer = tf.train.RMSPropOptimizer(
                self.model.learning_rate,
                self.model.decay,
                self.model.momentum,
                self.model.epsilon)

            grads_vars = self.optimizer.compute_gradients(
                self.loss, self.train_vars)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.apply_gradients(
                    grads_vars, global_step=self.global_step)

    def initialize_optimizer(self):
        
        var_list = [var for var in tf.global_variables()
                    if self.name in var.name]
        self.sess.run(tf.variables_initializer(var_list))

    def set_aux_model(self, prev_weights_path):

        # creating the auxiliary model, if necessary
        if not(hasattr(self, 'aux_model')):
            self.aux_model = NN_extended.replicate_model(self.model, '_aux', True)
            self.aux_model.add_assign_ops()
        self.prev_weights_path = prev_weights_path


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
        if len(self.model.dropout_layers)>0:
            feed_dict = {self.model.keep_prob: 1.}
        else:
            feed_dict = {}
        for Xb, Zb, _ in train_dat_gen:
            for i in range(self.K):
                Z[i] += [Zb[i]]
            feed_dict[self.model.x] = Xb
            train_preds += [self.sess.run(self.model.prediction, 
                                          feed_dict=feed_dict)]
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
    
            feed_dict={}
            feed_dict[self.model.x] = Xb
            feed_dict[self.model.y_] = posts_b
            if len(self.model.dropout_layers)>0:
                feed_dict[self.model.keep_prob] = 1-self.model.dropout_rate
            for k in range(self.K):
                # observed annotations from the k-th labeler
                for ell in range(self.c):
                    head_k_ell = self.model.branches['labeler_{}{}'.format(k,ell)]
            
                    feed_dict[head_k_ell.y_] = Zb[k]
                    feed_dict[head_k_ell.labeled_loss_weights] = posts_b[ell,:]
                    if len(head_k_ell.dropout_layers)>0:
                        feed_dict[head_k_ell.keep_prob] = 1-head_k_ell.dropout_rate
                
            self.sess.run(self.train_step, feed_dict=feed_dict)

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
            #self.aux_model.perform_assign_ops(
            #    self.prev_weights_path, self.sess)

            # M-step
            self.run_M_step(dat_gen, M_iter)
            self.t += 1

            # saving the weights
            self.model.save_weights(self.prev_weights_path)

            if test_dat_gen is not None:
                eval_accs += [eval_model(self.model,self.sess,
                                         rep_test_gen[t-t0])[0]]
                print({'0:.4f'}.format(eval_accs[-1]), end=', ')

        return eval_accs


def eval_model(model,sess,dat_gen):

    preds = []
    grounds = []
    if model.dropout_rate is None:
        feed_dict = {}
    else:
        feed_dict = {model.keep_prob: 1.}
    
    for Xb, Yb,_ in dat_gen:
        feed_dict.update({model.x:Xb})
        preds += [sess.run(model.prediction, 
                           feed_dict=feed_dict)]
        grounds += [np.argmax(Yb, axis=0)]
    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    acc = np.sum(preds==grounds) / len(preds)

    return acc, preds
            
