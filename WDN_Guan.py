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

    def __init__(self, BN_model, 
                 sess, K,
                 name,
                 input_to_DN=None,
                 **kwargs):

        self.K = K
        self.c = BN_model.class_num
        self.BN = BN_model
        self.sess = sess
        self.name = name
        if name is None:
            self.name = BN_model+'_WDN'    
        
        # take the last hidden layer of BN_model
        # if no alternative input to DN is provided
        if input_to_DN is None:
            if len(BN_model.var_dict)==0:
                input_to_DN = BN_model.output

            else:
                probed_inputs = list(BN_model.probes[0].keys())

                assert len(probed_inputs)>0, \
                    'BN should be probed in the input of the'+\
                    ' last hidden layer.'
                assert len(probed_inputs)==1, \
                    'There is ambiguity in which input-probe '+\
                    'is the input to the last layer'

                input_to_DN = BN_model.probes[0][probed_inputs[0]]
            
        # construct the DN models
        self.DN_dict = {}
        for k in range(K):
            layers_dict = {'last': ['fc', [self.c], 'M']}
            self.DN_dict['DN_{}'.format(k)] = NN_extended.CNN(
                input_to_DN, 
                layers_dict, '{}/DN_{}'.format(self.name,k),
                **kwargs)
            self.DN_dict['DN_{}'.format(k)].get_optimizer()

        self.DN_train_ops = [self.DN_dict['DN_{}'.format(k)].train_step
                             for k in range(K)]

    def initialize(self, BN_too=False):
        """Initializing all the models in the class
        """
        
        if BN_too:
            self.BN.initialize(self.sess)

        for _, DN in self.DN_dict.items():
            DN.initialize(self.sess)
        self.sess.run(tf.variables_initializer([self.av_logits,
                                                self.av_logits_global_step]))


    def train_DNs(self, eternal_gens, max_iter):
        """Training DNs using eternal generators per annotator
        """

        for _ in range(max_iter):
            for k, gen in enumerate(eternal_gens):
                Xb,Zb,_ = next(gen)
                feed_dict = {
                    self.BN.x: Xb,
                    self.BN.keep_prob: 1.,
                    self.DN_dict['DN_{}'.format(k)].y_: Zb[k],
                    self.DN_dict['DN_{}'.format(k)].keep_prob: 
                    1. - self.DN_dict['DN_{}'.format(k)].dropout_rate}

                self.sess.run(self.DN_train_ops[k], feed_dict=feed_dict)

    def get_optimizer_av_logits(self, 
                                eps=1e-6, 
                                optimizer_name='SGD',
                                lr_schedule=None):
        """Setting up the loss function and optimizer for
        training the averaging logits
        """

        with tf.variable_scope(self.name):
            self.av_logits = tf.get_variable('av_logits',
                                             initializer=tf.constant(
                                                 0., shape=[self.K]))
            posts_stack = tf.stack([self.DN_dict['DN_{}'.format(k)].posteriors
                                    for k in range(self.K)], axis=-1)
            # the following is a matrix-vector element-wise multiplication
            # when the vector is repeated along the rows to have the
            # save size of te matrix (repetition is not done explicitly
            # here, but through the broadcasting feature of multiplication)
            self.weighted_av_pred = tf.reduce_sum(
                posts_stack * tf.nn.softmax(self.av_logits), axis=-1)

            clipped_preds = tf.clip_by_value(self.weighted_av_pred, eps, 1-eps)

            # plceholder for the target distribution
            self.target_hist = tf.placeholder(tf.float32,  self.weighted_av_pred.shape)

            # define the loss function manually
            self.av_logits_loss = -tf.reduce_mean(
                tf.reduce_sum(
                    self.target_hist * tf.log(clipped_preds), axis=0))

            # the optimizer
            if lr_schedule is None:
                lr_schedule = self.DN_dict['DN_0'].learning_rate

            if optimizer_name=='SGD':
                self.av_logits_optimizer = tf.train.GradientDescentOptimizer(
                    lr_schedule, name='av_logits_SGD')
            elif optimizer_name=='Adam':
                self.av_logits_optimizer = tf.train_AdamOptimizer(
                    lr_schedule, name='av_logits_Adam')

            # getting the training step
            grad_var = self.av_logits_optimizer.compute_gradients(
                self.av_logits_loss, self.av_logits)
            self.av_logits_global_step = tf.Variable(0, trainable=False,
                                                     name='av_logits_global_step')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.av_logits_train_step = self.av_logits_optimizer.apply_gradients(
                    grad_var, global_step=self.av_logits_global_step)
            

    def train_av_logits(self, eternal_gen, max_iter):
        """Performing iterations of training the averaging logits
        """
        
        feed_dict = {MODEL[1].keep_prob: 1. for MODEL in self.DN_dict.items()}
        feed_dict[self.BN.keep_prob] = 1.

        for _ in range(max_iter):
            Xb, Zb, _ = next(eternal_gen)
            # get the target histogram from labelers' labels
            # Although Zb is a list, not an array, the following
            # adds all the arrays in this list elementwise
            target_hist_val = np.sum(Zb, axis=0)
            target_hist_val = target_hist_val / np.sum(target_hist_val, axis=0)


            feed_dict[self.BN.x] = Xb
            feed_dict[self.target_hist] = target_hist_val
            self.sess.run(self.av_logits_train_step, feed_dict=feed_dict)


def eval_DN(wdn_model, non_eternal_gen, DN_idx):
    """Evaluating DN (Doctor-Net) in a `WDN_Guan`
    class object

    Index of the DN (`DN_idx`) should be given in string format.
    """

    preds = []
    grounds = []
    
    DN_model = wdn_model.DN_dict['DN_{}'.format(DN_idx)]

    for Xb, Yb,_ in non_eternal_gen:
        feed_dict={wdn_model.BN.x:Xb,
                   wdn_model.BN.keep_prob: 1.,
                   DN_model.keep_prob: 1.}
        preds += [wdn_model.sess.run(DN_model.prediction, feed_dict=feed_dict)]

        if isinstance(Yb, list):
            Yb = Yb[int(DN_idx)]
        grounds += [np.argmax(Yb, axis=0)]
    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    acc = np.sum(preds==grounds) / len(preds)
    return acc, preds


def eval_average_DN(wdn_model, non_eternal_gen, weights=None):
    """Evaluating predictions obtianed by (weighted)
    averaging of an ensemble of models

    We assume the input weights sum to one (hence a PMF).
    """

    K = wdn_model.K
    if weights is None:
        weights = np.ones(K) / K

    preds = []
    grounds = []
    feed_dict = {MODEL[1].keep_prob:1. for MODEL in wdn_model.DN_dict.items()}
    feed_dict[wdn_model.BN.keep_prob] = 1.
    for Xb, Yb, _ in non_eternal_gen:
        feed_dict[wdn_model.BN.x] = Xb

        # average posterior
        posts = 0
        for k, M in enumerate(wdn_model.DN_dict.items()):
            posts += weights[k] * wdn_model.sess.run(M[1].posteriors, 
                                                     feed_dict=feed_dict)
        preds += [np.argmax(posts, axis=0)]
        grounds += [np.argmax(Yb, axis=0)]

    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    acc = np.sum(preds==grounds) / len(preds)

    return acc, preds


def eval_WDN_model(wdn_model, dat_gen):

    feed_dict = {MODEL[1].keep_prob:1. for MODEL in wdn_model.DN_dict.items()}
    feed_dict[wdn_model.BN.keep_prob] = 1.

    preds = []
    grnds = []
    for Xb,Yb,_ in dat_gen:
        feed_dict[wdn_model.BN.x] = Xb
        av_posts = wdn_model.sess.run(wdn_model.weighted_av_pred, 
                                      feed_dict=feed_dict)
        preds += [np.argmax(av_posts, axis=0)]
        grnds += [np.argmax(Yb, axis=0)]

    preds = np.concatenate(preds)
    grnds = np.concatenate(grnds)
    acc = np.sum(preds==grnds) / len(preds)
    
    return acc, preds
