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
                 DN_lr=None,
                 name=None):

        self.K = K
        self.c = BN_model.class_num
        self.BN = BN_model
        self.sess = sess
        self.name = name
        if name is None:
            self.name = BN_model+'_WDN'

        # get the same learning rate, if no other is available
        if DN_lr is None:
            DN_lr = BN_model.learning_rate

        # construct the DN models
        self.DN_dict = {}
        for k in range(K):
            layers_dict = {'last': ['fc', [self.c], 'M']}
            self.DN_dict['DN_{}'.format(k)] = NN_extended.CNN(
                BN_model.output, layers_dict, '{}/DN_{}'.format(self.name,k),
                lr_schedule=DN_lr, loss_name='CE_softclasses')
            self.DN_dict['DN_{}'.format(k)].get_optimizer()

        self.DN_train_ops = [self.DN_dict['DN_{}'.format(k)].train_step
                             for k in range(K)]

    def initialize(self):
        """Initializing all the models in the class
        """
        self.BN.initialize(self.sess)
        for _, DN in self.DN_dict.items():
            DN.initialize(self.sess)
        self.sess.run(tf.variables_initializer([self.av_logits,
                                                self.av_logits_global_step]))


    def train_DNs(self, train_dat_gen, max_iter):

        for _ in range(max_iter):
            Xb,Zb,_ = next(train_dat_gen)
            feed_dict = {self.BN.x: Xb}
            for k in range(self.K):
                feed_dict.update({
                    self.DN_dict['DN_{}'.format(k)].y_: Zb[k]})

            self.sess.run(self.DN_train_ops, feed_dict=feed_dict)

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
            

    def train_av_logits(self, train_gen, max_iter):
        """Performing iterations of training the averaging logits
        """
        
        for _ in range(max_iter):
            Xb, Zb, _ = next(train_gen)
            # get the target histogram from labelers' labels
            # Although Zb is a list, not an array, the following
            # adds all the arrays in this list elementwise
            target_hist_val = np.sum(Zb, axis=0)
            target_hist_val = target_hist_val / np.sum(target_hist_val, axis=0)

            self.sess.run(self.av_logits_train_step, 
                          feed_dict={self.BN.x:Xb,
                                     self.target_hist: target_hist_val})
        

def eval_average_of_models(models, sess, dat_gen, weights=None):
    """Evaluating predictions obtianed by (weighted)
    averaging of an ensemble of models

    We assume the input weights sum to one (hence a PMF).
    """

    if weights is None:
        weights = np.ones(len(models)) / len(models)

    preds = []
    grounds = []
    for Xb, Yb, _ in dat_gen:
        # average posterior
        posts = weights[0] * sess.run(models[0].posteriors, 
                                      feed_dict={models[0].x:Xb})
        for i, model in enumerate(models[1:]):
            posts += weights[i+1] * sess.run(model.posteriors, 
                                             feed_dict={model.x:Xb})
        preds += [np.argmax(posts, axis=0)]
        grounds += [np.argmax(Yb, axis=0)]

    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    acc = np.sum(preds==grounds) / len(preds)

    return acc, preds


def eval_WDN_model(WDN_model, dat_gen):
    preds = []
    grnds = []
    for Xb,Yb,_ in dat_gen:
        av_posts = WDN_model.sess.run(WDN_model.weighted_av_pred, 
                                      feed_dict={WDN_model.BN.x:Xb})
        preds += [np.argmax(av_posts, axis=0)]
        grnds += [np.argmax(Yb, axis=0)]

    preds = np.concatenate(preds)
    grnds = np.concatenate(grnds)
    acc = np.sum(preds==grnds) / len(preds)
    
    return acc, preds
