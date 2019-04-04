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
        self.model = model
        self.sess = sess
        # CL: crowd layers
        self.CL_dict = {}

        # constructing crowd layers
        for k in range(self.K):
            CL = NN_extended.CNN(self.model.posteriors, {}, 
                                 self.model.name+'_CL_{}'.format(k))
            with tf.variable_scope(CL.name):
                W_CL = NN_extended.weight_variable('Weight', 
                                                   [self.c, self.c],
                                                   CL.regularizer,
                                                   CL.custom_getter)
                CL.output = tf.matmul(W_CL, CL.output)
                if len(CL.dropout_layers) > 0:
                    CL.output = tf.nn.dropout(CL.output, CL.keep_prob)
                    
                # update the prediction and posterior ops
                CL.posteriors = tf.transpose(
                    tf.nn.softmax(tf.transpose(CL.output)))
                CL.prediction = tf.argmax(CL.posteriors, 0)
                CL.var_dict['last'] = [W_CL]

                self.CL_dict['CL_{}'.format(k)] = CL

    def get_optimizer(self, optimizer_name='SGD', 
                      trace_reg=False,
                      **opt_kwargs):
        """Setting up the loss and optimizer for the model

        The loss of this model is the sum of individual CE losses
        in all the crowd layers.
        """

        self.CL_losses = {}

        with tf.variable_scope(self.model.name):

            if trace_reg:
                self.tr_reg_lambda = tf.placeholder(tf.float32)
            else:
                self.tr_reg_lambda = tf.constant(0.)

            # the loss
            self.masked_y = []
            self.masked_output = []
            for k in range(self.K):
                CL = self.CL_dict['CL_{}'.format(k)]

                # filter one-hot label placeholder
                y_mask = tf.not_equal(tf.reduce_sum(CL.y_, axis=0),0)
                self.masked_y += [tf.boolean_mask(CL.y_, y_mask, axis=1)]
                self.masked_output += [tf.boolean_mask(CL.output, y_mask, axis=1)]

                self.CL_losses['CL_{}'.format(k)] = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.transpose(self.masked_y[k]), 
                    logits=tf.transpose(self.masked_output[k])) + \
                    self.tr_reg_lambda * tf.trace(CL.var_dict['last'][0])

            self.loss = tf.reduce_sum([tf.reduce_mean(loss) for _,loss in self.CL_losses.items()])

            # the optimizer
            if optimizer_name=='SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(**opt_kwargs)
            elif optimizer_name=='Adam':
                self.optimizer = tf.train.AdamOptimizer(**opt_kwargs)

            # the train step
            # variables of the main body
            tr_vars = [v for _,v in self.model.var_dict.items()] 
            # variables of the crowd layers
            tr_vars += [[v for _,v in MM.var_dict.items()] 
                        for _,MM in self.CL_dict.items()]

            grads_vars = self.optimizer.compute_gradients(self.loss, tr_vars)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.apply_gradients(
                    grads_vars, global_step=self.model.global_step)
        

    def initialize(self, identity_crowd_layer=True):
        """Initializing parameters and optimization variables
        """

        self.model.initialize(self.sess)

        if identity_crowd_layer:
            # initialize the matrix weights of crowd layers 
            # with identities
            for k in range(self.K):
                self.sess.run(tf.assign(
                    self.CL_dict['CL_{}'.format(k)].var_dict['last'][0], 
                    np.eye(self.c)))
