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

class model(object):

    def __init__(self, model, sess, K, name):

        self.K = K
        self.c = model.class_num
        self.model = model
        self.sess = sess
        self.name = name

    def build_crowd_layers(self):
        """Modeling all the crowd layers in a single tensor weight
        """

        shape_W = [self.K, self.c, self.c]
        shape_b = [self.K, self.c, 1]
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # parameters of the crowd layers
            init_W = np.stack([np.eye(self.c, dtype=np.float32) 
                               for k in range(self.K)], axis=0)
            self.crowd_W = tf.get_variable('crowd_W', initializer=init_W)
            init_b = tf.constant(0., shape=shape_b)
            self.crowd_b = tf.get_variable('crowd_b', initializer=init_b)

            # output of the crowd layers 
            # (K x 2 x b)
            self.crowd_output = tf.keras.backend.dot(
                self.crowd_W, self.model.posteriors) + self.crowd_b
            self.crowd_posteriors = tf.nn.softmax(self.crowd_output,
                                                  axis=1)
            
            # get the target placeholder for the crowds
            self.crowd_target = tf.placeholder(tf.float32, 
                                               self.crowd_output.shape)


    def get_optimizer(self, optimizer_name='SGD', 
                      trace_reg=False,
                      **opt_kwargs):
        """Setting up the loss and optimizer for the model

        The loss of this model is the sum of individual CE losses
        in all the crowd layers.
        """

        with tf.variable_scope(self.name):

            self.global_step = tf.Variable(0, 
                                           trainable=False, 
                                           name='global_step')
            # the loss
            vec = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.crowd_output,
                labels=self.crowd_target,
                dim=1)            
            mask = tf.equal(self.crowd_target[:,0,:], -1)
            zer = tf.zeros_like(vec)
            self.loss = tf.where(mask, x=zer, y=vec)

            # the optimizer
            if optimizer_name=='SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(**opt_kwargs)
            elif optimizer_name=='Adam':
                self.optimizer = tf.train.AdamOptimizer(**opt_kwargs)

            # the train step
            # variables of the main body
            train_vars = [V[i] for _,V in self.model.var_dict.items()
                          for i in range(len(V)) if V[i] in 
                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            train_vars += [self.crowd_W, self.crowd_b]
            self.grads_vars = self.optimizer.compute_gradients(self.loss, train_vars)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.apply_gradients(
                    self.grads_vars, global_step=self.global_step)
        

    def initialize_crowd_model(self):
        """Initializing parameters and optimization variables
        """

        # collect all the optimization parameters
        var_list = [V for V in tf.global_variables() if 
                    self.name in V.name]
        self.sess.run(tf.variables_initializer(var_list))

