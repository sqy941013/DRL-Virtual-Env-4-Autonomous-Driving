import tensorflow as tf
import numpy as np
# import gym
# from gym import wrappers
import tflearn
import argparse
import pprint as pp
import sys

from replay_buffer import ReplayBuffer


# ===========================
#   Actor 和 Critic 网络
# ===========================

class ActorNetwork(object):
    """
    将状态输入到Actor网络中，输出是在确定策略（deterministic policy）下的动作。
    输出层激活函数是tanh，使得输出的动作在[-action_bound,action_bound]范围内。
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, noise_option):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.noise_option = noise_option
        self.act_noise = tf.placeholder(tf.bool)

        # Actor Network
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # 固定周期来使用在线网络（online network）来更新目标网络（target network）的权重
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # 由Critic Network计算得到梯度
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # 结合梯度
        self.unnormalized_actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # 优化过程
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    # 创建Actor Network网络结构
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        # 400*400 全连接层
        net = tflearn.fully_connected(inputs, 400, weights_init='xavier', bias_init='zeros')
        # Batch Normalization
        net = tflearn.layers.normalization.batch_normalization(net)
        # ReLU激活层
        net = tflearn.activations.relu(net)
        # 300*300 全连接层
        net = tflearn.fully_connected(net, 300, weights_init='xavier', bias_init='zeros')
        # Batch Normalization
        net = tflearn.layers.normalization.batch_normalization(net)
        # ReLU激活层
        net = tflearn.activations.relu(net)

        # 添加高斯噪声来实现探索（Exploration）
        # 也可以尝试使用其他噪声

        tf_cond = tf.cond(self.act_noise, true_fn=lambda: tf.constant(1.0), false_fn=lambda: tf.constant(0.0))

        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        steering = tflearn.fully_connected(net, 1, activation='tanh', weights_init=w_init)
        acceleration = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=w_init)
        brake = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=w_init)

        out = tflearn.layers.merge_ops.merge([steering, acceleration, brake], mode='concat', axis=1)

        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
            self.act_noise: False
        })

    def predict(self, inputs, noise=False):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs, self.act_noise: noise
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs, self.act_noise: False
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, weights_init='xavier', bias_init='zeros')
        tflearn.add_weights_regularizer(net, 'L2', weight_decay=0.001)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        t1 = tflearn.fully_connected(net, 300, weights_init='xavier', bias_init='zeros')
        t2 = tflearn.fully_connected(action, 300, weights_init='xavier', bias_init='zeros')

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
