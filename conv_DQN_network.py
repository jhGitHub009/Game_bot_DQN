import numpy as np
import tensorflow as tf
import random
from collections import deque
from Convolutional_DQN import DQN

import gym
from gym import wrappers
from gym.envs.registration import register

class DQN:
    def __init__(self, session, input_row,input_col, output_size, conv1_filter_size=2, conv2_filter_size=3, conv3_filter_size=4, l_rate=0.2, name="main"):
        self.session = session
        self.input_row = input_row
        self.input_col = input_col
        self.output_size = output_size
        self.net_name = name
        self._build_network(conv1_filter_size,conv2_filter_size,conv3_filter_size,l_rate)

    def _build_network(self, conv1_filter_size, conv2_filter_size,conv3_filter_size, l_rate):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_row,self.input_col], name="input_x")
            self._X_img = tf.reshape(self._X, [-1, self.input_row,self.input_col, 1])

            # First layer of weights
            self._conv1 = tf.layers.conv2d(inputs=self._X_img, filters=conv1_filter_size, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            self._conv2 = tf.layers.conv2d(inputs=self._conv1, filters=conv2_filter_size, kernel_size=[3, 3], padding="SAME",activation=tf.nn.relu)
            self._conv3 = tf.layers.conv2d(inputs=self._conv2, filters=conv3_filter_size, kernel_size=[3, 3],padding="SAME", activation=tf.nn.relu)
            self._flat = tf.reshape(self._conv3, [-1, conv3_filter_size * self.input_row * self.input_col])
            self._dense1 = tf.layers.dense(inputs=self._flat,units=128, activation=tf.nn.relu)
            self._dense2 = tf.layers.dense(inputs=self._dense1, units=self.output_size, activation=tf.nn.relu)
            # Q prediction
            self._Qpred = tf.layers.dense(inputs=self._dense2, units=self.output_size,activation=tf.nn.softmax)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        # Loss function
        # self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._dense2, labels=self._Y))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        self._saver = tf.train.Saver()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_row, self.input_col])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

# env = gym.make('Breakout-v0')

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)
env = gym.make('CartPole-v2')

# Constants defining our neural network
input_row = 1
input_col = 1
for i,dim in enumerate(env.observation_space.shape):
    if i==0:
        input_row=dim
    else:
        input_col = input_col*dim
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 20000

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0,mainDQN.input_row,mainDQN.input_col)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, np.reshape(state, [1,mainDQN.input_row, mainDQN.input_col])])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 500001
    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_row,input_col, output_size, name="main")
        targetDQN = DQN(sess, input_row,input_col, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            # e = 1. / ((episode / 10) + 1)
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            sum_reward = 0
            while not done:
                # env.render()
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done: # Penalty
                    reward = -100
                    sum_reward = reward + sum_reward
                    env.reset()
                else:
                    sum_reward = reward + sum_reward
                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:   # Good enough. Let's move on
                    break

            print("Episode: {} steps: {} sum_reward:{}".format(episode, step_count,sum_reward))
            if step_count > 10000:
                pass
                # break

            if episode % 50 == 1: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
                mainDQN._saver.save(sess=sess, save_path='./output/mainDQN_conv', global_step=episode,latest_filename=None)
                targetDQN._saver.save(sess=sess, save_path='./output/targetDQN_conv', global_step=episode,latest_filename=None)
                print("Loss: ", loss)
                sess.run(copy_ops)

        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'gym-results', force=True)
        for i in range(5):
            bot_play(mainDQN, env=env2)
        env2.close()
def show_game_play(main_save):
    with tf.Session() as sess:
        # tf.reset_default_graph()
        mainDQN = DQN(sess, input_size, output_size, name="main")
        mainDQN._saver.restore(sess=sess,save_path=main_save)
        env2 = wrappers.Monitor(env, 'output', force=True)
        for i in range(5):
            bot_play(mainDQN, env=env2)
        env2.close()
if __name__ == "__main__":
    main()