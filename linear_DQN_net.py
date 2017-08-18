import numpy as np
import tensorflow as tf
import random
from collections import deque

import gym
from gym import wrappers
from gym.envs.registration import register

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=50, l_rate=0.1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            # Second layer of Weights
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],initializer=tf.contrib.layers.xavier_initializer())
            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)
        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        self._saver = tf.train.Saver()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

# env = gym.make('Breakout-v0')
# env = gym.make('BreakoutDeterministic-v4')

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)
env = gym.make('CartPole-v2')

# Constants defining our neural network
input_size = 1
for dim in env.observation_space.shape:
    input_size = input_size*dim
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 20000

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state.reshape(input_size))

        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            Q[0, action] = reward + dis * targetDQN.predict(next_state.reshape(input_size))[0, np.argmax(mainDQN.predict(next_state.reshape(input_size)))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, np.reshape(state.reshape(input_size), [1, mainDQN.input_size])])

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
    step = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state.reshape(input_size)))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 50000
    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
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
                    action = np.argmax(mainDQN.predict(state.reshape(input_size)))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                if done: # Penalty
                    # state = env.reset()
                    reward = -100
                    sum_reward = reward + sum_reward
                else:
                    sum_reward = reward + sum_reward

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                # bot_play(mainDQN, env=env)
                if step_count > 10000:   # Good enough. Let's move on
                    break

            print("Episode: {} steps: {} sum_reward:{}".format(episode, step_count, sum_reward))
            if step_count > 10000:
                pass
                # break

            if episode % 50 == 10: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
                mainDQN._saver.save(sess=sess, save_path='./output/mainDQN_linear', global_step=episode,latest_filename=None)
                targetDQN._saver.save(sess=sess, save_path='./output/targetDQN_linear', global_step=episode,latest_filename=None)
                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)
        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'output', force=True)
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