import gym
from gym import wrappers

env = gym.make('Breakout-v0')
env = wrappers.Monitor(env, 'gym-results', force=True)
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 100:
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    #obervation- [31,3]
    print(action,observation, reward, done)
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
env.close()
