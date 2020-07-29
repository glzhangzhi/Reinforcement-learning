import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
	env.render()
	env.step(env.action_space.sample())

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print(f'Episode finished after {t+1} times steps')
			break

import gym
env = gym.make('CartPole-v0')
# print(env.action_space)
# print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)