import gym
import math
import numpy as np

# random action
# env = gym.make('CartPole-v0')
# for i_episode in range(200):
#     observation = env.reset()
#     rewards = 0
#     for t in range(250):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         rewards += reward
#         if done:
#             print(f'episode finished after {t+1} timesteps, '
#                   f'total rewards is {reward}')
#             break
# env.close()


# hand-made policy
# def choose_action(observation):
#     pos, v, ang, rot = observation
#     return 0 if ang < 0 else 1


# env = gym.make('CartPole-v0')
# for i_episode in range(200):
#     observation = env.reset()
#     rewards = 0
#     for t in range(250):
#         env.render()
#         action = choose_action(observation)
#         observation, reward, done, info = env.step(action)
#         rewards += reward
#         if done:
#             print(f'episode finished after {t+1} timesteps, '
#                   f'total rewards is {reward}')
#             break
# env.close()


# q-table
def choose_action(state, q_table, action_space, epsilon):
    poss = np.random.random_sample()
    if poss > epsilon:
        return action_space.sample()
    else:
        return np.argmax(q_table[state])


def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1]
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])
        return tuple(state)


env = gym.make('CartPole-v0')

# 準備 Q table
# Environment 中各個 feature 的 bucket 分配數量
# 1 代表任何值皆表同一 state，也就是這個 feature 其實不重要
n_buckets = (1, 1, 6, 3)

# Action 數量
n_actions = env.action_space.n

# State 範圍
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# Q table，每個 state-action pair 存一值
q_table = np.zeros(n_buckets + (n_actions,))

# 一些學習過程中的參數


def get_epsilon(i): return max(
    0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減


def get_lr(i): return max(
    0.01, min(0.5, 1.0 - math.log10((i+1)/25)))  # learning rate; 隨時間遞減


gamma = 0.99  # reward discount factor

# Q-learning
for i_episode in range(200):
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)

    observation = env.reset()
    rewards = 0
    state = get_state(observation, n_buckets, state_bounds)  # 將連續值轉成離散
    for t in range(250):
        env.render()

        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, info = env.step(action)

        rewards += reward
        next_state = get_state(observation, n_buckets, state_bounds)

        # 更新 Q table
        # 進入下一個 state 後，預期得到最大總 reward
        q_next_max = np.amax(q_table[next_state])
        q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)])  # 就是那個公式

        # 前進下一 state
        state = next_state

        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
            break

env.close()
