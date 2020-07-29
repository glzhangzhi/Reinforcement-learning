import numpy as np
import copy
from time import sleep
import matplotlib.pyplot as plt


gamma = 0.9
alpha = 0.9
epsilon = 0.9

action = ['u', 'd', 'l', 'r']

w = 4
h = 4
t_w = 4
t_h = 4
n_actions = len(action)

env = np.full((w, h), '-')
env[t_w - 1][t_h - 1] = 't'

target = [t_w, t_h]

q_table = np.zeros((w, h, n_actions))


def choose_action(s, q_table, epsilon):
    state_action = q_table[s[0]][s[1]]
    if np.random.uniform() > epsilon or 0 in state_action:
        idx = np.random.randint(0, 4)
    else:
        idx = state_action.index(max(state_action))
    a = action[idx]
    return a


def feed_back(s, a):
    s_ = copy.deepcopy(s)
    if a == 'l' and s[1] > 0:
        s_[1] -= 1
    if a == 'r' and s[1] < 2:
        s_[1] += 1
    if a == 'u' and s[0] > 0:
        s_[0] -= 1
    if a == 'd' and s[0] < 2:
        s_[0] += 1

    if s_ == target:
        r = 1
    else:
        r = 0
    return s_, r


def rl(q_table, a, s, r, s_):
    global win
    idx = action.index(a)
    q_predict = q_table[s[0]][s[1]][idx]
    if r == 1:
        q_target = r
        win = True
    else:
        q_target = gamma * max(q_table[s_[0]][s_[1]])
    q_table[s[0]][s[1]][idx] += alpha * (q_target - q_predict)
    return q_table


def show(env, s):
    now_env = copy.deepcopy(env)
    now_env[s[0]][s[1]] = '我'
    for i in now_env:
        for j in i:
            print(j, end=' ')
        print('\n')
    print('-----------------------------')


C = []
for episode in range(20):
    s = [0, 0]
    win = False
    counter = 0
    while win == False:
        a = choose_action(s, q_table, epsilon)
        # if epsilon < 1:
        # 	epsilon = epsilon * 1.05
        # 	if epsilon >= 1:
        # 		epsilon = 1
        s_, r = feed_back(s, a)
        q_table = rl(q_table, a, s, r, s_)
        s = s_
        counter += 1
        # show(env, s)
        # sleep(0.1)
    C.append(counter)

plt.plot(list(range(1, len(C) + 1)), C, linestyle='-')
plt.xlabel('iter')
plt.ylabel('step')
plt.show()
