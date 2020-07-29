import numpy as np
import copy
from time import sleep
import matplotlib.pyplot as plt
from pickle import dump, load


gamma = 0.9
alpha = 0.9
epsilon = 0.9

action = ['u', 'd', 'l', 'r']

env = [['-', '-', '-', '-', '-'],
	   ['-', '-', '-', '-', '-'],
	   ['-', '-', '-', '-', '-'],
	   ['-', '-', '-', '-', '-'],
	   ['-', '-', '-', '-', 't']]

for i in env:
	if 't' in i:
		for j in i:
			if j == 't':
				x = env.index(i)
				y = i.index(j)

target = [x, y]

# q_table = [[[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]],
# 		   [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]],
# 		   [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]],
# 		   [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]],
# 		   [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]]

with open('q_table.pickle', 'rb') as f:
	q_table = load(f)
	print('数据载入完毕')

def choose_action(s, q_table, epsilon):
	state_action = q_table[s[0]][s[1]]
	if np.random.uniform() > epsilon or 0 in state_action:
		idx = np.random.randint(0, 4)
	else:
		idx = state_action.index(max(state_action))
	a = action[idx]
	print(a)
	return a


def feed_back(s, a):
	s_ = copy.deepcopy(s)
	if a == 'l' and s[1] > 0:
		s_[1] -= 1
	if a == 'r' and s[1] < 4:
		s_[1] += 1
	if a == 'u' and s[0] > 0:
		s_[0] -= 1
	if a == 'd' and s[0] < 4:
		s_[0] += 1

	if s_ == target:
		r = 1
	else:
		r = 0
	return s_, r

def rl(q_table, a, s, r, s_, a_):
	global win
	idx_a = action.index(a)
	idx_a_ = action.index(a_)
	q_predict = q_table[s[0]][s[1]][idx_a]
	q_ = q_table[s_[0]][s_[1]][idx_a_]
	if r == 1:
		q_target = r
		win = True
	else:
		q_target = gamma * q_
	q_table[s[0]][s[1]][idx_a] += alpha * (q_target - q_predict)
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
for episode in range(1000):
	s = [0, 0]
	win = False
	counter = 0
	while win == False:
		a = choose_action(s, q_table, epsilon)
		# if epsilon < 1:
		# 	epsilon = epsilon * 1.005
		# 	if epsilon >= 1:
		# 		epsilon = 1
		s_, r = feed_back(s, a)
		a_ = choose_action(s_, q_table, epsilon)
		q_table = rl(q_table, a, s, r, s_, a_)
		s = s_
		a = a_
		counter += 1
		# show(env, s)
		# sleep(0.3)
	C.append(counter)
	print(episode, counter)
with open('q_table.pickle', 'wb') as f:
	dump(q_table, f)
	print('数据保存完毕')
plt.plot(list(range(1, len(C) + 1)), C, linestyle='-')
plt.xlabel('iter')
plt.ylabel('step')
plt.show()

