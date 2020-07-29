import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

n_states = 20  # 状态空间的大小
actions = ['left', 'right']  # 所有可能的动作
epsilon = 0.9  #贪婪程度
alpha = 0.1  # 学习率
gamma = 0.9  # 奖励递减值
max_episodes =   50  # 最大训练次数
fresh_time = 0.05  # 移动间隔时间

def build_q_table(n_states, actions):
	'''
	根据已知的状态空间大小和动作集合，生成空白的Q值表

	'''
	table = pd.DataFrame(
		np.zeros((n_states, len(actions))),
		columns = actions,
		)
	return table


def choose_action(state, q_table):
	'''
	根据已有的Q值表选择给定状态下最佳的动作

	'''
	state_action = q_table.iloc[state, :]
	# 列出当前状态下所有可能的动作
	if (np.random.uniform() > epsilon or state_action.all() == 0):
	# 如果贪婪指数高过阈值或者该状态下有未被探索过的动作
		action_name = np.random.choice(actions)
		# 随机选择一个动作
	else:
		action_name = state_action.idxmax()
		# 选择Q值最大的动作
	return action_name


def get_env_feedback(s, a):
	'''
	环境对于目标现在的状态和即将采取的动作给与反应，即给出应用该动作后的反应
	以及采取该动作后的奖励值

	如果下一个状态可以拿到宝藏，则奖励为1，其余奖励为0

	'''
	if a == 'right':
		if s == n_states - 2:
			s_ = 'terminal'
			r = 1
		else:
			s_ = s + 1
			r = 0
	else:
		r = 0
		if s == 0:
			s_ = s
		else:
			s_ = s - 1
	return s_, r


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(n_states-1) + ['T']   # '-----T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(fresh_time)


def rl():
	q_table = build_q_table(n_states, actions)  # 生成空白Q值表
	step_list = []
	for episode in range(max_episodes):
		step_counter = 0  # 初始化计步器
		s = 0  # 设定每一局开始时所在位置
		is_terminated = False  # 将获胜状态改为否
		update_env(s, episode, step_counter)  # 根据目前状态绘制界面
		while not is_terminated:
			a = choose_action(s, q_table)  # 根据目前的状态和Q值表选择动作
			s_, r = get_env_feedback(s, a)  # 环境根据目前的状态和要执行的动作改变状态并给与奖励
			q_predict = q_table.loc[s, a]
			# Q值的预测值为当前状态和动作对应的Q值，如果epsilon=1，理论上说这个值应该等于现在这个状态对应最大的Q值
			if s_ != 'terminal':
				q_target = r + gamma * q_table.iloc[s_, :].max()
				# 如果下一步没有拿到宝藏，则目标值等于下一步行动可能获得的最大的Q值乘以gamma
			else:
				q_target = r  # 如果下一步行动能够拿到宝藏，则目标值=1
				is_terminated = True  # 并且结束游戏
			q_table.loc[s, a] += alpha * (q_target - q_predict)  # 使用预测值和目标值的误差更新Q值表
			s = s_  # 改变现在的状态

			update_env(s, episode, step_counter + 1)  # 更新环境
			step_counter += 1  #计步器+1
		step_list.append(step_counter)
	return q_table, step_list

if __name__ == '__main__':
	q_table, step_list = rl()
	print('\r\nQ-table:\n')
	print(q_table)
	plt.plot(list(range(1, len(step_list) + 1)), step_list, linestyle='-')
	plt.show()
