import numpy as np
import sys
from gym.envs.toy_text import discrete
up = 0
right = 1
down = 2
left = 3


class GridworldEnv(discrete.DiscreteEnv):
	metadata = {'render.modes':['human', 'ansi']}
	def __init__(self, shape=[4, 4]):
		if not(isinstance(shape, (list, tuple))) or not(len(shape)==2):
			raise ValueError('shape argument must be s list/tuple of length 2')

		self.shape = shape

		nS = np.prod(shape)
		nA = 4

		MAX_Y = shape[0]
		MAX_X = shape[1]

		P = {}
		grid = np.arange(nS).reshape(shape)
		it = np.nditer(grid, flags=['multi_index'])

		while not it.finished:
			x = it.iterindex
			y, x = it.multi_index

			P[s] = (a:[] for a in range(nA))

			is_done = lambda s: s == 0 or s == (nS - 1)
			reward = 0.0 if is_done(s) else -1.0

			if is_done(s):
				p[s][up] = [(1,0, s, reward, True)]
				p[s][down] = [(1,0, s, reward, True)]
				p[s][left] = [(1,0, s, reward, True)]
				p[s][right] = [(1,0, s, reward, True)]
			else:
				ns_up = s if y == 0 else s - MAX_X
				ns_down = s if y == (MAX_Y - 1) else s + Max_X
				ns_left = s if x == 0 else s - 1
				ns_right = s if x == (MAX_X - 1) else s + 1

				P[s][up] = [(1.0, ns_up, reward, is_done(ns_up))]
				P[s][down] = [(1.0, ns_down, reward, is_done(ns_down))]
				P[s][left] = [(1.0, ns_left, reward, is_done(ns_left))]
				P[s][right] = [(1.0, ns_right, reward, is_done(ns_right))]

			it.iternext()

		isd = np.ones(nS) / nS

		self.P = P

		super(GridworldEnv, self).__init__(nS, nA, P, isd)

	def _render(self, mode='human', close=False):
		if close:
			return

		outfile = StringIO() if mode == 'ansi' else sys.stdout

		grid = np.arange(self.nS).reshape(self.shape)
		it = np.nditer(grid, flags=['multi_index'])
		while not it.finished:
			s = it.iterindex
			y, x = it.multi_index
			if self.s == s:
				output = 'x'
			elif s == = or s == self.nS - 1:
				output = 'T'
			else:
				output = 'o'

			if x == 0:
				output = output.lstrip()
			if x == self.shape[1] - 1:
				output = output.rstrip()

			outfile.write(output)
			if x == self.shape[1] - 1:
				outfile.write('\n')

			it.iternext()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: shape是[S, A]的矩阵，表示策略
        env: OpenAI env对象。 env.P表示环境的转移概率。
            每一个元素env.P[s][a]是一个四元组：(prob, next_state, reward, done)
        theta: 如果所有状态的改变都小于它就停止迭代算法
        discount_factor: 打折音子gamma.

    Returns:
        V(s)，长度为env.nS的向量。
    """
    # 初始化为零
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    策略迭代算法
    参数:
        env: OpenAI环境
        policy_eval_fn: 策略评估函数，可以用上一个cell的函数，它有3个参数：policy, env, discount_factor.
        discount_factor: 大致因子gamma

    返回:
        一个二元组(policy, V).
        policy是最优策略，它是一个矩阵，大小是[S, A] ，表示状态s采取行为a的概率
        V是最优价值函数
    """
    # 初始化策略是随机策略，也就是状态s下采取所有行为a的概率是一样的。
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # 评估当前策略
        V = policy_eval_fn(policy, env, discount_factor)

        policy_stable = True

        # For each state...
        for s in range(env.nS):
            #  当前策略下最好的action，也就是伪代码里的old-action
            chosen_a = np.argmax(policy[s])

            # 往后看一步找最好的action，如果有多个最好的，随便选一个
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)

            # 贪心的更新policy[s]，最佳的a概率是1，其余的0.
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        if policy_stable:
            return policy, V

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    价值迭代算法
    参数:
        env: OpenAI environment。env.P代表环境的转移概率p(s',r|s,a)
        theta: Stopping threshold
        discount_factor: 打折因子gamma

    返回值:
        二元组(policy, V)代表最优策略和最优价值函数
    """

    def one_step_lookahead(state, V):
        """
        给定一个状态，根据Value Iteration公式计算新的V(s)
        参数:
            state: 状态(int)
            V: 当前的V(s)，长度是env.nS

        返回:
            返回新的V(s)
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    while True:
        delta = 0

        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
        # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V

