import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

batch_size = 32
lr = 0.01
epsilon = 0.9
gamma = 0.9
target_replace_iter = 100
memory_capaicity = 200
env = gym.make('CartPole-v0')
env = env.unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capaicity, n_states * 2 + 2))
        # 每条记忆的大小为当前状态n_state+行为1+奖励1+下一步状态n_state
        # self.optimizer = torch.optim.SGD(self.target_net.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters())
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)

        if np.random.uniform() < epsilon:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0, 0]
        else:
            action = np.random.randint(0, n_actions)
        return action

    def store_transiton(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memory_capaicity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(memory_capaicity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :n_states])
        b_a = torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_states:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0]
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer()


dqn = DQN()
for i_episode in range(400):
    s = env.reset()
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        x, d_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transiton(s, a, r, s_)

        if dqn.memory_counter > memory_capaicity:
            dqn.learn()

        if done:
            break

        s = s_
