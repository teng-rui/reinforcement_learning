import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import argparse
import numpy as np
import gym
import pybullet_envs
import time
import random
import os
import wandb
import tqdm


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Actor-critic agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()

        # Define the critic network and actor_mean. Both of them have two hidden layers with size 64 and use tanh() activation function.

        # Use the orthognoal initialization for the weight and constant for the bias.

        # But for the last layer of critic, set std=1. And for the last layer of actor_mean, set std=0.01.
        # actor_logstd is a learnable value, and its initial value is 0
        self.critic_hidden1 = layer_init(nn.Linear(state_dim, 64))
        self.critic_output = layer_init(nn.Linear(64, 1), std=1)

        self.actor_mean_hidden1 = layer_init(nn.Linear(state_dim, 64))
        self.actor_mean_output = layer_init(nn.Linear(64, action_dim), std=0.01)

        self.actor_logstd = torch.nn.Parameter(torch.tensor([0.0] * action_dim))

    def critic(self, x):
        x = self.critic_hidden1(x)
        x = torch.tanh(x)
        x = self.critic_output(x)
        return x

    def actor_mean(self, x):
        x = self.actor_mean_hidden1(x)
        x = torch.tanh(x)
        x = self.actor_mean_output(x)
        return x

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)  # x-(batch, num_envs, state_dim) action_mean (batch, num_envs, action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (action_dim)->(batch, num_envs, action_dim)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()  # (batch, num_envs, action_dim)
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1)
        # probs.log_prob(action).sum(-1): (batch,num_envs) or (num_envs) if batch==1
        # action (batch, num_envs, action_dim) or (num_envs, action_dim) if batch==1

    def get_value(self, x):
        return self.critic(x).squeeze(-1)  # return shape (batch, num_envs) or (num_envs) if batch==1


# Replay Buffer
class PPOBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 buffer_size,
                 num_envs,
                 gae_lambda,
                 gamma, ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size  # 2048
        self.num_envs = num_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.states, self.actions, self.logprobs, self.rewards, self.start_episodes, self.values = None, None, None, None, None, None
        self.returns, self.advantages = None, None

        self.ptr, self.size = 0, 0
        self.sampler_ready = False

        self.reset()  # initialize replay buffer

    def reset(self, ):
        self.states = np.zeros((self.buffer_size, self.num_envs, self.state_dim), dtype=np.float32)  # 2048*N*state_dim
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.logprobs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.start_episodes = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

        # self.returns ans self.advantages are needed to be calculated and filled in conpute_returns_and_advantages fn
        self.returns = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

        self.ptr, self.size = 0, 0
        self.sampler_ready = False

    def add(self, state, action, reward, start_episode, value, logprob):
        if len(self.states.shape) == 2:  # reset buffer before filling data
            self.reset()

        self.states[self.ptr] = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        self.actions[self.ptr] = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        self.rewards[self.ptr] = reward
        self.start_episodes[self.ptr] = start_episode
        self.values[self.ptr] = value.detach().cpu().numpy() if isinstance(action, torch.Tensor) else value
        self.logprobs[self.ptr] = logprob.detach().cpu().numpy() if isinstance(action, torch.Tensor) else logprob

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(self, last_value: torch.Tensor, done: bool):
        """ Compute the returns and davantage according to GAE formula, see (11) (12) in https://arxiv.org/pdf/1707.06347.pdf """
        # Implement the GAE to claculate the returns and advantages and fill them into buffer.

        last_value = last_value.detach().cpu().numpy()

        gae_advantage = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        last_value[np.where(done == True)] = 0
        running_add = self.rewards[-1] - (self.values[-1]) + (self.gamma * last_value)
        gae_advantage[-1] = running_add
        for t in reversed(range(0, self.buffer_size - 1)):
            running_add = running_add * self.gamma * self.gae_lambda + self.gamma * self.values[t + 1]
            running_add[np.where(self.start_episodes[t] == True)] = 0
            running_add = running_add + (self.rewards[t]) - (self.values[t])

            gae_advantage[t] = running_add

        self.advantages = gae_advantage

        self.returns = self.advantages + self.values

    def get(self, batch_size):
        # Draw samples to train the actor and critic networks. 
        # returns (TD(\lambda)) are the target values used to update the value function.

        # flatten the replay buffer.
        assert self.buffer_size == self.size, "The buffer is not full."  # only train the agent when the buffer is full.
        if len(self.states.shape) == 3:  # flatten the buffer if haven't done yet.
            self._flatten_buffer()

        indices = np.random.permutation(self.buffer_size * self.num_envs)
        start_idx = 0

        while start_idx < self.buffer_size * self.num_envs:  ##why multiplying num_env
            batch_idx = indices[start_idx: start_idx + batch_size]
            data_to_return = (self.states[batch_idx],
                              self.actions[batch_idx],
                              self.logprobs[batch_idx],
                              self.returns[batch_idx],
                              self.advantages[batch_idx])
            yield tuple(map(self._to_tensor, data_to_return))
            start_idx += batch_size

    def _to_tensor(self, arr):
        return torch.from_numpy(arr).to(device)

    def _flatten_buffer(self, ):
        # change the shape of tensor from (buffer_size, num_envs, .) to (buffer_size * num_envs, .)
        # This step is needed before updating weights of nn. 
        self.states = self.states.reshape(-1, self.state_dim)
        self.logprobs = self.logprobs.reshape(-1)
        self.actions = self.actions.reshape(-1, self.action_dim)
        self.advantages = self.advantages.reshape(-1)
        self.returns = self.returns.reshape(-1)
        self.values = self.values.reshape(-1)
        self.rewards = self.rewards.reshape(-1)
        self.start_episodes = self.start_episodes.reshape(-1)


class PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr,
                 num_timesteps_per_env,
                 num_envs,
                 buffer,
                 envs,
                 mini_batch_size,
                 clip_coef,
                 ent_coef,
                 vf_coef,
                 max_grad_norm=0.5,
                 update_epochs=10,  # ?? not 32??
                 ):

        self.agent = Agent(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        self.clip_ratio = clip_coef  # 0.2
        self.num_timesteps_per_env = num_timesteps_per_env  # 2048
        self.buffer = buffer
        self.envs = envs
        self.num_envs = num_envs
        self.mini_batch_size = mini_batch_size
        self.ent_coef = ent_coef  # 0.0
        self.vf_coef = vf_coef  # 0.5
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.lr = lr

        self.total_timesteps = 0  # used to account timesteps
        # keep track states and start_episode
        self.states = self.envs.reset()
        self.start_episodes = np.ones(self.states.shape[0])  # used to record terminal states

    def collect_data(self, num_timesteps_per_env):
        """ Collect data by interacting with environments, and for each environment, we collect num_timesteps_per_env. 
            Thus, the totoal env timesteps collected are num_envs * num_timesteps_per_env.    
            After collecting data, the returns and advantages will be calculated via GAE.
        """
        eval_info = {}

        for step in range(0, num_timesteps_per_env):
            self.total_timesteps += 1 * self.num_envs

            # collect data from the environment and add them into buffer
            action, action_logprob, action_entropy = self.agent.get_action(torch.FloatTensor(self.states))
            values = self.agent.get_value(torch.FloatTensor(self.states))
            previous_observation = self.states

            # Perform the action on the environment, get new state and reward
            self.states, rewards, dones, infos = self.envs.step(action.detach().numpy())
            self.start_episodes = dones

            # Store action's outcome (so that the agent can improve its policy)
            self.buffer.add(previous_observation, action, rewards, dones, values, action_logprob)

            for info in infos:  # infos is the information dictionary obtained when calling env.step(): ns, r, d, infos = envs.step(a)
                if 'episode' in info.keys():
                    eval_info[self.total_timesteps] = info['episode']['r']
                    break

        with torch.no_grad():
            last_value = self.agent.get_value(torch.FloatTensor(self.states).to(device))

        # Calculate the returns and advantages via GAE, please implement this.
        self.buffer.compute_returns_and_advantages(last_value, self.start_episodes)
        return eval_info

    def train(self, ):
        """ Update the actor and critic network according to eq(7),(9) in https://arxiv.org/pdf/1707.06347.pdf"""
        for epoch in range(self.update_epochs):  # 10
            for data in self.buffer.get(self.mini_batch_size):  # minibatchsize 64*N
                states, actions, logprobs, returns, advantages = data

                # calculate the loss and update the weights of nn
                # Implementation the loss according to eq (7)(9) in https://arxiv.org/pdf/1707.06347.pdf
                # For the value loss, we just use the simple mse loss. 'returns' are the targets.
                # The final loss is the weighted sum of policy loss, value loss and entropy loss. The weigths are 1, self.vf_coeff, self.ent_coeff.

                # 1. complete the missing losses: pg_loss, entropy_loss and v_loss
                _, new_logprobs, entropy = self.agent.get_action(torch.FloatTensor(states),
                                                                 torch.FloatTensor(actions))  # size of (num_envs)
                ratio = new_logprobs - logprobs.detach()
                term1 = torch.mul(torch.exp(ratio), advantages.detach()).unsqueeze(0)  # size of (num_envs)
                term2 = torch.mul(torch.clamp(torch.exp(ratio), min=1 - self.clip_ratio, max=1 + self.clip_ratio),
                                  advantages.detach()).unsqueeze(0)

                pg_loss = torch.sum(torch.min(torch.cat((term1, term2), 0), 0)[0])
                # current actor for new ratio, but the actor used to explore for old ratio and advantage
                v_loss = torch.sum(torch.square(returns - self.agent.get_value(torch.FloatTensor(states))))
                # current critic for new value but the critic before this iteration of updating for returns?

                entropy_loss = torch.sum(entropy)
                loss = -pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                # 2. update weights with self.optimizer
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (logprobs - new_logprobs).mean()

            # early stop if the kl between new_policy and old_policy is larger than target_kl
            if approx_kl > args.target_kl:
                break

                # return stats
        return {'step': global_step,
                'lr': self.optimizer.param_groups[0]['lr'],
                'val_loss': v_loss.item(),
                'pi_loss': pg_loss.item(),
                'entropy': entropy.mean().item(),
                'approx_kl': approx_kl.item(), }

    def update_lr(self, update, num_updates):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]['lr'] = lrnow

    def save(self, filename):
        torch.save(self.agent.state_dict(), filename + '_ppo.pth')

    def load(self, filename):
        self.agent.load_state_dict(torch.load(filename + "_ppo.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', default='PPO')
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--max_timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    # Algorithm specific arguments
    parser.add_argument('--n_minibatch', type=int, default=32,
                        help='the number of mini batch')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='the number of parallel game environment')
    parser.add_argument('--num_timesteps_per_env', type=int, default=2048,
                        help='the number of timesteps per environment to collect during interacting with environments.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip_coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update_epochs', type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument('--target_kl', type=float, default=0.03,
                        help='break the training when KL divergence between the new and the old policy is larger than target_kl.')
    # options
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the experiment')
    parser.add_argument('--anneal_lr', action='store_true',
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    if args.seed == 0:
        args.seed = int(time.time())
    args.batch_size = int(args.num_envs * args.num_timesteps_per_env)  # num_env:1, batch_size:2048
    args.minibatch_size = int(args.batch_size // args.n_minibatch)  # minibatch size: 64 n_minibatch:32
    global_step = 0
    num_updates = args.max_timesteps // args.batch_size  # 976

    experiment_name = f"{args.env}_{args.algo_name}_{args.seed}_{int(time.time())}"
    wandb.init(project='rl_project', config=vars(args), name=experiment_name)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


    def make_env(gym_id, seed, ):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(
                env)  # Done flag will be automatically handled here and we don't need to call reset() every episode.
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.NormalizeReward(env)

            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk


    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env, args.seed + i) for i in range(args.num_envs)])

    buffer_kwargs = {'state_dim': envs.single_observation_space.shape[0],
                     'action_dim': envs.single_action_space.shape[0],
                     'buffer_size': args.num_timesteps_per_env,  # 2048
                     'num_envs': args.num_envs,
                     'gae_lambda': args.gae_lambda,
                     'gamma': args.gamma, }

    buffer = PPOBuffer(**buffer_kwargs)

    ppo_kwargs = {'state_dim': envs.single_observation_space.shape[0],
                  'action_dim': envs.single_action_space.shape[0],
                  'lr': args.lr,
                  'num_timesteps_per_env': args.num_timesteps_per_env,
                  'num_envs': args.num_envs,
                  'buffer': buffer,
                  'envs': envs,
                  'mini_batch_size': args.minibatch_size,
                  'clip_coef': args.clip_coef,
                  'ent_coef': args.ent_coef,
                  'vf_coef': args.vf_coef, }
    ppo = PPO(**ppo_kwargs)

    for update in tqdm.tqdm(range(1, num_updates + 1)):
        print('iteration', update)
        # collect data from env and return the stats
        eval_info = ppo.collect_data(num_timesteps_per_env=args.num_timesteps_per_env)

        for k, v in eval_info.items():
            wandb.log({'eval/': {'timesteps': k, 'returns': v}})

        if args.anneal_lr:
            ppo.update_lr(update, num_updates)

        # train the policy with collected data
        update_info = ppo.train()
        wandb.log({'train/': update_info})

    if args.save_model:
        ppo.save(f"./{experiment_name}")

    envs.close()
