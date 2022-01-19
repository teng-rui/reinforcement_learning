import argparse
import pickle

import tqdm
import copy
import time
import random

import pybullet_envs
import gym
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Process

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Helper Functions ##########
def eval_policy(policy, eval_env, eval_episodes=10):
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment

    avg_reward = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            # NEW states are normalized
            state = (np.array(state).reshape(1, -1) - mean) / std

            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    return {'returns': avg_reward}



########## Define Replay Buffer ##########
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, not_done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
        )

    def load_dataset(self, dataset, ratio):
        size = int(dataset['states'].shape[0] * ratio)

        # resize
        self.state = np.zeros((size, state_dim))
        self.action = np.zeros((size, action_dim))
        self.next_state = np.zeros((size, state_dim))
        self.reward = np.zeros((size, 1))
        self.not_done = np.zeros((size, 1))


        for index in range(size):
            self.add(dataset['states'][index],
                dataset['actions'][index],
                dataset['next_states'][index],
                dataset['rewards'][index],
                dataset['not_dones'][index])

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

########## Define Agent ##########
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

########## TD3-BC ##########
class TD3_BC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            # NEW
            alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        # NEW
        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # NEW: batch
    def train(self, batch):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            t_q1, t_q2 = self.critic_target(next_state, next_action)
            t_q = reward + not_done * self.discount * torch.min(t_q1, t_q2)

        # Get current Q estimates
        c_q1, c_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(c_q1, t_q) + F.mse_loss(c_q2, t_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            # NEW  a weighted behavior cloning loss is added to the policy update
            # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # get action_dim x 1
            pi = self.actor(state)
            Q = self.critic.forward(state, pi)[0]
            lambada = self.alpha / Q.abs().mean().detach()
            actor_loss = -lambada * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models, both actor and critic
            for p, t_p in zip(self.critic.parameters(), self.critic_target.parameters()):
                t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

            for p, t_p in zip(self.actor.parameters(), self.actor_target.parameters()):
                t_p.data.copy_(self.tau * p.data + (1 - self.tau) * t_p.data)

        return {"critic_loss": critic_loss.item(),
                "critic": c_q1.mean().item()}

    def save(self, filename):
        torch.save({'critic': self.critic.state_dict(),
                    'actor': self.actor.state_dict(), }, filename + "_td3.pth")

    def load(self, filename):
        policy_dicts = torch.load(filename + "_td3.pth")

        self.critic.load_state_dict(policy_dicts['critic'])
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(policy_dicts['actor'])
        self.target_actor = copy.deepcopy(self.actor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', default='TD3_BC')
    parser.add_argument('--env', default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--max_timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    # Algorithm specific arguments
    parser.add_argument("--expl_noise", default=0.1, type=float,
                        help="Std of Gaussian exploration noise")
    parser.add_argument("--discount", default=0.99, type=float,
                        help="Discount factor.")
    parser.add_argument("--tau", default=0.005, type=float,
                        help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float,
                        help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int,
                        help="Frequency of delayed policy updates")

    # NEW TD3-BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--ratio", type=float ,default=1)

    # options
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the experiment')
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    if args.seed == 0:
        args.seed = int(time.time())

    experiment_name = f"{args.env}_{args.algo_name}_{args.seed}_{int(time.time())}"

    # print("be3b707ad05b424084782b43fa36f150a1d847f9")
    wandb.init(project="rl_project", config=vars(args), name=experiment_name)

    # Init env and seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(args.env)
    eval_env.seed(args.seed + 100)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # prefill random initialization data
    # replay_buffer = fill_initial_buffer(env, replay_buffer, args.n_random_timesteps)
    # NEW load dataset given to replay_buffer with given ratio
    dataset_path = 'halfcheetah_mixed.pickle'
    with open(dataset_path, 'rb') as dataset:
        dataset_obj = pickle.load(dataset)

    replay_buffer.load_dataset(dataset_obj, args.ratio)

    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    # init td3_bc
    td3_bc_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq,

        # NEW TD3-BC
        "alpha": args.alpha
    }
    td3_bc = TD3_BC(**td3_bc_kwargs)

    state, done = env.reset(), False
    episode_timesteps = 0
    for t in tqdm.tqdm(range(args.max_timesteps)):

        #NEW offline dataset
        # update policy per data point
        policy_update_info = td3_bc.train(replay_buffer.sample(args.batch_size))
        wandb.log({"train/": policy_update_info})

        # Evaluate episode
        if t % args.eval_freq == 0:
            eval_info = eval_policy(td3_bc, eval_env)
            eval_info.update({'timesteps': t})
            print(f"Time steps: {t}, Eval_info: {eval_info}")
            wandb.log({"eval/": eval_info})

    if args.save_model:
        td3_bc.save(f"./{experiment_name}")

    env.close()