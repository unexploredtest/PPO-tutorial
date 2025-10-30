import numpy as np
import gymnasium as gym
import torch
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torch.distributions import Categorical

from network import Mlp


class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        if(isinstance(self.env.action_space, gym.spaces.Discrete)):
            self.act_dim = self.env.action_space.n
        else:
            self.act_dim = self.env.action_space.shape[0]

        self.actor = Mlp(self.obs_dim, self.act_dim)
        self.critic = Mlp(self.obs_dim, 1)

        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gama = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        self.num_minibatchs = 100

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        current_step = 0

        while(current_step < total_timesteps):
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self._rollout()

            V, _ = self._evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.shape[0]
            indices = np.arange(step)
            minibatch_size = step // self.num_minibatchs
            for _ in range(self.n_updates_per_iteration):
                # Learning Rate Annealing
                frac = (current_step - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr 

                np.random.shuffle(indices)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    index = indices[start:end]
                    mini_obs = batch_obs[index]
                    mini_acts = batch_acts[index]
                    mini_log_probs = batch_log_probs[index]
                    mini_advantage = A_k[index]
                    mini_rtgs = batch_rtgs[index]

                    V, curr_log_probs = self._evaluate(mini_obs, mini_acts)

                    ratios = torch.exp(curr_log_probs - mini_log_probs)

                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    actor_loss = (-torch.min(surr1, surr2)).mean()

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    critic_loss = torch.nn.MSELoss()(V, mini_rtgs)

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            current_step += np.sum(batch_lens)

    def predict(self, obs):
        with torch.no_grad():
            if(isinstance(self.env.action_space, gym.spaces.Discrete)):
                action_probs = self.actor(obs)
                return torch.argmax(action_probs).item()
            else:
                action_mean = self.actor(obs)
                return action_mean.numpy()

    def _rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []
        batch_lens = []

        t = 0

        while t < self.timesteps_per_batch:

            ep_rewards = []

            obs, info = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self._get_action(obs)
                if(isinstance(self.env.action_space, gym.spaces.Discrete)):
                    obs, reward, done, truncated, info = self.env.step(action.item())
                else:
                    obs, reward, done, truncated, info = self.env.step(action)

                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)

        batch_rtgs = self._compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def _get_action(self, obs):
        # https://stackoverflow.com/questions/60531143/how-to-adapt-ppo-from-continuous-to-discrete-action-spaces
        if(isinstance(self.env.action_space, gym.spaces.Discrete)):
            values = self.actor(obs)
            softmax_probs = F.softmax(values, dim=0)
            m = Categorical(softmax_probs)
            action = m.sample()

            return action, softmax_probs[action].detach()
        else:
            mean = self.actor(obs)

            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.detach().numpy(), log_prob.detach()

    def _compute_rtgs(self, batch_rewards):
        batch_rtgs = []

        for ep_rewards in reversed(batch_rewards):

            discounted_reward = 0

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gama
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs

    def _evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        if(isinstance(self.env.action_space, gym.spaces.Discrete)):
            values = self.actor(batch_obs)
            dist = F.softmax(values, dim=1)
            actions = batch_acts.to(torch.int32)
            softmax_probs = dist[torch.arange(dist.size(0)), actions]

            return V, softmax_probs
        else:
            mean = self.actor(batch_obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)

            return V, log_probs


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = PPO(env)
    model.learn(200_000)

    env2 = gym.make('CartPole-v1', render_mode="human")

    obs, _ = env2.reset()
    done = False

    while not done:
        action = model.predict(obs)
        obs, reward, done, truncated, info = env2.step(action)

