from gym.spaces import Discrete

from rlkit.envs.env_utils import get_dim
import numpy as np
import warnings

from collections import OrderedDict, deque
from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace = True,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # self._prev_actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        self._replace = replace

        self._top = 0
        self._size = 0

    # def add_sample(self, observation, action, prev_action, reward, next_observation,
    #                terminal, **kwargs): # env_info, 
    #     self._observations[self._top] = observation
    #     self._actions[self._top] = action
    #     self._prev_actions[self._top] = prev_action
    #     self._rewards[self._top] = reward
    #     self._terminals[self._top] = terminal
    #     self._next_obs[self._top] = next_observation

    #     # for key in self._env_info_keys:
    #     #     self._env_infos[key][self._top] = env_info[key]
            
    #     self._advance()
    
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs): # env_info, 
        self._observations[self._top] = observation
        self._actions[self._top] = action
        # self._prev_actions[self._top] = prev_action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        # for key in self._env_info_keys:
        #     self._env_infos[key][self._top] = env_info[key]
            
        self._advance()

    def terminate_episode(self):
        pass

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            stack_size=1,
            action_history_len=0,
            env_info_sizes=None,
            train_with_action_history=False
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space  #.shape[0] * stack_size
        self._action_space = env.action_space

        if train_with_action_history:
            obs_dim = get_dim(self._ob_space) * stack_size + get_dim(self._action_space) * max(stack_size - 1, 1)
        else:
            obs_dim = get_dim(self._ob_space) * stack_size

        act_dim = get_dim(self._action_space) * (action_history_len)

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=obs_dim,
            action_dim=act_dim,
            env_info_sizes=env_info_sizes
        )

        self.obs_mean = None
        self.obs_std = None

        self.act_mean = None
        self.act_std = None

    # def add_sample(self, observation, action, prev_action, reward, terminal,
    #                next_observation, **kwargs):
    #     if isinstance(self._action_space, Discrete):
    #         new_action = np.zeros(self._action_dim)
    #         new_action[action] = 1
    #     else:
    #         new_action = action

    #     return super().add_sample(
    #         observation=observation,
    #         action=new_action,
    #         prev_action=prev_action,
    #         reward=reward,
    #         next_observation=next_observation,
    #         terminal=terminal,
    #         # **kwargs
    #     )

    def calculate_statistics(self):
        self.obs_mean = np.mean(self._observations[:self._top], axis=0, keepdims=True)
        self.obs_std = np.std(self._observations[:self._top], axis=0, keepdims=True)

        self.act_mean = np.mean(self._actions[:self._top], axis=0, keepdims=True)
        self.act_std = np.std(self._actions[:self._top], axis=0, keepdims=True)

        return self.obs_mean, self.obs_std, self.act_mean, self.act_std

    def set_statistics(self, obs_mean, obs_std, act_mean, act_std):
        self.obs_mean, self.obs_std, self.act_mean, self.act_std = obs_mean, obs_std, act_mean, act_std
        
    def get_statistics(self):
        return self.obs_mean, self.obs_std, self.act_mean, self.act_std

    def random_batch(self, batch_size, standardize=False):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            obss = (self._observations[indices] - self.obs_mean) / self.obs_std
            # actions = (self._actions[indices] - self.act_mean) / self.act_std
            next_obss = (self._next_obs[indices] - self.obs_mean) / self.obs_std
        else:
            obss = self._observations[indices] 
            # actions = self._actions[indices]           
            next_obss = self._next_obs[indices]

        actions = self._actions[indices]
        
        batch = dict(
            observations=obss,
            actions=actions,
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obss,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch
    
    def get_batch(self, batch_size, standardize=False):
        datasize = min(batch_size, self._top)        
        indices = np.arange(datasize)
        # if not self._replace and self._size < batch_size:
        #     warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            obss = (self._observations[indices] - self.obs_mean) / self.obs_std
            # actions = (self._actions[indices] - self.act_mean) / self.act_std
            next_obss = (self._next_obs[indices] - self.obs_mean) / self.obs_std
        else:
            obss = self._observations[indices] 
            # actions = self._actions[indices]           
            next_obss = self._next_obs[indices]

        actions = self._actions[indices]
        
        batch = dict(
            observations=obss,
            actions=actions,
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obss,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            # **kwargs
        )


class InitObsBuffer():
    def __init__(
            self,
            env,
            init_obs
    ):
        self.env = env
        self.obs_mean = None
        self.obs_std = None
        
        self._init_obss = init_obs
        self._size = len(init_obs)
        self._replace = False
        
    def set_statistics(self, obs_mean, obs_std):
        self.obs_mean, self.obs_std = obs_mean, obs_std
        
    def random_batch(self, batch_size, standardize=False):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            init_obs = (self._init_obss[indices] - self.obs_mean) / self.obs_std
            
        else:
            init_obs = self._init_obss[indices] 
            
        return init_obs
    

class MDPReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            obs_dim=None,
            act_dim=None,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space  #.shape[0] * stack_size
        self._action_space = env.action_space

        # if train_with_action_history:
        #     obs_dim = get_dim(self._ob_space) * stack_size + get_dim(self._action_space) * max(stack_size - 1, 1)
        # else:
        if obs_dim is None:
            obs_dim = get_dim(self._ob_space)
        
        if act_dim is None:
            act_dim = get_dim(self._action_space)

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=obs_dim,
            action_dim=act_dim,
            env_info_sizes=env_info_sizes
        )

        self.obs_mean = None
        self.obs_std = None

        self.act_mean = None
        self.act_std = None

    # def add_sample(self, observation, action, prev_action, reward, terminal,
    #                next_observation, **kwargs):
    #     if isinstance(self._action_space, Discrete):
    #         new_action = np.zeros(self._action_dim)
    #         new_action[action] = 1
    #     else:
    #         new_action = action

    #     return super().add_sample(
    #         observation=observation,
    #         action=new_action,
    #         prev_action=prev_action,
    #         reward=reward,
    #         next_observation=next_observation,
    #         terminal=terminal,
    #         # **kwargs
    #     )

    def calculate_statistics(self):
        self.obs_mean = np.mean(self._observations[:self._top], axis=0, keepdims=True)
        self.obs_std = np.std(self._observations[:self._top], axis=0, keepdims=True)

        self.act_mean = np.mean(self._actions[:self._top], axis=0, keepdims=True)
        self.act_std = np.std(self._actions[:self._top], axis=0, keepdims=True)

        return self.obs_mean, self.obs_std, self.act_mean, self.act_std

    def set_statistics(self, obs_mean, obs_std, act_mean, act_std):
        self.obs_mean, self.obs_std, self.act_mean, self.act_std = obs_mean, obs_std, act_mean, act_std
        
    def get_statistics(self):
        return self.obs_mean, self.obs_std, self.act_mean, self.act_std

    def random_batch(self, batch_size, standardize=False):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            obss = (self._observations[indices] - self.obs_mean) / self.obs_std
            # actions = (self._actions[indices] - self.act_mean) / self.act_std
            next_obss = (self._next_obs[indices] - self.obs_mean) / self.obs_std
        else:
            obss = self._observations[indices] 
            # actions = self._actions[indices]           
            next_obss = self._next_obs[indices]

        actions = self._actions[indices]
        
        batch = dict(
            observations=obss,
            actions=actions,
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obss,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch
    
    def random_batch_weighted_sampling(self, batch_size, sample_weights=None, standardize=False):
        if sample_weights is None:
            indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        else:
            weight_sum = sample_weights.sum()
            sample_weights = sample_weights / weight_sum
            
            indices = np.random.choice(self._size, p=sample_weights, size=batch_size, replace=True)
        
        
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            obss = (self._observations[indices] - self.obs_mean) / self.obs_std
            # actions = (self._actions[indices] - self.act_mean) / self.act_std
            next_obss = (self._next_obs[indices] - self.obs_mean) / self.obs_std
        else:
            obss = self._observations[indices] 
            # actions = self._actions[indices]           
            next_obss = self._next_obs[indices]

        actions = self._actions[indices]
        
        batch = dict(
            observations=obss,
            actions=actions,
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obss,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch
    
    def get_batch(self, batch_size=None, standardize=False):
        if batch_size is not None:
            datasize = min(batch_size, self._top)
        else:
            datasize = self._top
            
        indices = np.arange(datasize)
        # if not self._replace and self._size < batch_size:
        #     warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')

        if standardize and self.obs_mean is not None:
            obss = (self._observations[indices] - self.obs_mean) / self.obs_std
            # actions = (self._actions[indices] - self.act_mean) / self.act_std
            next_obss = (self._next_obs[indices] - self.obs_mean) / self.obs_std
        else:
            obss = self._observations[indices] 
            # actions = self._actions[indices]           
            next_obss = self._next_obs[indices]

        actions = self._actions[indices]
        
        batch = dict(
            observations=obss,
            actions=actions,
            # prev_actions=self._prev_actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=next_obss,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            # **kwargs
        )

