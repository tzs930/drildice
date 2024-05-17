import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from core.hscic import estimate_hscic, estimate_hsic

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class HSICBC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True, reg_coef=1.):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(HSICBC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device
    
        self.obs_dim = obs_dim
        self.action_dim = action_dim        
        self.stacksize = stacksize
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.reg_coef = reg_coef
        
        self.num_eval_iteration = 50
        self.envname = envname
        
        self.wandb = None
        if wandb:
            self.wandb = wandb
            self.wandb.init()

        self.save_policy_path = save_policy_path        
        
        # For standardization
        self.standardize = standardize

        self.obs_mean_tt = torch.tensor(self.replay_buffer.obs_mean, device=device)
        self.obs_std_tt = torch.tensor(self.replay_buffer.obs_std, device=device)
        self.act_mean_tt = torch.tensor(self.replay_buffer.act_mean, device=device)
        self.act_std_tt = torch.tensor(self.replay_buffer.act_std, device=device)

        self.obs_mean = self.replay_buffer.obs_mean
        self.obs_std = self.replay_buffer.obs_std
        self.act_mean = self.replay_buffer.act_mean
        self.act_std = self.replay_buffer.act_std
        
    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024, num_valid=2000):
        max_score = -100000.
        
        batch_valid = self.replay_buffer_valid.random_batch(num_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']
        actions_valid = batch_valid['actions'][:, -self.action_dim:]        
        prev_expert_action_valid = batch_valid['actions'][:, :-self.action_dim] # For debugging
                
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        prev_expert_action_valid = torch.tensor(prev_expert_action_valid, dtype=torch.float32, device=self.device)
        
        for num in range(0, int(total_iteration)+1):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
            
            obs = batch['observations']
            actions = batch['actions'][:, -self.action_dim:]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)            

            # neg_likelihood = -self.policy.log_prob(obs, actions).mean()
            train_epsilon = self.policy(obs).mean - actions
            train_mse = (train_epsilon ** 2).mean()
            train_hsic = estimate_hsic(X=obs, Y= train_epsilon, Kx_sigma2=1., Ky_sigma2=1.)
            
            train_loss = train_mse + self.reg_coef * train_hsic
            
            self.policy_optimizer.zero_grad()
            train_loss.backward()
            self.policy_optimizer.step()

            if (num) % eval_freq == 0:
                valid_epsilon = self.policy(obs_valid).mean - actions_valid
                valid_mse = (valid_epsilon ** 2).mean()
                valid_hsic = estimate_hsic(X=obs_valid, Y= valid_epsilon, Kx_sigma2=1., Ky_sigma2=1.)
                
                valid_loss = valid_mse + self.reg_coef * valid_hsic

                eval_ret_mean, eval_ret_std = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num}: train_policy_loss={train_loss.item():.2f}, val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'** MSE :  (train) {train_mse:.6f} (valid) {valid_mse:.6f}')
                print(f'** HSIC : (train) {train_hsic:.6f} (valid) {valid_hsic:.6f}')
                
                if self.wandb:
                    self.wandb.log({'train_total_loss':       train_loss.item(), 
                                    'valid_total_loss':       valid_loss.item(),
                                    'train_MSE':              train_mse.item(),
                                    'valid_MSE':              valid_mse.item(),
                                    'train_HSIC':             train_hsic.item(),
                                    'valid_HSIC':             valid_hsic.item(),
                                    'eval_episode_return':    eval_ret_mean
                                    }, step=num+1)

                if eval_ret_mean > max_score:
                    print(f'** max score record! ')
                    max_score = eval_ret_mean
                    copy_nn_module(self.policy, self.best_policy)
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
                    
        if self.save_policy_path:
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_best.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.best_policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_best.pt')
            
            print(f'** save model to ', f'{self.save_policy_path}/bc_actor_last.pt')
            os.makedirs(self.save_policy_path, exist_ok=True)
            torch.save(self.policy.state_dict(), 
                    f'{self.save_policy_path}/bc_actor_last.pt')
    
                   
    def evaluate(self, num_iteration=5):
        rets = []
        maxtimestep = 1000
        for num in range(0, num_iteration):
            obs_list = []
            obs = np.zeros(self.obs_dim * self.stacksize)
            
            obs_ = self.env.reset()
            obs_list.append(obs_)

            obs = np.zeros(self.obs_dim * self.stacksize)
            obs[- self.obs_dim:] = obs_

            done = False
            t = 0
            ret = 0.
            
            while not done and t < maxtimestep:
                if self.standardize:
                    obs = (obs - self.obs_mean[0]) / self.obs_std[0]
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()
                
                next_obs, rew, done, _ = self.env.step(action)
                ret += rew
                
                obs_ = next_obs 
                obs_list.append(obs_)

                if len(obs_list) < self.stacksize:
                    obs_ = np.concatenate(obs_list)
                    obs = np.zeros(self.obs_dim * self.stacksize)
                    obs[-(len(obs_list)) * self.obs_dim:] = obs_
                
                else:
                    obs = np.concatenate(obs_list[-self.stacksize:])
                    
                t += 1
            
            rets.append(ret)
        
        return np.mean(rets), np.std(rets)
    
