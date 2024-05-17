import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch import autograd

from core.hscic import estimate_hscic, estimate_hsic

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DICEBCv2(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True,
                 nu_hidden_size=256, e_hidden_size=256, alpha=0.01, inner_steps=10, gamma=0.99, 
                 init_obs_buffer=None, init_obs_buffer_valid=None, expert_policy=None,
                 n_train=1, n_valid=1, train_lambda=None, lamb_scale=1.0, use_target_nu=False, weight_norm=True, weighted_replay_sampling=False):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(DICEBCv2, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        
        self.init_obs_buffer = init_obs_buffer
        self.init_obs_buffer_valid = init_obs_buffer_valid
        
        self.device = device
        
        self.n_train = n_train
        self.n_valid = n_valid
    
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize
        self.gamma = gamma
        self.weighted_replay_sampling = weighted_replay_sampling
        
        if train_lambda is None:
            self.train_lambda = True if self.gamma == 1. else False
        else:
            self.train_lambda = train_lambda
        
        self.lamb_scale = lamb_scale
        self.use_target_nu = use_target_nu
        self.weight_norm = weight_norm
        
        self.nu_hidden_size = nu_hidden_size
        self.nu_network = nn.Sequential(
            nn.Linear(self.obs_dim, self.nu_hidden_size, device=self.device),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.nu_hidden_size, 1, device=self.device)
        )
        
        self.e_hidden_size = e_hidden_size
        self.e_network = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.e_hidden_size, device=self.device),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.e_hidden_size, 1, device=self.device)
        )
        
        # if self.train_lamb:
        self.lamb = torch.tensor(0., requires_grad=True)
        
        # soft-chi-square 
        self.f_fn = \
            lambda x: torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1,  0.5 * (x - 1) ** 2)
            # lambda x:  x * (torch.log(x + 1e-10) - 1) + 1 if x  < 1 else 0.5 * (x - 1) ** 2
        self.w_fn =  \
            lambda x: torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))),  x + 1)
            # lambda x: torch.exp(torch.min(x, 0)) if x < 0 else x + 1
        self.f_w_fn = \
            lambda x: torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))) * \
                                         (torch.min(x,  torch.zeros_like(x)) - 1) + 1, 0.5 * x ** 2)
            # lambda x: torch.exp(torch.min(x, 0)) * (torch.min(x, 0) - 1) + 1 if x < 0 else 0.5 * x ** 2
        
        self.nu_optimizer = optim.Adam(self.nu_network.parameters(), lr=lr)
        self.e_optimizer  = optim.Adam(self.e_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.lamb_optimizer = optim.Adam([self.lamb], lr=lr)
        
        self.alpha = alpha
        self.inner_steps = inner_steps
        
        self.num_eval_iteration = 100
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

        self.expert_policy = expert_policy
        
    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        max_score = -100000.
        
        batch_valid = self.replay_buffer_valid.random_batch(self.n_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']
        actions_valid = batch_valid['actions'][:, -self.action_dim:]
        next_obs_valid = batch_valid['next_observations']
        terminals_valid = batch_valid['terminals']
                
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        next_obs_valid = torch.tensor(next_obs_valid, dtype=torch.float32, device=self.device)
        terminals_valid = torch.tensor(terminals_valid, dtype=torch.float32, device=self.device)
        
        for num in range(0, int(total_iteration)+1):
            for _ in range(self.inner_steps):
                
                batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
                obs = batch['observations']
                actions = batch['actions'][:, -self.action_dim:]
                next_obs = batch['next_observations']
                terminals = batch['terminals']
                
                init_obs = self.init_obs_buffer.random_batch(batch_size, standardize=self.standardize)
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                init_obs = torch.tensor(init_obs, dtype=torch.float32, device=self.device)
                terminals = torch.tensor(terminals, dtype=torch.float32, device=self.device).reshape(-1)
                
                batch0 = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
                obs0 = batch0['observations']
                obs0 = torch.tensor(obs0, dtype=torch.float32, device=self.device)
                
                # nu training
                C_pi_s = ((actions - self.policy(obs).mean)**2).mean(dim=-1)
                # nu_s_0 = self.nu_network(init_obs).reshape(-1)
                nu_s_0 = self.nu_network(obs0).reshape(-1)
                nu_s = self.nu_network(obs).reshape(-1)
                nu_s_prime =  self.nu_network(next_obs).reshape(-1)
                e_target = C_pi_s + (1 - terminals) * self.gamma * nu_s_prime - nu_s
            
                w_nu = self.w_fn ( (e_target - self.lamb_scale * self.lamb) / self.alpha)
                f_w_nu = self.f_w_fn( (e_target - self.lamb_scale * self.lamb) / self.alpha)
                
                nu_loss0 = (1 - self.gamma) * nu_s_0.mean()
                nu_loss1 = - self.alpha * f_w_nu.mean()
                nu_loss2 = (w_nu * (e_target - self.lamb_scale * self.lamb)).mean()
                nu_loss3 = self.lamb_scale * (self.lamb + self.lamb ** 2 /2 )
                
                nu_loss = nu_loss0 + nu_loss1 + nu_loss2 + nu_loss3
                
                self.nu_optimizer.zero_grad()
                nu_loss.backward()
                self.nu_optimizer.step()
                
                if self.train_lambda:
                    C_pi_s = ((actions - self.policy(obs).mean)**2).mean(dim=-1)
                    nu_s = self.nu_network(obs).reshape(-1)
                    nu_s_prime = self.nu_network(next_obs).reshape(-1)
                    e_target = C_pi_s + (1 - terminals) * self.gamma * nu_s_prime - nu_s
                    
                    w_nu = self.w_fn ( (e_target - self.lamb_scale * self.lamb) / self.alpha)
                    f_w_nu = self.f_w_fn( (e_target - self.lamb_scale * self.lamb) / self.alpha)
                    
                    lamb_loss1 = -self.alpha * f_w_nu.mean()
                    lamb_loss2 = (w_nu * (e_target - self.lamb_scale * self.lamb )).mean()
                    # gendice-style lamb regularization
                    lamb_loss3 = self.lamb_scale * (self.lamb + self.lamb ** 2 /2 )
                    
                    lamb_loss = lamb_loss1 + lamb_loss2 + lamb_loss3
                    
                    self.lamb_optimizer.zero_grad()
                    lamb_loss.backward()
                    self.lamb_optimizer.step()
                    
                else:
                    lamb_loss = 0.
                    
                # e training
                # s_a = torch.cat([obs, actions], dim=-1)
                # e_pred = self.e_network(s_a)
                # w_e = self.w_fn (e_pred / self.alpha)
                # f_w_e = self.f_w_fn(e_pred / self.alpha)
                
                # e_loss = ((e_pred - e_target.detach())**2).mean() 
                # self.e_optimizer.zero_grad()
                # e_loss.backward()
                # self.e_optimizer.step()
            
            if self.weighted_replay_sampling:
                full_batch = self.replay_buffer.get_batch(standardize=self.standardize)
                obs = full_batch['observations']
                actions = full_batch['actions']
                next_obs = full_batch['next_observations']
                terminals = full_batch['terminals']
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                terminals = torch.tensor(terminals, dtype=torch.float32, device=self.device).reshape(-1)
                
                C_pi_s = ((actions - self.policy(obs).mean)**2).mean(dim=-1)
                nu_s = self.nu_network(obs).reshape(-1)
                nu_s_prime =  self.nu_network(next_obs).reshape(-1)

                e_target = C_pi_s + (1 - terminals) * self.gamma * nu_s_prime - nu_s
                w_e = self.w_fn ((e_target - self.lamb) / self.alpha)
                full_weights = w_e.cpu().detach().numpy()
                
                batch = self.replay_buffer.random_batch_weighted_sampling(batch_size, sample_weights=full_weights, standardize=self.standardize)
                
                obs = batch['observations']
                actions = batch['actions']
                next_obs = batch['next_observations']
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                
                C_pi_s = ((actions - self.policy(obs).mean)**2).mean(dim=-1)
                policy_loss = C_pi_s.mean()
                
            else:
                batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
                
                obs = batch['observations']
                actions = batch['actions']
                next_obs = batch['next_observations']
                
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                
                # s_a = torch.cat([obs, actions], dim=-1)
                # e_pred = self.e_network(s_a)
                
                C_pi_s = ((actions - self.policy(obs).mean)**2).mean(dim=-1)
                nu_s = self.nu_network(obs).reshape(-1)
                nu_s_prime =  self.nu_network(next_obs).reshape(-1)

                e_target = C_pi_s + (1 - terminals) * self.gamma * nu_s_prime - nu_s
                w_e = self.w_fn ((e_target - self.lamb) / self.alpha)
                
                if self.weight_norm:
                    normalized_w_e = w_e / w_e.mean()
                    policy_loss = (normalized_w_e.detach() * C_pi_s).mean()
                else:
                    policy_loss = (w_e.detach() * C_pi_s).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if (num) % eval_freq == 0:
                # s_a_valid = torch.cat([obs_valid, actions_valid], dim=-1)
                # e_pred_valid = self.e_network(s_a_valid)
                # w_e_valid = self.w_fn (e_pred_valid / self.alpha)
                C_pi_s_valid = ((actions_valid - self.policy(obs_valid).mean)**2).mean(dim=-1)
                nu_s_valid = self.nu_network(obs_valid).reshape(-1)
                nu_s_prime_valid =  self.nu_network(next_obs_valid).reshape(-1)

                e_target_valid = C_pi_s_valid + (1 - terminals_valid) * self.gamma * nu_s_prime_valid - nu_s_valid
                w_e_valid = self.w_fn ((e_target_valid - self.lamb_scale * self.lamb)/ self.alpha)
                
                C_pi_s_valid = ((actions_valid - self.policy(obs_valid).mean)**2).mean(dim=-1)
                policy_loss_valid = (w_e_valid.detach() * C_pi_s_valid).mean()
                
                mse_loss_train = C_pi_s.mean()
                mse_loss_valid = C_pi_s_valid.mean()

                eval_ret_mean, eval_ret_std, true_mse, CVAR_1percent, CVAR_5percent, CVAR_25percent = self.evaluate(num_iteration=self.num_eval_iteration)

                nu_valid_s = self.nu_network(obs_valid).reshape(-1).detach().mean()
                
                w_train_avg = w_e.mean()
                ess_train = (w_e.sum()) ** 2 / (w_e **2 ).sum()
                
                w_valid_avg = w_e_valid.mean()
                ess_valid = (w_e_valid.sum()) ** 2 / (w_e_valid **2 ).sum()
                
                print(f'** iter{num}: train_policy_loss={policy_loss.item():.6f}, val_policy_loss={policy_loss_valid.item():.6f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'-- MSE : (train) {mse_loss_train:.6f} (valid) {mse_loss_valid:.6f} (true) {true_mse:.6f}')
                print(f'-- CVAR_1percent={CVAR_1percent:.6f}, CVAR_5percent={CVAR_5percent:.6f}, CVAR_25percent={CVAR_25percent:.6f}')
                
                if self.wandb:
                    log_dict = {            
                            'nu/loss0':             nu_loss0.item(),
                            'nu/loss1':             nu_loss1.item(),
                            'nu/loss2':             nu_loss2.item(),
                            'nu/loss3':             nu_loss3.item(),
                            'nu/nu_valid_s':        nu_valid_s.item(),
                            'nu/w_train_avg':       w_train_avg.item(),
                            'nu/ESS_train':         ess_train.item(),
                            'nu/w_valid_avg':       w_valid_avg.item(),
                            'nu/ESS_valid':         ess_valid.item(),
                            'nu_loss':              nu_loss.item(), 
                            'lambda/value':         self.lamb.item(),
                            'lambda/loss':          lamb_loss,
                            'train_policy_loss':    policy_loss.item(),
                            'valid_policy_loss':    policy_loss_valid.item(),
                            'train_mse':            mse_loss_train.item(),
                            'valid_mse':            mse_loss_valid.item(),
                            'target_mse':           true_mse,
                            'eval/episode_return':  eval_ret_mean,
                            'eval/CVAR1':           CVAR_1percent,
                            'eval/CVAR5':           CVAR_5percent,
                            'eval/CVAR25':          CVAR_25percent,
                    }
                    
                    if self.train_lambda:
                        log_dict['lambda/loss1'] = lamb_loss1
                        log_dict['lambda/loss2'] = lamb_loss2
                        log_dict['lambda/loss3'] = lamb_loss3
                        
                    self.wandb.log(log_dict, step=num+1)    

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
        true_se_list = []

        maxtimestep = 1000
        for num in range(0, num_iteration):
            obs_ = self.env.reset()
            
            done = False
            t = 0
            ret = 0.
            
            while not done and t < maxtimestep:
                # Before standardization
                expert_input = np.expand_dims(obs_, axis=0).astype(np.float32)                
                noise_input = np.zeros([1, self.env.action_space.shape[0]]).astype(np.float32)
                expert_action, _, _ = self.expert_policy.run(None, {'observations': expert_input, 'noise': noise_input})

                # obs = np.concatenate([obs_, [0.]])
                obs = obs_
                
                if self.standardize:
                    obs = (obs - self.obs_mean[0]) / (self.obs_std[0])
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action = self.policy(obs).mean.cpu().detach().numpy()

                true_se = np.mean((expert_action - action) ** 2)
                true_se_list.append(true_se)
                
                next_obs, rew, done, _ = self.env.step(action)
                ret += rew
                
                obs_ = next_obs 
                    
                t += 1
            
            rets.append(ret)
        
        num_CVAR_1percent = int(num_iteration * 0.01)
        CVAR_1percent = np.sort(rets)[:num_CVAR_1percent].mean()
        
        num_CVAR_5percent = int(num_iteration * 0.05)
        CVAR_5percent = np.sort(rets)[:num_CVAR_5percent].mean()

        num_CVAR_25percent = int(num_iteration * 0.25)
        CVAR_25percent = np.sort(rets)[:num_CVAR_25percent].mean()

        return np.mean(rets), np.std(rets), np.mean(true_se_list), CVAR_1percent, CVAR_5percent, CVAR_25percent
    
    