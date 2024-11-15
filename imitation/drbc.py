import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from scipy.optimize import minimize

def copy_nn_module(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DRBC(nn.Module):
    def __init__(self, policy, env, best_policy=None,
                 replay_buffer=None, replay_buffer_valid=None, seed=0, 
                 device='cpu', lr=3e-4, envname=None, wandb=None, save_policy_path=None, 
                 obs_dim=1, action_dim=1, stacksize=1, standardize=True, expert_policy=None,
                 n_train=1, n_valid=1, add_absorbing_state=False, rho=1.):
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        super(DRBC, self).__init__()

        self.env = env
        self.policy = policy
        self.best_policy = best_policy
        self.replay_buffer = replay_buffer
        self.replay_buffer_valid = replay_buffer_valid
        self.device = device
        self.add_absorbing_state = add_absorbing_state
        self.rho = rho
        
        self.n_train = n_train
        self.n_valid = n_valid
    
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stacksize = stacksize

        # self.eta = torch.tensor(0.1, requires_grad=True, device=device)
        # self.eta_optimizer = optim.Adam([self.eta], lr=lr)
        min_action = env.action_space.low
        max_action = env.action_space.high
        
        self.min_dual = -1.0
        self.max_dual = (1 + rho) * (np.linalg.norm(max_action - min_action) ** 2)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

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
        self.zero = torch.tensor(0, device=device)
        
    def optimize_dual(self, pi_theta, pi_e):
        # reused from the original DR-BC implementation: 
        # https://github.com/ferocious-cheetah/DRBC/blob/e15c49cfcde0ed33956bc9be1f59a06cec5d6d0a/drobc.py#L181
        sup = np.max(np.linalg.norm(pi_theta - pi_e, axis=1)**2)  # Calculate norm across columns
        def dual_func(eta):
            '''Total variation uncertainty dual function.'''
            res = np.mean(np.maximum(np.linalg.norm(pi_theta-pi_e, axis=1)**2 - eta, 0))
            res += self.rho * np.maximum(sup-eta, 0)
            res += eta
            return res
        bounds = [(self.min_dual, self.max_dual)]
        res = minimize(dual_func, x0=0, method='SLSQP', bounds=bounds, tol=1e-3)
        if not res.success:
            raise ValueError("Dual optimization failed!")
        return res.fun, sup
        
    def train(self, total_iteration=1e6, eval_freq=1000, batch_size=1024):
        max_score = -100000.
        
        batch_valid = self.replay_buffer_valid.random_batch(self.n_valid, standardize=self.standardize)
        
        obs_valid = batch_valid['observations']
        actions_valid = batch_valid['actions'][:, -self.action_dim:]       
                
        obs_valid = torch.tensor(obs_valid, dtype=torch.float32, device=self.device)
        actions_valid = torch.tensor(actions_valid, dtype=torch.float32, device=self.device)
        
        for num in range(0, int(total_iteration)+1):
            batch = self.replay_buffer.random_batch(batch_size, standardize=self.standardize)
            
            obs = batch['observations']
            actions = batch['actions'] #[:, -self.action_dim:]
            
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)            

            # DRBC Objective
            # min_eta  term1 + term2 + eta
            # term1 = sum_{s,a} max( loss(a, policy(s) - eta), 0 )
            # term2 = rho * max( max_{s,a} loss(a, policy(s)) - eta, 0)
            
            pi_theta = self.policy(obs).mean.cpu().detach().numpy()
            pi_e = actions.cpu().detach().numpy()
            dual, sup = self.optimize_dual(pi_theta, pi_e)
            if sup is not None:
                sup = torch.tensor(sup, device=self.device)
            dual = torch.tensor(dual, device=self.device)
            
            policy_loss1 = torch.square(torch.linalg.norm(self.policy(obs).mean - actions, axis=1)) - dual
            policy_loss1 = torch.mean(torch.maximum(policy_loss1, self.zero))
            policy_loss2 = self.rho * torch.maximum(sup - dual, self.zero)
            policy_loss3 = dual
            policy_loss = policy_loss1 + policy_loss2 + policy_loss3
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if (num) % eval_freq == 0:
                valid_epsilon = ((self.policy(obs_valid).mean - actions_valid)**2).mean(dim=-1)     # (batch_size, action_dim)

                pi_theta_valid = self.policy(obs_valid).mean.cpu().detach().numpy()
                pi_e_valid = actions_valid.cpu().detach().numpy()
                dual, sup = self.optimize_dual(pi_theta_valid, pi_e_valid)
                if sup is not None:
                    sup = torch.tensor(sup, device=self.device)
                dual = torch.tensor(dual, device=self.device)
                valid_loss1 = torch.square(torch.linalg.norm(self.policy(obs_valid).mean - actions_valid, axis=1)) - dual
                valid_loss1 = torch.mean(torch.maximum(valid_loss1, self.zero))
                valid_loss2 = self.rho * torch.maximum(sup - dual, self.zero)
                valid_loss3 = dual
                valid_loss = valid_loss1 + valid_loss2 + valid_loss3
                
                valid_mse = valid_epsilon.mean()

                eval_ret_mean, eval_ret_std, true_mse, CVAR_1percent, CVAR_5percent, CVAR_25percent = self.evaluate(num_iteration=self.num_eval_iteration)
                
                print(f'** iter{num}: train_policy_loss={policy_loss.item():.2f}, val_policy_loss={valid_loss.item():.2f}, eval_ret={eval_ret_mean:.2f}+-{eval_ret_std:.2f} ({obs_valid.shape[0]})',)
                print(f'-- MSE :  (valid) {valid_mse:.6f} (true) {true_mse:.6f}')
                print(f'-- CVAR_1percent={CVAR_1percent:.6f}, CVAR_5percent={CVAR_5percent:.6f}, CVAR_25percent={CVAR_25percent:.6f}')
                
                if self.wandb:
                    log_dict = {
                                'train_policy_loss':       policy_loss.item(), 
                                'valid_policy_loss':       valid_loss.item(), 
                                'train_loss_term1':        policy_loss1.item(),
                                'train_loss_term2':        policy_loss2.item(),
                                'valid_loss_term1':        valid_loss1.item(),
                                'valid_loss_term2':        valid_loss2.item(),
                                'valid_mse':               valid_mse.item(),
                                'eta':                     dual,
                                'target_mse':              true_mse,
                                'eval/episode_return':     eval_ret_mean,
                                'eval/CVAR1':              CVAR_1percent,
                                'eval/CVAR5':              CVAR_5percent,
                                'eval/CVAR25':             CVAR_25percent,
                                }
                    
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

                if self.add_absorbing_state:
                    obs = np.concatenate([obs_, [0.]]) # absorbing
                else:
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
    
