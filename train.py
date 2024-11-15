import os
wandb_dir = './wandb_offline'
os.environ['WANDB_DIR'] = wandb_dir
os.environ['D4RL_DATASET_DIR'] = './dataset'
import wandb
import envs
import d4rl
import gym
import pickle

import numpy as np
import torch
import time

from imitation.bc import BC
from imitation.drildice import DRILDICE
from imitation.optidiceil import OptiDICEIL
from imitation.aw import AdvWBC
from imitation.demodice import DemoDICE

from argparse import ArgumentParser
from itertools import product

from core.policy import TanhGaussianPolicy
from core.replay_buffer import InitObsBuffer, MDPReplayBuffer
from core.preprocess import preprocess_dataset_with_subsampling
from rlkit.envs.wrappers import NormalizedBoxEnv

import onnx
import onnxruntime as ort

STD_EPSILON = 1e-8

def train(configs):
    env = NormalizedBoxEnv(gym.make(configs['envname']))
    # obs_dim    = env.observation_space.low.size + 1 ## absorbing
    obs_dim    = env.observation_space.low.size 
    action_dim = env.action_space.low.size
    
    d4rl_env = gym.make(configs['d4rl_env_name'])
    
    stacksize = configs['stacksize']
    if stacksize == 0:
        stacksize = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    envname, envtype = configs['envname'], configs['envtype']
    
    traj_load_path = configs['traj_load_path']
    print(f'-- Loading dataset from {traj_load_path}...')
    dataset = d4rl_env.get_dataset()
    print(f'-- Done!')
    
    print(f'-- Preprocessing dataset... ({envtype}, {stacksize})')
    train_data, n_train = preprocess_dataset_with_subsampling(dataset, configs['idxfile'], start_traj_idx=0, 
                                                     num_trajs=configs['train_num_trajs'],
                                                     add_absorbing_state=False)
    valid_data, n_valid = preprocess_dataset_with_subsampling(dataset, configs['idxfile'], start_traj_idx=900, 
                                                     num_trajs=configs['valid_num_trajs'],
                                                     add_absorbing_state=False)
    
    
    print(f'** num. of train data : {n_train},   num. of valid data : {n_valid}')
    
    train_init_obss = train_data['init_observations']
    valid_init_obss = valid_data['init_observations']
    
    replay_buffer = MDPReplayBuffer(
        configs['replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    replay_buffer.add_path(train_data)

    replay_buffer_valid = MDPReplayBuffer(
        configs['replay_buffer_size'],
        env,
        obs_dim=obs_dim
    )
    replay_buffer_valid.add_path(valid_data)
    
    init_obs_buffer = InitObsBuffer(
        env, train_init_obss
    )
    init_obs_buffer_valid = InitObsBuffer(
        env, valid_init_obss
    )
    
    if configs['standardize']:
        obs_mean, obs_std, act_mean, act_std = replay_buffer.calculate_statistics()
        obs_std += STD_EPSILON
        act_std += STD_EPSILON
        
        replay_buffer_valid.set_statistics(obs_mean, obs_std, act_mean, act_std)
        
        init_obs_buffer.set_statistics(obs_mean, obs_std)
        init_obs_buffer_valid.set_statistics(obs_mean, obs_std)
        
    # to use wandb, initialize here, e.g.
    # wandb = None
        
    if 'DRILDICE' in configs['method']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,
            device=device
        )
        
        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device
        )
       
        trainer = DRILDICE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            init_obs_buffer = init_obs_buffer,
            init_obs_buffer_valid = init_obs_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            wandb = wandb,
            alpha = configs['reg_coef'],
            standardize = configs['standardize'],
            gamma = configs['gamma'],
            inner_steps = configs['inner_steps'],
            expert_policy = configs['expert_policy'],
            n_train=n_train,
            n_valid=n_valid,
            weight_norm=configs['weight_norm'],
            train_lambda=configs['train_lambda'],
            weighted_replay_sampling =configs['weighted_replay_sampling'] 
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'])
    
    elif 'DemoDICE' in configs['method']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,
            device=device
        )
        
        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device
        )
       
        trainer = DemoDICE(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            init_obs_buffer = init_obs_buffer,
            init_obs_buffer_valid = init_obs_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            wandb = wandb,
            alpha = configs['reg_coef'],
            standardize = configs['standardize'],
            gamma = configs['gamma'],
            inner_steps = configs['inner_steps'],
            expert_policy = configs['expert_policy'],
            n_train=n_train,
            n_valid=n_valid,
            weight_norm=configs['weight_norm'],
            train_lambda=configs['train_lambda'],
            weighted_replay_sampling =configs['weighted_replay_sampling'] 
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'])
    
    elif 'ADVWBC' in configs['method']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,
            device=device
        )
        
        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device
        )
       
        trainer = AdvWBC(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            init_obs_buffer = init_obs_buffer,
            init_obs_buffer_valid = init_obs_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            wandb = wandb,
            standardize = configs['standardize'],
            gamma = configs['gamma'],
            inner_steps = configs['inner_steps'],
            expert_policy = configs['expert_policy'],
            n_train=n_train,
            n_valid=n_valid,
            weight_norm=configs['weight_norm'],
            train_lambda=configs['train_lambda'],
            weighted_replay_sampling =configs['weighted_replay_sampling'] 
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'])
        
    elif 'OPTIDICEIL' in configs['method']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,
            device=device
        )
        
        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device
        )
       
        trainer = OptiDICEIL(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            init_obs_buffer = init_obs_buffer,
            init_obs_buffer_valid = init_obs_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,
            stacksize = stacksize,
            wandb = wandb,
            alpha = configs['reg_coef'],
            standardize = configs['standardize'],
            gamma = configs['gamma'],
            inner_steps = configs['inner_steps'],
            expert_policy = configs['expert_policy'],
            n_train=n_train,
            n_valid=n_valid,
            weight_norm=configs['weight_norm'],
            train_lambda=configs['train_lambda'],
            weighted_replay_sampling =configs['weighted_replay_sampling'] 
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'])
    
    elif 'BC' in configs['method']:
        policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device            
        )
        
        best_policy = TanhGaussianPolicy(
            hidden_sizes=configs['layer_sizes'],
            obs_dim=obs_dim ,
            action_dim=action_dim,            
            device=device            
        )
        
        trainer = BC(
            policy = policy,
            best_policy = best_policy,
            env = env,
            replay_buffer = replay_buffer,
            replay_buffer_valid = replay_buffer_valid,
            seed = configs['seed'],
            device = device,
            envname = envname,
            lr = configs['lr'],
            save_policy_path = configs['save_policy_path'],
            obs_dim = obs_dim,
            action_dim = action_dim,            
            stacksize = stacksize,
            wandb = wandb,            
            standardize=configs['standardize'],
            expert_policy=configs['expert_policy'],
            n_train=n_train,
            n_valid=n_valid
        )

        trainer.train(total_iteration = configs['total_iteration'],
                      eval_freq = configs['eval_freq'],
                      batch_size = configs['batch_size'])

    else: 
        raise NotImplementedError       

 
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pid", help="process_id", default=0, type=int)
    args = parser.parse_args()
    pid = args.pid

    time.sleep(pid%60 * 10)
    # Hyperparameter Grid
    # candidates: 'BC', 'DemoDICE', 'ADVWBC', 'OPTIDICEIL', 'DRILDICE'
    method_reg_gamma_list = [
                        ('BC',                0.,     1.0),
                    ]

    stacksizelist     = [0]
    seedlist          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size_list   = [512]
    env_num_trajs_list = [
                            ('Hopper',      [10,20,30,40,50]),
                            ('Walker2d',    [10,20,30,40,50]),
                            ('HalfCheetah', [50,100,150,200,250]),
                         ]
    num_trajs_idx_list = [0,1,2,3,4]
    train_lambda_list = [False]
    inner_steps_list  = [1]
    subsample_dist_list = ['beta']
    # Scenario 1 : 'action-dependent', 'state-depedent'
    # Scenario 2 : 'beta'
    # Scenario 3 : 'geometric-frags+full-trajs1'
    
    standardize = True    
    
    env_num_trajs, method_reg_gamma, inner_steps, seed, batch_size, num_trajs_idx, train_lambda, subsample_dist = \
        list(product(env_num_trajs_list, method_reg_gamma_list, inner_steps_list, seedlist, batch_size_list, num_trajs_idx_list, train_lambda_list, subsample_dist_list))[pid]    
    
    subsample_num = 50
    envtype, num_trajs_list = env_num_trajs
    num_trajs = num_trajs_list[num_trajs_idx]
    
    method, reg_coef, gamma = method_reg_gamma
    stacksize = 0
    
    ib_coef = 0.
    algorithm = f'{method}'

    if stacksize == 0 :        # MDP
        partially_observable = False
        envname = f'{envtype}-v2'        
    else:                      # POMDP
        partially_observable = True
        envname = f'PO{envtype}-v0'
        
    envtype_lower = envtype.lower()
    traj_load_path = f'/tmp/{envtype_lower}_expert-v2.hdf5'
    d4rl_env_name = f'{envtype_lower}-expert-v2'

    num_trajs = num_trajs
    
    if subsample_dist == 'uniform':
        idxfilename = f'results/{d4rl_env_name}-uniform-freq{subsample_num}-idx.pickle'
    elif subsample_dist == 'uniform-frags':
        idxfilename = f'results/{d4rl_env_name}-uniform-fragments-n{subsample_num}-idx.pickle'
    elif subsample_dist == 'uniform-frags+full-trajs1':
        idxfilename = f'results/{d4rl_env_name}-geometric-fragments-n{subsample_num}-idx-num_full_trajs1.pickle'
    elif subsample_dist == 'geometric':
        idxfilename = f'results/{d4rl_env_name}-geometric-n50-idx.pickle'
    elif subsample_dist == 'geometric-frags':
        idxfilename = f'results/{d4rl_env_name}-geometric-fragments-n{subsample_num}-idx.pickle'
    elif subsample_dist == 'geometric-frags+full-trajs1':
        idxfilename = f'results/{d4rl_env_name}-geometric-fragments-n{subsample_num}-idx-num_full_trajs1.pickle'
    elif subsample_dist == 'beta':
        idxfilename = f'results/{d4rl_env_name}-beta-n50-idx.pickle'
    else:
        raise NotImplementedError
    
    with open(idxfilename, 'rb') as f:
        idxfile = pickle.load(f)

    expert_policy_path = f'dataset/{envtype_lower}_params.sampler.onnx'
    expert_policy = ort.InferenceSession(expert_policy_path)
    
    # if 'WRB' in method:
    #     weighted_replay_sampling = True
    # else:
    weighted_replay_sampling = False
        
    if ('WN' in method) or ('DemoDICE' in method) or ('ADVWBC' in method):
        weight_norm = True
    else:
        weight_norm = False

    configs = dict(
        method=method,
        algorithm=algorithm,
        layer_sizes=[256, 256],
        additional_network_size=256,
        replay_buffer_size=int(1E6),
        traj_load_path='',
        train_num_trajs=num_trajs,
        valid_num_trajs=int(num_trajs*0.2),
        idxfile=idxfile,
        eval_freq=10000,
        lr=3e-5,
        inner_lr=3e-5,
        envtype=envtype_lower,
        d4rl_env_name=d4rl_env_name,
        envname=envname,
        stacksize=stacksize,
        pid=pid,
        save_policy_path=None,   # not save when equals to None
        seed=seed,
        total_iteration=1e6,
        partially_observable=partially_observable,
        use_discriminator_action_input=True,
        info_bottleneck_loss_coef=ib_coef,
        reg_coef=reg_coef,  
        inner_steps=inner_steps,
        batch_size=batch_size,
        # ridge_lambda=ridge_lambda,
        standardize=standardize,
        gamma=gamma,
        expert_policy=expert_policy,
        subsample_dis=subsample_dist,
        weight_norm=weight_norm,
        train_lambda=train_lambda,
        weighted_replay_sampling =weighted_replay_sampling 
    )

    configs['traj_load_path'] = traj_load_path
    configs['save_policy_path'] = f'results/{envname}/{algorithm}/inner_steps{inner_steps}/alpha{reg_coef}/num_trajs{num_trajs}/seed{seed}'
    
    # print(configs)
    train(configs)
