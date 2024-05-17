import d4rl
import gym
import pickle
import numpy as np

import os
os.environ['D4RL_DATASET_DIR'] = './dataset'

def get_uniform_subsampled_idx(dataset, subsample_num):
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    
    n_data = len(terminals)
    
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1
        
        subsample_offset = np.arange(start_idx+1, tidx-1, subsample_num)
        num_random_indices = len(subsample_offset)
        random_indices = np.random.choice(subsample_num, num_random_indices)
        subsampled_idxs = subsample_offset + random_indices
        
        if subsampled_idxs[-1] > tidx:
            subsampled_idxs = subsampled_idxs[:-1]
            
        subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs) + [tidx])
        
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_uniform_fragments_subsampled_idx(dataset, subsample_num, num_full_trajs=1):
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    
    n_data = len(terminals)
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:        
            subsample_start_idx = start_idx + np.random.choice(episode_len)
            subsample_end_idx = min(subsample_start_idx + subsample_num, tidx)            
            subsampled_idxs = np.arange(subsample_start_idx, subsample_end_idx+1)
            # subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs) + [tidx])
            
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_geometric_fragments_subsampled_idx(dataset, p, subsample_num, num_full_trajs=1):
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    
    n_data = len(terminals)
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:        
            # subsample_start_idx = start_idx + np.random.choice(episode_len)
            subsample_start_idx = start_idx + np.clip(np.random.geometric(p), 0, episode_len-1)
            subsample_end_idx = min(subsample_start_idx + subsample_num, tidx)            
            subsampled_idxs = np.arange(subsample_start_idx, subsample_end_idx+1)
            # subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs))
        
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict


def get_action_dependent_sampled_idx(dataset, positive_action_ratio=0.5, n=50, num_full_trajs=0):
    action_median = np.median(dataset['actions'])
    # np.count_nonzero(dataset['actions'] >= action_median) : 52590
    # np.count_nonzero(dataset['actions'] <  action_median) : 52590
    
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    actions = np.array(dataset['actions'])
    
    n_data = len(terminals)
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:        
            # subsample_start_idx = start_idx + np.random.choice(episode_len)
            total_idxs = np.arange(start_idx, tidx+1)
            total_pos_action_idxs = total_idxs[np.nonzero(actions[total_idxs] >= action_median)[0]]
            total_neg_action_idxs = total_idxs[np.nonzero(actions[total_idxs] <  action_median)[0]]
            
            num_pos_act_idxs = int(positive_action_ratio * n)
            num_neg_act_idxs = n - num_pos_act_idxs
            
            subsampled_pos_indxs = np.random.choice(total_pos_action_idxs, num_pos_act_idxs, replace=False)
            subsampled_neg_indxs = np.random.choice(total_neg_action_idxs, num_neg_act_idxs, replace=False)
            
            subsampled_idxs = np.sort(list(subsampled_pos_indxs) + list(subsampled_neg_indxs))
            subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs))
        
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_state_dependent_sampled_idx(dataset, positive_position_ratio=0.5, n=50, num_full_trajs=1):
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    observations = np.array(dataset['observations'])
    
    n_data = len(terminals)
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    state_mean = np.mean(dataset['observations'], axis=0)
    # eps_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # select optimal eps that evenly divides the dataset
    smallest_diff = np.inf
    smallest_num_inner_ball_states = None
    smallest_num_outer_ball_states = None
    
    # for eps in eps_list:
    l2_distances_total = np.sqrt(np.sum((observations - state_mean[None])**2, axis=1))
    eps = np.median(l2_distances_total)
    num_inner_ball_states = np.sum(l2_distances_total <= eps)
    num_outer_ball_states = n_data - num_inner_ball_states
    diff = abs(num_inner_ball_states - num_outer_ball_states)
            
    print(f'** optimal_eps: {eps} | num_inner_ball_states: {num_inner_ball_states} | num_outer_ball_states: {num_outer_ball_states} | diff: {diff}')
    # state1_median = np.median(dataset['observations'][:, 1])
    
    # ==state0_median==
    # np.count_nonzero(dataset['observations'][:, 0] >= state0_median ) : 52590
    # np.count_nonzero(dataset['observations'][:, 0] <  state0_median ) : 52590
    # ==state1_median==
    # np.count_nonzero(dataset['observations'][:, 1] >= state1_median ) : 52590
    # np.count_nonzero(dataset['observations'][:, 1] <  state1_median ) : 52590
    idx_dict = {}
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:        
            # subsample_start_idx = start_idx + np.random.choice(episode_len)
            total_idxs = np.arange(start_idx, tidx+1)
            
            l2_distances = np.sqrt(np.sum((observations[total_idxs] - state_mean[None])**2, axis=1))
            
            positive_indices = (l2_distances >=  eps)
            negative_indices = (l2_distances <  eps)
            
            total_pos_action_idxs = total_idxs[np.nonzero(positive_indices)[0]]
            total_neg_action_idxs = total_idxs[np.nonzero(negative_indices)[0]]
            
            num_pos_act_idxs = int(positive_position_ratio * n)
            num_neg_act_idxs = n - num_pos_act_idxs
            
            subsampled_pos_indxs = np.random.choice(total_pos_action_idxs, num_pos_act_idxs, replace=False or len(total_pos_action_idxs) < num_pos_act_idxs)
            subsampled_neg_indxs = np.random.choice(total_neg_action_idxs, num_neg_act_idxs, replace=False or len(total_neg_action_idxs) < num_neg_act_idxs)
            
            subsampled_idxs = np.sort(list(subsampled_pos_indxs) + list(subsampled_neg_indxs))
            subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs))
        
        idx_dict[traj_count] = {
            'start_idx':        start_idx,
            'terminal_idx':     tidx,
            'subsampled_idxs':  subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_action_dependent_sampled_idx(dataset, positive_position_ratio=0.5, n=50, num_full_trajs=1):
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    observations = np.array(dataset['observations'])
    actions = np.array(dataset['actions'])
    
    n_data = len(terminals)
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    action_mean = np.mean(dataset['actions'], axis=0)
    
    # select optimal eps that evenly divides the dataset
    l2_distances_total = np.sqrt(np.sum((actions - action_mean[None])**2, axis=1))
    eps = np.median(l2_distances_total)
    num_inner_ball_states = np.sum(l2_distances_total <= eps)
    num_outer_ball_states = n_data - num_inner_ball_states
    diff = abs(num_inner_ball_states - num_outer_ball_states)
            
    print(f'** optimal_eps: {eps} | num_inner_ball_states: {num_inner_ball_states} | num_outer_ball_states: {num_outer_ball_states} | diff: {diff}')
    # state1_median = np.median(dataset['observations'][:, 1])
    
    # ==state0_median==
    # np.count_nonzero(dataset['observations'][:, 0] >= state0_median ) : 52590
    # np.count_nonzero(dataset['observations'][:, 0] <  state0_median ) : 52590
    # ==state1_median==
    # np.count_nonzero(dataset['observations'][:, 1] >= state1_median ) : 52590
    # np.count_nonzero(dataset['observations'][:, 1] <  state1_median ) : 52590
    idx_dict = {}
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:        
            # subsample_start_idx = start_idx + np.random.choice(episode_len)
            total_idxs = np.arange(start_idx, tidx+1)
            
            l2_distances = np.sqrt(np.sum((actions[total_idxs] - action_mean[None])**2, axis=1))
            
            positive_indices = (l2_distances >=  eps)
            negative_indices = (l2_distances <  eps)
            
            total_pos_action_idxs = total_idxs[np.nonzero(positive_indices)[0]]
            total_neg_action_idxs = total_idxs[np.nonzero(negative_indices)[0]]
            
            num_pos_act_idxs = int(positive_position_ratio * n)
            num_neg_act_idxs = n - num_pos_act_idxs
            
            subsampled_pos_indxs = np.random.choice(total_pos_action_idxs, num_pos_act_idxs, replace=False or len(total_pos_action_idxs) < num_pos_act_idxs)
            subsampled_neg_indxs = np.random.choice(total_neg_action_idxs, num_neg_act_idxs, replace=False or len(total_neg_action_idxs) < num_neg_act_idxs)
            
            subsampled_idxs = np.sort(list(subsampled_pos_indxs) + list(subsampled_neg_indxs))
            subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs))
        
        idx_dict[traj_count] = {
            'start_idx':        start_idx,
            'terminal_idx':     tidx,
            'subsampled_idxs':  subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_beta_subsampled_idx(dataset, alpha=0.5, beta=0.5, n=50, num_full_trajs=1):
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    
    n_data = len(terminals)
    
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:
            episode_len = (tidx - start_idx) + 1
            
            subsample_offset = start_idx
            subsampled_idxs = subsample_offset + np.array(np.random.beta(alpha, beta,  n) * episode_len, dtype=int)
            subsampled_idxs.sort()
            
            while subsampled_idxs[-1] >= tidx:
                subsampled_idxs = subsampled_idxs[:-1]
                
            subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs) + [tidx])
        
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict

def get_geometric_subsampled_idx(dataset, p, subsample_num, num_full_trajs=1):
    idx_dict = {}
    
    terminals = np.array(dataset['terminals'])
    timeouts = np.array(dataset['timeouts'])
    
    n_data = len(terminals)
    
    traj_count = 0
    
    terminal_idxs = np.nonzero(np.logical_or(terminals, timeouts))[0]
    
    start_idx = 0
    for tidx in terminal_idxs:
        episode_len = (tidx - start_idx) + 1

        if traj_count < num_full_trajs:
            subsampled_idxs = np.arange(start_idx, tidx+1)

        else:
            subsample_offset = start_idx
            subsample_start_idx = start_idx + np.random.geometric(p)
            subsample_end_idx = min(subsample_start_idx + subsample_num, tidx)
            
            subsampled_idxs = np.arange(subsample_start_idx, subsample_end_idx+1)
            subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs))
            # subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs) + [tidx])
            
            while subsampled_idxs[-1] >= tidx:
                subsampled_idxs = subsampled_idxs[:-1]
        # subsampled_idxs = np.unique([start_idx] + list(subsampled_idxs) + [tidx])
        
        idx_dict[traj_count] = {
            'start_idx': start_idx,
            'terminal_idx': tidx,
            'subsampled_idxs': subsampled_idxs
        }
        
        traj_count += 1
        start_idx = tidx + 1
    
    return idx_dict



if __name__ == "__main__":
    np.random.seed(0)
    
    envlist = ['hopper', 'walker2d', 'halfcheetah']
    # subsample_methods = ['uniform', 'beta', 'geometric', 'uniform-fragment', 'geometric-fragment']
    # subsample_methods = ['uniform-fragments', 'geometric-fragments']
    subsample_methods = [ 'beta']
    alpha_beta_list = [(1.0, 1.0), (10.0, 1.0), (1.0, 10.0), (10.0, 10.0)]
    # ratiolist = [0.1, 0.3, 0.5, 0.7, 0.9]
    subsample_num_list = [50]
    
    for env in envlist:
        for subsample_method in subsample_methods:
            d4rl_envname = f'{env}-expert-v2'
            d4rl_env = gym.make(d4rl_envname)
            dataset = d4rl_env.get_dataset()
        
            # save dictionary: idx_dict[traj_idx] = {'start_idx', 'terminal_idx', 'subsampled_idxs'}
            if subsample_method == 'uniform-fragments':
                for subsample_num in subsample_num_list:
                    filename = f'results/{d4rl_envname}-uniform-fragments-n{subsample_num}-idx-num_full_trajs1-no-init.pickle'
                    idx_dict = get_uniform_fragments_subsampled_idx(dataset, subsample_num, num_full_trajs=1)
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                    print(f'{filename} saved!')

            elif subsample_method == 'geometric-fragments':
                for subsample_num in subsample_num_list:
                    # filename = f'results/{d4rl_envname}-geometric-fragments-n{subsample_num}-idx.pickle'
                    filename = f'results/{d4rl_envname}-geometric-fragments-n{subsample_num}-idx-num_full_trajs1-no-init.pickle'
                    p = 0.01
                    idx_dict = get_geometric_fragments_subsampled_idx(dataset, p, subsample_num, num_full_trajs=1)
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                    print(f'{filename} saved!')
                
            elif subsample_method == 'state-dependent':
                for subsample_num in subsample_num_list:
                    for ratio in ratiolist:
                        # filename = f'results/{d4rl_envname}-geometric-fragments-n{subsample_num}-idx.pickle'
                        filename = f'results/{d4rl_envname}-{subsample_method}-n{subsample_num}-r{ratio}-idx-l2-median-full-trajs1.pickle'
                        
                        idx_dict = get_state_dependent_sampled_idx(dataset, ratio, subsample_num, num_full_trajs=1)
                        
                        with open(filename, 'wb') as f:
                            pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        print(f'{filename} saved!')
                
            elif subsample_method == 'action-dependent':
                for subsample_num in subsample_num_list:
                    for ratio in ratiolist:
                        # filename = f'results/{d4rl_envname}-geometric-fragments-n{subsample_num}-idx.pickle'
                        filename = f'results/{d4rl_envname}-{subsample_method}-n{subsample_num}-r{ratio}-idx-l2-median-full-trajs1.pickle'
                        
                        idx_dict = get_action_dependent_sampled_idx(dataset, ratio, subsample_num, num_full_trajs=1)
                        
                        with open(filename, 'wb') as f:
                            pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        print(f'{filename} saved!')
                
            elif subsample_method == 'uniform':
                for subsample_num in subsample_num_list:
                    filename = f'results/{d4rl_envname}-uniform-freq{subsample_num}-idx.pickle'
                    idx_dict = get_uniform_subsampled_idx(dataset, subsample_num)
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                    print(f'{filename} saved!')
            
            elif subsample_method == 'geometric':
                for subsample_num in subsample_num_list:
                    p = 0.005
                    
                    filename = f'results/{d4rl_envname}-geometric-n{subsample_num}-idx.pickle'
                    idx_dict = get_geometric_subsampled_idx(dataset, p, subsample_num)
                        
                    with open(filename, 'wb') as f:
                        pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                    print(f'{filename} saved!')
                
            elif subsample_method == 'beta':
                for subsample_num in subsample_num_list:
                    for alpha_beta in alpha_beta_list:
                        alpha, beta  = alpha_beta
                        filename = f'results/{d4rl_envname}-beta-n{subsample_num}_alpha{alpha}-beta{beta}-idx-full-trajs1.pickle'
                        idx_dict = get_beta_subsampled_idx(dataset, alpha=alpha, beta=beta, n=subsample_num)
                            
                        with open(filename, 'wb') as f:
                            pickle.dump(idx_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                        print(f'{filename} saved!')
            
        