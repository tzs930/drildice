import numpy as np


def preprocess_dataset(mdpfile, start_idx=0, num_dataset=10000):
    
    observations = np.array(mdpfile['observations'])[start_idx:start_idx+num_dataset]
    next_observations = np.array(mdpfile['next_observations'])[start_idx:start_idx+num_dataset]
    
    terminals = np.array(mdpfile['terminals'])[start_idx:start_idx+num_dataset]
    timeouts = np.array(mdpfile['timeouts'])[start_idx:start_idx+num_dataset]
    rewards = np.array(mdpfile['rewards'])[start_idx:start_idx+num_dataset]
    actions = np.array(mdpfile['actions'])[start_idx:start_idx+num_dataset]

    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    init_observations_list = []    
    
    if start_idx == 0:
        idx_from_initial_state = 0
    else:
        # should not be treated as initial observation
        idx_from_initial_state = 1

    for i in range(num_dataset):
        if idx_from_initial_state == 0:
            init_observations_list.append(observations[i])

        idx_from_initial_state += 1
        if terminals[i] or timeouts[i]:
            idx_from_initial_state = 0

    init_observations = np.array(init_observations_list)

    new_paths = {
        'init_observations': init_observations,
        'observations':      observations,
        'next_observations': next_observations,
        'rewards':           rewards,
        'actions':           actions,
        'terminals':         terminals,
        'timeouts':          timeouts        
    }
    
    return new_paths

def preprocess_dataset_with_subsampling(mdpfile, idxfile, start_traj_idx=0, num_trajs=10, add_absorbing_state=False):
    
    observations = np.array(mdpfile['observations'], dtype=float)
    next_observations = np.array(mdpfile['next_observations'], dtype=float)
    
    terminals = np.array(mdpfile['terminals'], dtype=float)
    timeouts = np.array(mdpfile['timeouts'], dtype=float)
    rewards = np.array(mdpfile['rewards'], dtype=float)
    actions = np.array(mdpfile['actions'], dtype=float)
    
    n_data = observations.shape[0]
    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    init_observations_idxs = []
    # terminal_indices = idxfile[start_traj_idx + t]['terminal_idxs']
    
    # idx_from_initial_state = idxfile[start_traj_idx]['start_idx']
    absorbing_states = []
    if add_absorbing_state:        
        absorbing_actions = []
        absorbing_next_states = []
        
        observations = np.concatenate([observations, np.zeros((n_data,1))], axis=-1)
        next_observations = np.concatenate([next_observations, np.zeros((n_data,1))], axis=-1)
        
        # absorbing_state = np.zeros_like(observations[0])
        # absorbing_state[-1] = 1.
        
        # terminals[:] = 0.
        # timeouts[:] = 0.
    
    subsampled_idxs = []
    
    for t in range(num_trajs):
        init_observations_idxs.append(idxfile[start_traj_idx + t]['start_idx'])
        subsampled_idxs += list(idxfile[start_traj_idx + t]['subsampled_idxs'])
        
        if add_absorbing_state:
            terminal_idx = idxfile[start_traj_idx + t]['terminal_idx']

            if terminals[terminal_idx] == 1.:
                terminal_state = next_observations[terminal_idx]
                terminal_action = actions[terminal_idx]
                terminal_state_1 = np.zeros_like(observations[0])
                terminal_state_1[-1] = 1.
                
                absorbing_state = terminal_state_1
                absorbing_action = actions[terminal_idx]
                absorbing_state_1 = terminal_state_1
                
                absorbing_states.append(terminal_state)
                absorbing_actions.append(terminal_action)
                absorbing_next_states.append(terminal_state_1)
                
                absorbing_states.append(absorbing_state)
                absorbing_actions.append(absorbing_action)
                absorbing_next_states.append(absorbing_state_1)
                    
        
    init_observations_idxs = np.array(init_observations_idxs)
    subsampled_idxs = np.array(subsampled_idxs)
    
    init_observations = observations[init_observations_idxs]
    
    observations_ = observations[subsampled_idxs]
    next_observations_ = next_observations[subsampled_idxs]
    rewards_ = rewards[subsampled_idxs]
    actions_ = actions[subsampled_idxs]
    terminals_ = terminals[subsampled_idxs]
    timeouts_ = timeouts[subsampled_idxs]
    
    if add_absorbing_state and len(absorbing_states) > 0:
        absorbing_states = np.array(absorbing_states)
        absorbing_actions = np.array(absorbing_actions)
        absorbing_next_states = np.array(absorbing_next_states)
        
        n_absorbing = len(absorbing_states)
        print(f'** n_absorbing : {n_absorbing}')
        
        observations_ = np.vstack([observations_, absorbing_states])
        actions_ = np.vstack([actions_, absorbing_actions])
        next_observations_ = np.vstack([next_observations_, absorbing_next_states])
        
        rewards_ = np.concatenate([rewards_, np.zeros(n_absorbing)])
        terminals_ = np.concatenate([terminals_, np.zeros(n_absorbing)])
        timeouts_ = np.concatenate([timeouts_, np.zeros(n_absorbing)])

    n_result_data = len(observations_)

    new_paths = {
        'init_observations': init_observations,
        'observations':      observations_,
        'next_observations': next_observations_,
        'rewards':           rewards_,
        'actions':           actions_,
        'terminals':         terminals_,
        'timeouts':          timeouts_
    }
    
    return new_paths, n_result_data

def preprocess_dataset_with_prev_actions(mdpfile, envtype, stacksize=1, partially_observable=False, action_history_len=2):
    
    indx = list(np.arange(20))
    # Indices of position information observations
    if partially_observable:
        envtype_to_idx = {
            'hopper': indx[:5], 
            'ant': indx[:13], 
            'walker2d': indx[:8], 
            'halfcheetah': indx[:4] + indx[8:13]
        }
        obs_idx = envtype_to_idx[envtype]
        observations = np.array(mdpfile['observations'])[:, obs_idx]
        next_observations = np.array(mdpfile['next_observations'])[:, obs_idx]
    else:
        observations = np.array(mdpfile['observations'])
        next_observations = np.array(mdpfile['next_observations'])
    
    terminals = np.array(mdpfile['terminals']) 
    timeouts = np.array(mdpfile['timeouts'])
    rewards = np.array(mdpfile['rewards'])
    actions = np.array(mdpfile['actions'])

    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    n_data = observations.shape[0]
    new_observations_list = []
    new_next_observations_list = []
    prev_action_list = []
    action_history_list = []
    
    idx_from_initial_state = 0
    num_trajs = 0

    for i in range(n_data):
        if idx_from_initial_state == 0:
            prev_action = np.zeros(action_dim)
        else:
            prev_action = actions[i-1]
        prev_action_list.append(prev_action)

        if idx_from_initial_state < stacksize:
            if idx_from_initial_state == 0:
                initial_obs = observations[i]
            
            new_observation = np.zeros(obs_dim * stacksize)
            new_observation_ = np.concatenate(observations[i-idx_from_initial_state: i+1])
            new_observation[-(idx_from_initial_state+1) * obs_dim:] = new_observation_
            
            new_next_observation = np.zeros(obs_dim * stacksize)
            new_next_observation_ = np.concatenate(next_observations[i-idx_from_initial_state: i+1])
            new_next_observation[-(idx_from_initial_state+1) * obs_dim:] = new_next_observation_
            
            if idx_from_initial_state + 1 != stacksize:
                new_next_observation[-(idx_from_initial_state+2) * obs_dim:-(idx_from_initial_state+1) * obs_dim] \
                    = initial_obs
            
        else:
            new_observation = np.concatenate(observations[i+1-stacksize:i+1])
            new_next_observation = np.concatenate(next_observations[i+1-stacksize:i+1])

        if idx_from_initial_state < action_history_len:
            action_history = np.zeros(action_dim * action_history_len)
            action_history_ = np.concatenate(actions[i-idx_from_initial_state: i+1])
            action_history[-(idx_from_initial_state+1) * action_dim:] = action_history_
            
        else:
            action_history = np.concatenate(actions[i+1-action_history_len:i+1])


        new_observations_list.append(new_observation)
        new_next_observations_list.append(new_next_observation)
        action_history_list.append(action_history)

        idx_from_initial_state += 1
        if terminals[i] or timeouts[i]:
            idx_from_initial_state = 0
            num_trajs += 1    

    new_observations = np.array(new_observations_list)
    new_next_observations = np.array(new_next_observations_list)
    new_actions = np.array(action_history_list)

    new_paths = {
        'observations': new_observations,
        'next_observations': new_next_observations,
        'rewards': rewards,
        'actions': new_actions,
        'terminals': terminals,
        'timeouts': timeouts        
    }
    
    return new_paths

def data_select_num_transitions(path, num_transitions=1000, start_idx=0, random=False):
    new_path = {}
    
    if random:
        num_full_trajs = len(path['observations'])
        choice_idx = np.random.choice(num_full_trajs, num_transitions)
        
    else:
        choice_idx = np.arange(start_idx, start_idx + num_transitions)
        
    for key in path.keys():
        new_path[key] = np.array(path[key])[choice_idx]
        
    return new_path