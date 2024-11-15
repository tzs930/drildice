import cvxpy as cp
import numpy as np

from mdp import construct_l2_feature, evaluate_policy

LOGIT_MARGIN = np.log(100.)
cost_scale = 1.

def train_BC_agent(mdp, seed, trajectory_dict, state_space, pi_expert, num_points_per_axis=5, verbose=True):
    """
    Train a behavior cloning agent using the expert trajectories.
    """
    # Define the state space
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    next_features = trajectory_dict['next_features']
    dones = trajectory_dict['dones']
    
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
    
    tau = 1.
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_loss = cp.mean( cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True))
    
    # constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN] #., cp.sum(logits, axis=1) == 1.]
    constraints = [logits.mean() == 0, logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    bc_problem = cp.Problem(cp.Minimize(bc_loss), constraints=constraints)
    result = bc_problem.solve(solver='MOSEK', verbose=verbose)

    # Evaluate the learned policy
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # state_action_mapping = np.argmax(feature_space @ W_pi.value, -1)
    # policy_s_a = np.zeros( (mdp.S, mdp.A) )
    # policy_s_a[np.arange(mdp.S), state_action_mapping] = 1.

    logits_feature = feature_space @ W_pi.value * tau
    log_sum_exp_logits = cp.log_sum_exp(logits_feature, axis=1, keepdims=True).value
    policy_s_a = cp.exp(logits_feature - log_sum_exp_logits).value #/ np.exp(logits_feature).sum(axis=1, keepdims=True)
    
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
        
    # print(f'-- Return: {np.mean(return_list)} +- {np.std(return_list)/np.sqrt(len(return_list))}')
    # print(f'-- Target MSE: {np.mean(target_mse_list)} +- {np.std(target_mse_list)/np.sqrt(len(target_mse_list))}')
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
    
    return policy_s_a, stats, W_pi.value

def train_DRBC_agent(mdp, seed, trajectory_dict, state_space, pi_expert, state_to_pos, W_bc_pi=None, initial_state_pos=None, alpha=0.1, num_points_per_axis=5, verbose=False):
    """
    Train a behavior cloning agent using the expert trajectories.
    """
    # Define the state space
    
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    features_next = trajectory_dict['next_features']
    dones = trajectory_dict['dones']
    
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    num_states = len(state_space)
    num_datasets = features.shape[0] 
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    features_init = construct_l2_feature(np.array([list(initial_state_pos)] * num_datasets), num_points_per_axis=num_points_per_axis)

    # step 1: Behavior cloning
    tau = 1.
    
    if W_bc_pi is None:
        W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
        
        logits = features @ W_pi * tau
        softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
        log_px = logits - softmax_logits
        bc_loss = cp.mean( cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True) )
        
        constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
        bc_problem = cp.Problem(cp.Minimize(bc_loss), constraints=constraints)
        result = bc_problem.solve(solver='MOSEK', verbose=verbose)
        
        W_pi = W_pi.value
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    else:
        W_pi = W_bc_pi
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    # Evaluate the learned policy
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # y_preds_total = np.exp(feature_space @ W_pi) / np.exp(feature_space @ W_pi).sum(axis=1, keepdims=True)
    
    # state_action_mapping = np.argmax(y_preds_total, -1)
    # policy_s_a_pre = np.zeros( (mdp.S, mdp.A) )
    # policy_s_a_pre[np.arange(mdp.S), state_action_mapping] = 1.
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a_pre = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a_pre, pi_expert)
    
    # step 2: Dual variable learning (eta)
    # policy_loss = np.linalg.norm(y_preds - actions, axis=1)**2
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_losses = -cp.multiply(actions, log_px).sum(axis=1, keepdims=True)
    
    min_dual = -1.
    max_dual = (1. + alpha) * np.max( bc_losses.value )
    
    sup = np.max(bc_losses.value, keepdims=True)
    eta = cp.Variable(1)
    
    res = cp.mean(cp.pos(bc_losses - eta))
    res += alpha * cp.pos(sup - eta)
    res += eta
    
    constraints = [eta >= min_dual, eta <= max_dual]
    dual_problem = cp.Problem(cp.Minimize(res), constraints=constraints)
    result = dual_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ============================= 1-0 cost ============================ #
    # logits = features @ W_pi * tau
    # policy_actions_discrete = np.argmax(logits, -1)
    # expert_actions_discrete = np.argmax(actions, -1)
    
    # costs = np.array((policy_actions_discrete != expert_actions_discrete), dtype=np.float32).reshape(-1, 1)    
    # costs_cum = costs * cost_scale
    # ============================= CE cost ============================ #
    
    # logits = features @ W_pi * tau
    # softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    # log_px = logits - softmax_logits
    # costs = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    # costs_cum = costs * cost_scale
    
    # e_nu = costs_cum + gamma * nu_preds_next - nu_preds
    # # e_nu = costs_cum + (1. - dones) * gamma * nu_preds_next - nu_preds
    # w_opt_nu = cp.exp(e_nu / alpha - 1.)
    # nu_loss_1 = cp.mean(alpha * w_opt_nu)
    # nu_loss = (1 - gamma) * nu_loss_0 + nu_loss_1

    # step 2: DR-BC
    W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
    
    logits = features @ W_pi #* tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    # bc_loss = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    bc_losses = -cp.multiply(actions, log_px).sum(axis=1, keepdims=True)
    sup = cp.max(bc_losses, keepdims=True)
    
    drbc_loss = cp.mean(cp.pos(bc_losses - eta.value))
    drbc_loss += alpha * cp.pos(sup - eta.value)
    drbc_loss += eta.value

    # constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN] #., cp.sum(logits, axis=1) == 1.]
    constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    # constraints = []
    
    bc_problem = cp.Problem(cp.Minimize(drbc_loss), constraints=constraints)
    result = bc_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ====== Evaluation =====#
    logits_feature = feature_space @ W_pi.value * tau
    softmax_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True).value
    log_p_feature = logits_feature - softmax_logits_feature
    # policy_s_a = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    policy_s_a = np.exp(log_p_feature)
    
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
        
    # nu_total = feature_space @ W_nu.valueX
    
    pos_next_total  = []
    for s in range(mdp.S):
        a = np.argmax(policy_s_a[s])
        s_next = np.argmax(mdp.T[s, a])
        pos_next = state_to_pos[s_next]
        pos_next_total.append(pos_next)
    
    next_feature_total = construct_l2_feature(np.array(pos_next_total), num_points_per_axis=num_points_per_axis)
    # nu_next_total = next_feature_total @ W_nu.value
    
    # ============================= 1-0 cost ============================ #
    costs_total = cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True) * cost_scale
    costs_total = np.array((np.argmax(pi_expert, -1) != np.argmax(policy_s_a, -1)), dtype=np.float32).reshape(-1, 1)
    
    # ============================= CE cost ============================ #
    # costs_total = cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True).value * cost_scale
    # e_nu_total = costs_total + gamma * nu_next_total - nu_total
    # w_nu = np.exp(e_nu_total / alpha - 1.).reshape(-1)
    # print(f'-- Return: {np.mean(return_list)} +- {np.std(return_list)/np.sqrt(len(return_list))}')
    # print(f'-- Target MSE: {np.mean(target_mse_list)} +- {np.std(target_mse_list)/np.sqrt(len(target_mse_list))}')
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
      
    return policy_s_a, stats, _, policy_s_a_pre

def train_DrilDICE_agent(mdp, seed, trajectory_dict, state_space, pi_expert, state_to_pos, W_bc_pi=None, initial_state_pos=None, gamma=0.99, alpha=0.1, num_points_per_axis=5, verbose=False):
    """
    Train a behavior cloning agent using the expert trajectories.
    """
    # Define the state space
    
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    features_next = trajectory_dict['next_features']
    dones = trajectory_dict['dones']
    
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    num_states = len(state_space)
    num_datasets = features.shape[0] 
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    features_init = construct_l2_feature(np.array([list(initial_state_pos)] * num_datasets), num_points_per_axis=num_points_per_axis)

    # step 1: Behavior cloning
    tau = 1.
    
    if W_bc_pi is None:
        W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
        
        logits = features @ W_pi * tau
        softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
        log_px = logits - softmax_logits
        bc_loss = cp.mean( cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True) )
        
        constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
        bc_problem = cp.Problem(cp.Minimize(bc_loss), constraints=constraints)
        result = bc_problem.solve(solver='MOSEK', verbose=verbose)
        
        W_pi = W_pi.value
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    else:
        W_pi = W_bc_pi
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    # Evaluate the learned policy
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # y_preds_total = np.exp(feature_space @ W_pi) / np.exp(feature_space @ W_pi).sum(axis=1, keepdims=True)
    
    # state_action_mapping = np.argmax(y_preds_total, -1)
    # policy_s_a_pre = np.zeros( (mdp.S, mdp.A) )
    # policy_s_a_pre[np.arange(mdp.S), state_action_mapping] = 1.
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a_pre = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a_pre, pi_expert)
    
    # step 2: Weight learning
    W_nu = cp.Variable((num_features, 1))  # w = W_nu^T \phi(s)
    
    nu_inits = features_init @ W_nu
    nu_preds = features @ W_nu
    nu_preds_next = features_next @ W_nu
    nu_loss_0 = cp.mean(nu_inits)
    
    # f: x * log(x) ; KL-Divergence
    # costs = ((y_preds - actions) ** 2).mean(-1, keepdims=True)
    # costs_cum = costs #-costs
    # costs = - (actions * np.log(y_preds + 1e-6)).sum(-1, keepdims=True)
    
    # ============================= 1-0 cost ============================ #
    logits = features @ W_pi * tau
    policy_actions_discrete = np.argmax(logits, -1)
    expert_actions_discrete = np.argmax(actions, -1)
    
    costs = np.array((policy_actions_discrete != expert_actions_discrete), dtype=np.float32).reshape(-1, 1)    
    costs_cum = costs * cost_scale
    # ============================= CE cost ============================ #
    
    # logits = features @ W_pi * tau
    # softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    # log_px = logits - softmax_logits
    # costs = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    # costs_cum = costs * cost_scale
    
    e_nu = costs_cum + gamma * nu_preds_next - nu_preds
    # e_nu = costs_cum + (1. - dones) * gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu / alpha - 1.)
    nu_loss_1 = cp.mean(alpha * w_opt_nu)
    nu_loss = (1 - gamma) * nu_loss_0 + nu_loss_1

    # constraints = [W_nu_norm <= 100.]
    constraints = []
    dice_problem = cp.Problem(cp.Minimize(nu_loss), constraints=constraints)
    result = dice_problem.solve(solver='MOSEK', verbose=verbose)
    
    # step 3: Weighted BC
    W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
    
    nu_preds = features @ W_nu.value
    nu_preds_next = features_next @ W_nu.value
    e_nu = costs_cum + gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu / alpha - 1.).value
    w = w_opt_nu
    
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_loss = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    wloss = cp.mean(cp.multiply(w, bc_loss))  #+ l2_coef * l2_reg

    # constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN] #., cp.sum(logits, axis=1) == 1.]
    constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    
    bc_problem = cp.Problem(cp.Minimize(wloss), constraints=constraints)

    result = bc_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ====== Evaluation =====#
    logits_feature = feature_space @ W_pi.value * tau
    softmax_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True).value
    log_p_feature = logits_feature - softmax_logits_feature
    # policy_s_a = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    policy_s_a = np.exp(log_p_feature)
    
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
        
    nu_total = feature_space @ W_nu.value
    
    pos_next_total  = []
    for s in range(mdp.S):
        a = np.argmax(policy_s_a[s])
        s_next = np.argmax(mdp.T[s, a])
        pos_next = state_to_pos[s_next]
        pos_next_total.append(pos_next)
    
    next_feature_total = construct_l2_feature(np.array(pos_next_total), num_points_per_axis=num_points_per_axis)
    nu_next_total = next_feature_total @ W_nu.value
    
    # ============================= 1-0 cost ============================ #
    costs_total = cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True) * cost_scale
    costs_total = np.array((np.argmax(pi_expert, -1) != np.argmax(policy_s_a, -1)), dtype=np.float32).reshape(-1, 1)
    
    # ============================= CE cost ============================ #
    # costs_total = cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True).value * cost_scale
    
    e_nu_total = costs_total + gamma * nu_next_total - nu_total
    w_nu = np.exp(e_nu_total / alpha - 1.).reshape(-1)

    # print(f'-- Return: {np.mean(return_list)} +- {np.std(return_list)/np.sqrt(len(return_list))}')
    # print(f'-- Target MSE: {np.mean(target_mse_list)} +- {np.std(target_mse_list)/np.sqrt(len(target_mse_list))}')
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
      
    return policy_s_a, stats, w_nu, policy_s_a_pre

def train_OPTIDICEBC_agent(mdp, seed, trajectory_dict, state_space, pi_expert, state_to_pos, W_bc_pi=None, initial_state_pos=None, gamma=0.99, alpha=0.1, num_points_per_axis=5, verbose=False):
     # Define the state space
    
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    features_next = trajectory_dict['next_features']
    dones = trajectory_dict['dones']
    
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    num_states = len(state_space)
    num_datasets = features.shape[0] 
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    features_init = construct_l2_feature(np.array([list(initial_state_pos)] * num_datasets), num_points_per_axis=num_points_per_axis)

    # step 1: Behavior cloning
    tau = 1.
    
    if W_bc_pi is None:
        W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
        
        logits = features @ W_pi * tau
        softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
        log_px = logits - softmax_logits
        bc_loss = cp.mean( cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True) )
        
        constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
        bc_problem = cp.Problem(cp.Minimize(bc_loss), constraints=constraints)
        result = bc_problem.solve(solver='MOSEK', verbose=verbose)
        
        W_pi = W_pi.value
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    else:
        W_pi = W_bc_pi
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    # Evaluate the learned policy
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # y_preds_total = np.exp(feature_space @ W_pi) / np.exp(feature_space @ W_pi).sum(axis=1, keepdims=True)
    
    # state_action_mapping = np.argmax(y_preds_total, -1)
    # policy_s_a_pre = np.zeros( (mdp.S, mdp.A) )
    # policy_s_a_pre[np.arange(mdp.S), state_action_mapping] = 1.
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a_pre = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a_pre, pi_expert)
    
    # step 2: Weight learning
    W_nu = cp.Variable((num_features, 1))  # w = W_nu^T \phi(s)
    
    nu_inits = features_init @ W_nu
    nu_preds = features @ W_nu
    nu_preds_next = features_next @ W_nu
    nu_loss_0 = cp.mean(nu_inits)
    
    # f: x * log(x) ; KL-Divergence
    # costs = ((y_preds - actions) ** 2).mean(-1, keepdims=True)
    # costs_cum = costs #-costs
    # costs = - (actions * np.log(y_preds + 1e-6)).sum(-1, keepdims=True)
    
    # ============================= 1-0 cost ============================ #
    logits = features @ W_pi * tau
    policy_actions_discrete = np.argmax(logits, -1)
    expert_actions_discrete = np.argmax(actions, -1)
    
    costs = - np.array((policy_actions_discrete != expert_actions_discrete), dtype=np.float32).reshape(-1, 1)    
    costs_cum = costs * cost_scale
    # ============================= CE cost ============================ #
    
    # logits = features @ W_pi * tau
    # softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    # log_px = logits - softmax_logits
    # costs = -cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    # costs_cum = costs * cost_scale
    
    e_nu = costs_cum + gamma * nu_preds_next - nu_preds
    # e_nu = costs_cum + (1. - dones) * gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu / alpha - 1.)
    nu_loss_1 = cp.mean(alpha * w_opt_nu)
    nu_loss = (1 - gamma) * nu_loss_0 + nu_loss_1

    # constraints = [W_nu_norm <= 100.]
    constraints = []
    dice_problem = cp.Problem(cp.Minimize(nu_loss), constraints=constraints)
    result = dice_problem.solve(solver='MOSEK', verbose=verbose)
    
    # step 3: Weighted BC
    W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
    
    nu_preds = features @ W_nu.value
    nu_preds_next = features_next @ W_nu.value
    e_nu = costs_cum + gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu / alpha - 1.).value
    w = w_opt_nu
    
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_loss = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    wloss = cp.mean(cp.multiply(w, bc_loss))  #+ l2_coef * l2_reg

    # constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN] #., cp.sum(logits, axis=1) == 1.]
    constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    
    bc_problem = cp.Problem(cp.Minimize(wloss), constraints=constraints)

    result = bc_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ====== Evaluation =====#
    logits_feature = feature_space @ W_pi.value * tau
    softmax_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True).value
    log_p_feature = logits_feature - softmax_logits_feature
    # policy_s_a = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    policy_s_a = np.exp(log_p_feature)
    
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
        
    nu_total = feature_space @ W_nu.value
    
    pos_next_total  = []
    for s in range(mdp.S):
        a = np.argmax(policy_s_a[s])
        s_next = np.argmax(mdp.T[s, a])
        pos_next = state_to_pos[s_next]
        pos_next_total.append(pos_next)
    
    next_feature_total = construct_l2_feature(np.array(pos_next_total), num_points_per_axis=num_points_per_axis)
    nu_next_total = next_feature_total @ W_nu.value
    
    # ============================= 1-0 cost ============================ #
    costs_total = cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True) * cost_scale
    costs_total = - np.array((np.argmax(pi_expert, -1) != np.argmax(policy_s_a, -1)), dtype=np.float32).reshape(-1, 1)
    
    # ============================= CE cost ============================ #
    # costs_total = -cp.sum(-cp.multiply(pi_expert, log_p_feature), axis=1, keepdims=True).value * cost_scale
    
    e_nu_total = costs_total + gamma * nu_next_total - nu_total
    w_nu = np.exp(e_nu_total / alpha - 1.).reshape(-1)

    # print(f'-- Return: {np.mean(return_list)} +- {np.std(return_list)/np.sqrt(len(return_list))}')
    # print(f'-- Target MSE: {np.mean(target_mse_list)} +- {np.std(target_mse_list)/np.sqrt(len(target_mse_list))}')
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
      
    return policy_s_a, stats, w_nu, policy_s_a_pre

def train_DEMODICE_agent(mdp, seed, trajectory_dict, state_space, pi_expert, state_to_pos, initial_state_pos=None, gamma=0.99, alpha=0.1, WIDTH=11, HEIGHT=11, num_points_per_axis=5, verbose=False):
    """
    Train a behavior cloning agent using the expert trajectories.
    """
    # Define the state space
    tau = 1.0
    
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    features_next = trajectory_dict['next_features']
    dones = trajectory_dict['dones']
    
    num_states = len(state_space)
    num_datasets = features.shape[0] 
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    features_init = construct_l2_feature(np.array([list(initial_state_pos)] * num_datasets), num_points_per_axis=num_points_per_axis)

    W_nu = cp.Variable((num_features, 1))  # w = W_nu^T \phi(s)
    
    nu_inits = features_init @ W_nu
    # nu_inits = features @ W_nu
    nu_preds = features @ W_nu
    nu_preds_next = features_next @ W_nu
    nu_loss_0 = cp.mean(nu_inits)
    
    # f: x * log(x) ; KL-Divergence
    e_nu = gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu / alpha - 1.)
    nu_loss_1 = cp.mean(alpha * w_opt_nu)                                
    nu_loss = (1 - gamma) * nu_loss_0 + nu_loss_1
    
    constraints = []
    dice_problem = cp.Problem(cp.Minimize(nu_loss), constraints=constraints)
    result = dice_problem.solve(solver='MOSEK', verbose=verbose)
    
    # step 2. weighted BC
    W_pi = cp.Variable((num_features, num_actions))
    
    nu_preds = features @ W_nu.value
    nu_preds_next = features_next @ W_nu.value
    e_nu = gamma * nu_preds_next - nu_preds
    w_opt_nu = cp.exp(e_nu - 1.)
    w = w_opt_nu
    
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_loss = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    wloss = cp.mean(cp.multiply(w, bc_loss))
    
    constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    
    bc_problem = cp.Problem(cp.Minimize(wloss), constraints=constraints)
    
    # wmse = cp.mean(cp.multiply(w, cp.mean( ((y_preds - actions) ** 2), axis=1, keepdims=True)))    
    # bc_problem = cp.Problem(cp.Minimize(wmse), constraints=constraints)

    result = bc_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ====== Evaluation =====#
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # logits_feature = feature_space @ W_pi.value * tau
    # softmax_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    # policy_s_a = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
    
    nu_total = feature_space @ W_nu.value
    
    pos_next_total  = []
    for s in range(mdp.S):
        a = np.argmax(policy_s_a[s])
        s_next = np.argmax(mdp.T[s, a])
        pos_next = state_to_pos[s_next]
        pos_next_total.append(pos_next)
    
    next_feature_total = construct_l2_feature(np.array(pos_next_total), num_points_per_axis=num_points_per_axis)
    nu_next_total = next_feature_total @ W_nu.value
    
    e_nu_total = gamma * nu_next_total - nu_total
    w_nu = np.exp(e_nu_total - 1.).reshape(-1)
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
    
    return policy_s_a, stats, w_nu

def train_ADWBC_agent(mdp, seed, trajectory_dict, state_space, pi_expert, W_bc_pi=None, WIDTH=11, HEIGHT=11, num_points_per_axis=5, verbose=False):
    """
    Train a behavior cloning agent using the expert trajectories.
    """
    # Define the state space
    
    tau = 1.0
    features = trajectory_dict['features']
    actions = trajectory_dict['actions']
    features_next = trajectory_dict['next_features']
    
    num_states = len(state_space)
    num_datasets = features.shape[0] 
    num_features = features.shape[-1]
    num_actions = actions.shape[-1]
    
    if W_bc_pi is None:
        W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
        # y_preds = features @ W_pi
        
        logits = features @ W_pi    
        softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
        bc_loss = cp.sum(-cp.multiply(actions, logits) + softmax_logits)
        
        constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
        bc_problem = cp.Problem(cp.Minimize(bc_loss), constraints=constraints)
        result = bc_problem.solve(solver='MOSEK', verbose=verbose)
        
        W_pi = W_pi.value
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
        
    else:
        W_pi = W_bc_pi
        y_preds = np.exp(features @ W_pi) / np.exp(features @ W_pi).sum(axis=1, keepdims=True)
    
    # Evaluate the learned policy
    feature_space = construct_l2_feature(np.array(state_space), num_points_per_axis=num_points_per_axis)
    # y_preds_total = np.exp(feature_space @ W_pi) / np.exp(feature_space @ W_pi).sum(axis=1, keepdims=True)
    # state_action_mapping = np.argmax(y_preds_total, -1)
    # policy_s_a_pre = np.zeros( (mdp.S, mdp.A) )
    # policy_s_a_pre[np.arange(mdp.S), state_action_mapping] = 1.
    # _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a_pre, pi_expert)
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a_pre = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a_pre, pi_expert)
    
    # ========2. weight learning========
    W_nu = cp.Variable((num_features, 1))  # w = W_nu^T \phi(s)
    
    # y_preds  = (features @ W_pi)
    # w_logit = features @ W_nu
    w_preds = features @ W_nu
    sup_loss = - (actions * np.log(y_preds)).sum(-1, keepdims=True)
    weighted_sup_loss = cp.mean(cp.multiply(w_preds, sup_loss))
    W_norm = cp.norm(W_nu, 'fro') ** 2
    # constraints = [ (sup_loss.mean() - weighted_sup_loss)**2 <= 1., w_preds >= 0.]
    constraints = [W_norm <= 0.5, w_preds.mean() == 1., w_preds >= 0.]
    w_problem = cp.Problem(cp.Maximize(weighted_sup_loss), constraints=constraints)
    result = w_problem.solve(solver='MOSEK', verbose=verbose)
    
    # ========3. weighted BC========
    W_pi = cp.Variable((num_features, num_actions))  # pi = softmax(W_pi^T \phi(s))
    # y_preds = features @ W_pi
    
    w_preds = (features @ W_nu).value
    w = np.maximum(w_preds, 0)
    # w = w_preds / w_preds.mean()
    
    logits = features @ W_pi * tau
    softmax_logits = cp.log_sum_exp(logits, axis=1, keepdims=True)
    log_px = logits - softmax_logits
    bc_loss = cp.sum(-cp.multiply(actions, log_px), axis=1, keepdims=True)
    wloss = cp.mean(cp.multiply(w, bc_loss))  #+ l2_coef * l2_reg
    
    # W_norm = cp.norm(W_pi, 'fro') ** 2
    
    constraints = [logits.mean() == 0., logits >= -LOGIT_MARGIN, logits <= LOGIT_MARGIN]
    bc_problem = cp.Problem(cp.Minimize(wloss), constraints=constraints)

    result = bc_problem.solve(solver='MOSEK', verbose=verbose)
   
    # ====== Evaluation =====#
    # logits_feature = feature_space @ W_pi.value * tau
    # policy_s_a = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)    
    # _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
    logits_feature = feature_space @ W_pi * tau
    log_sum_exp_logits_feature = cp.log_sum_exp(logits_feature, axis=1, keepdims=True)
    policy_s_a = cp.exp(logits_feature - log_sum_exp_logits_feature).value
    # policy_s_a_pre = np.exp(logits_feature) / np.exp(logits_feature).sum(axis=1, keepdims=True)
    _, return_list, target_mse_list, target_loss01_list = evaluate_policy(seed, mdp, policy_s_a, pi_expert)
    
    w_s = (feature_space @ W_nu.value).reshape(-1)

    # print(f'-- (Post, {alpha}) Return: {np.mean(return_list)} +- {np.std(return_list)/np.sqrt(len(return_list))}')
    # print(f'-- (Post, {alpha}) Target MSE: {np.mean(target_mse_list)} +- {np.std(target_mse_list)/np.sqrt(len(target_mse_list))}')
    
    stats= {
        'ret_mean': np.mean(return_list),
        'ret_stderr': np.std(return_list)/np.sqrt(len(return_list)),
        'tmse_mean': np.mean(target_mse_list),
        'tmse_stderr': np.std(target_mse_list)/np.sqrt(len(target_mse_list)),
        't01_mean': np.mean(target_loss01_list),
        't01_stderr': np.std(target_loss01_list)/np.sqrt(len(target_loss01_list)),
        'cvar25': np.sort(return_list)[:int(0.25*len(return_list))].mean()
    }
    
    return policy_s_a, stats, w_s
