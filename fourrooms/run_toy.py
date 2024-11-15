import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import os
import pickle

from mdp import MDP, solve_MDP, compute_marginal_distribution, generate_trajectory, construct_l2_feature
from trainer import train_BC_agent, train_DRBC_agent, train_DrilDICE_agent,  train_ADWBC_agent, train_DEMODICE_agent, train_OPTIDICEBC_agent

def subsample_idx_state_dependent(room_infos, n_transitions=1000, room_fraction=0.9, dominant_room=0):
    """
    Subsample the trajectories.
    """
    total_idxs = np.arange(len(room_infos))
    
    dominant_room_idxs = (room_infos == dominant_room)
    nondominant_room_idxs = (room_infos != dominant_room)
    
    num_dominant_room = int(room_fraction * n_transitions)
    num_non_dominant_room = n_transitions - num_dominant_room
        
    subsample_list =  []
    
    subsampled_dominant_room_idxs = np.random.choice(total_idxs[dominant_room_idxs], num_dominant_room, replace=False or num_dominant_room > len(total_idxs[dominant_room_idxs]))
    subsample_list.append(subsampled_dominant_room_idxs)    
    
    subsampled_nondominant_room_idxs = np.random.choice(total_idxs[nondominant_room_idxs], num_non_dominant_room, replace=False or num_non_dominant_room > len(total_idxs[nondominant_room_idxs]))
    subsample_list.append(subsampled_nondominant_room_idxs)
    
    subsampled_idxs = np.sort(np.concatenate(subsample_list))

    return subsampled_idxs

# Define four-rooms MDP
class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

WIDTH, HEIGHT = 11, 11
S = WIDTH * HEIGHT + 1
A = 4  # UP, DOWN, LEFT, RIGHT
gamma = 0.99

wall = {(5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 9), (5, 10), (0, 5), (2, 5), (3, 5), (4, 5), (6, 4), (7, 4), (9, 4), (10, 4)}

state_space = []
pos_to_state = {}
state_to_pos = {}
for y in range(HEIGHT):
    for x in range(WIDTH):
        s = y * WIDTH + x
        # pos_to_state[(x, y, 0)] = s
        # state_to_pos[s] = (x, y, 0)
        # state_space.append(state_to_pos[s])
        pos_to_state[(x, y)] = s
        state_to_pos[s] = (x, y)
        state_space.append(state_to_pos[s])

def move(s, a):
    s_x, s_y = state_to_pos[s]
    
    # regular state
    if a == Action.UP: s_y += 1
    elif a == Action.DOWN: s_y -= 1
    elif a == Action.LEFT: s_x -= 1
    elif a == Action.RIGHT: s_x += 1
    else: raise ValueError(f"a={a}")
    
    
    if (s_x, s_y) in wall: return s
    s_x, s_y = np.clip(s_x, 0, WIDTH - 1), np.clip(s_y, 0, HEIGHT - 1)
    
    return pos_to_state[(s_x, s_y)]


# # Add absorbing state
initial_state_pos = (1, HEIGHT - 2) 
goal_state_pos = (WIDTH - 2, 1)

# Add absorbing state
state_to_pos[WIDTH*HEIGHT] = goal_state_pos
state_space.append(state_to_pos[WIDTH*HEIGHT])
# pos_to_state[(0, 0, 1)] = WIDTH*HEIGHT 
# state_space.append(state_to_pos[WIDTH*HEIGHT])

ood_initial_state_pos_1 = (WIDTH - 2, HEIGHT - 2)
ood_initial_state_pos_2 = (1, 1)

imbalanced_init_states = [pos_to_state[ood_initial_state_pos_1], 
                          pos_to_state[ood_initial_state_pos_2]]

T = np.zeros((S, A, S))
R = np.zeros((S, A))
for s in range(S - 1):
    for a in range(A):
        EPS = 0.1
        # EPS = 0.033
        T[s, a, move(s, 0)] += EPS
        T[s, a, move(s, 1)] += EPS
        T[s, a, move(s, 2)] += EPS
        T[s, a, move(s, 3)] += EPS
        T[s, a, move(s, a)] += 1 - np.sum(T[s, a, :])
        assert np.isclose(np.sum(T[s, a]), 1), f"{T[s, a, :]}"
        
T[S - 1, :, S - 1] = 1
T[pos_to_state[goal_state_pos], :, :] = 0 
T[pos_to_state[goal_state_pos], :, S - 1] = 1  # goal state to absorbing state
R[pos_to_state[goal_state_pos], :] = 1

true_mdp = MDP(S, A, T, R, gamma)
true_mdp.initial_state = pos_to_state[initial_state_pos]

# Compute an optimal policy
_, V_opt, Q_opt = solve_MDP(true_mdp)
pi_opt = np.zeros((S, A))

# Deterministic optimal policy
for s in range(S):
    best_a = np.argmax(Q_opt[s, :])
    # best_a_indices = np.where(np.isclose(Q_opt[s, :], Q_opt[s, best_a]))[0]
    pi_opt[s, :] = 0.
    pi_opt[s, best_a] = 1. #/ len(best_a_indices)

# Prepare for the data-collection policy
pi_dir = np.random.dirichlet(np.ones(A), S)
pi_expert = pi_opt #* 0.5 + pi_dir * 0.5

os.makedirs('fourrooms', exist_ok=True)

def draw_state_marginal(mdp, pi, filepath):
        COLOR_WALL = '#990000'
        COLOR_INITIAL_STATE = '#ff9966'
        COLOR_GOAL_STATE = '#99ff99'
        COLOR_ARROW = '#bd534d'
        
        COLOR_OOD_INITIAL_STATE_1 = '#9999ff'
        COLOR_OOD_INITIAL_STATE_2 = '#ff99ff'

        # Draw background
        fig = plt.figure(figsize=(5, 5))

        ax = fig.add_subplot(111, aspect='equal')
        for x in range(-1, WIDTH + 1):
            ax.add_patch(patches.Rectangle((x, -1), 1, 1, linewidth=0, facecolor=COLOR_WALL))
            ax.add_patch(patches.Rectangle((x, HEIGHT), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        for y in range(-1, HEIGHT + 1):
            ax.add_patch(patches.Rectangle((-1, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
            ax.add_patch(patches.Rectangle((WIDTH, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        for x, y in wall:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))

        cm = plt.get_cmap('Blues') 
        cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        
        pi_max_idx = np.argmax(pi, -1)
        pi_det = np.zeros( (mdp.S, mdp.A) )
        pi_det[np.arange(mdp.S), pi_max_idx ] = 1.
        # pi_det = pi[:, np.argmax(pi, axis=1)]
        pi = pi_det

        d_pi = compute_marginal_distribution(mdp, pi).reshape(mdp.S, mdp.A)
        d_pi_s = np.sum(d_pi, axis=1)
        
        # d_pi_s = visit_counts

        # Draw d_pi
        for s in range(mdp.S - 1):
            x, y = state_to_pos[s]
            # x, y, absorb = state_to_pos[s]
            if (x, y) in wall: continue
            # if absorb == 1: continue 
            
            d_s = d_pi_s[s] / np.max(d_pi_s[:S-1])  # 0 ~ 1
            ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=scalarMap.to_rgba(1 - np.power(1 - d_s, 2))))

        ax.add_patch(patches.Rectangle(initial_state_pos, 1, 1, linewidth=3, edgecolor=COLOR_INITIAL_STATE, fill=False))
        ax.add_patch(patches.Rectangle((WIDTH - 2, 1), 1, 1, linewidth=3, edgecolor=COLOR_GOAL_STATE, fill=False))
        # ax.add_patch(patches.Rectangle(ood_initial_state_pos_1, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_1, fill=False))
        # ax.add_patch(patches.Rectangle(ood_initial_state_pos_2, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_2, fill=False))

        # Draw policy
        for s in range(mdp.S - 1):
            x, y = state_to_pos[s]
            # x, y, absorb = state_to_pos[s]
            if (x, y) in wall or (x, y) == (WIDTH - 2, 1): continue
            # if absorb == 1: continue 
            
            for a in range(mdp.A):
                if pi[s, a] > 1e-10:
                    if a == Action.UP:
                        # ax.arrow(x + 0.5, y + 0.5, 0, +0.45 * pi[s, a], head_width=0.05, head_length=0.05, fc='k', ec='k') alpha=min(pi[s, a] * 2, 1)
                        ax.arrow(x + 0.5, y + 0.5, 0, 0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.DOWN:
                        ax.arrow(x + 0.5, y + 0.5, 0, -0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.LEFT:
                        ax.arrow(x + 0.5, y + 0.5, -0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.RIGHT:
                        ax.arrow(x + 0.5, y + 0.5, 0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        return d_pi_s

def generate_full_state_space_transitions(mdp, pi, num_features_per_axis=5):
    features = []
    actions = []
    next_features = []
    true_states = []
    dones = []
    
    for s in range(mdp.S):
        a = np.random.choice(np.arange(mdp.A), p=pi[s, :])
        s1 = np.random.choice(np.arange(mdp.S), p=mdp.T[s, a, :])
        
        pos = np.array([state_to_pos[s]])
        feature = construct_l2_feature(pos, num_points_per_axis=num_features_per_axis)[0]
        
        pos1 = np.array([state_to_pos[s1]])
        feature1 = construct_l2_feature(pos1, num_points_per_axis=num_features_per_axis)[0]
        
        if pos[0][0] <= 5 and pos[0][1] > 5:
            room_idx = 0
        elif pos[0][0] > 5 and pos[0][1] > 4:
            room_idx = 1
        elif pos[0][0] <= 5 and pos[0][1] <= 5:
            room_idx = 2
        elif pos[0][0] > 5 and pos[0][1] <= 4:
            room_idx = 3
        
        onehot_action = np.zeros(mdp.A, dtype=float)
        onehot_action[a] = 1.
        
        onehot_state = np.zeros(mdp.S + 1, dtype=float)
        onehot_state[s] = 1.
        
        features.append(feature)
        actions.append(onehot_action)
        next_features.append(feature1)
        true_states.append(onehot_state)
        dones.append(float(mdp.absorbing_state == s1))
            
    return features, actions, next_features, true_states, dones

def draw_visitation(mdp, pi, visitation, filepath):
        COLOR_WALL = '#990000'
        COLOR_INITIAL_STATE = '#ff9966'
        COLOR_GOAL_STATE = '#99ff99'
        COLOR_ARROW = '#bd534d'
        
        COLOR_OOD_INITIAL_STATE_1 = '#9999ff'
        COLOR_OOD_INITIAL_STATE_2 = '#ff99ff'

        # Draw background
        fig = plt.figure(figsize=(5, 5))

        ax = fig.add_subplot(111, aspect='equal')
        for x in range(-1, WIDTH + 1):
            ax.add_patch(patches.Rectangle((x, -1), 1, 1, linewidth=0, facecolor=COLOR_WALL))
            ax.add_patch(patches.Rectangle((x, HEIGHT), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        for y in range(-1, HEIGHT + 1):
            ax.add_patch(patches.Rectangle((-1, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
            ax.add_patch(patches.Rectangle((WIDTH, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        for x, y in wall:
            ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))

        cm = plt.get_cmap('Blues') 
        cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        pi_max_idx = np.argmax(pi, -1)
        pi_det = np.zeros( (mdp.S, mdp.A) )
        pi_det[np.arange(mdp.S), pi_max_idx ] = 1.
        # pi_det = pi[:, np.argmax(pi, axis=1)]
        pi = pi_det

        d_pi_s = visitation

        # Draw d_pi
        for s in range(mdp.S - 1):
            x, y = state_to_pos[s]
            # x, y, absorb = state_to_pos[s]
            if (x, y) in wall: continue
            # if absorb == 1: continue 
            
            d_s = d_pi_s[s] / np.max(d_pi_s[:S-1])  # 0 ~ 1
            ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=scalarMap.to_rgba(1 - np.power(1 - d_s, 2))))
            ax.annotate(f'{d_s:.2f}', (x + 0.5, y + 0.5), color='black', fontsize=8, ha='center', va='center')

        ax.add_patch(patches.Rectangle(initial_state_pos, 1, 1, linewidth=3, edgecolor=COLOR_INITIAL_STATE, fill=False))
        ax.add_patch(patches.Rectangle((WIDTH - 2, 1), 1, 1, linewidth=3, edgecolor=COLOR_GOAL_STATE, fill=False))
        # ax.add_patch(patches.Rectangle(ood_initial_state_pos_1, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_1, fill=False))
        # ax.add_patch(patches.Rectangle(ood_initial_state_pos_2, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_2, fill=False))

        # Draw policy
        for s in range(mdp.S - 1):
            x, y = state_to_pos[s]
            # x, y, absorb = state_to_pos[s]
            if (x, y) in wall or (x, y) == (WIDTH - 2, 1): continue
            # if absorb == 1: continue 
            
            for a in range(mdp.A):
                if pi[s, a] > 1e-10:
                    if a == Action.UP:
                        # ax.arrow(x + 0.5, y + 0.5, 0, +0.45 * pi[s, a], head_width=0.05, head_length=0.05, fc='k', ec='k') alpha=min(pi[s, a] * 2, 1)
                        ax.arrow(x + 0.5, y + 0.5, 0, 0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.DOWN:
                        ax.arrow(x + 0.5, y + 0.5, 0, -0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.LEFT:
                        ax.arrow(x + 0.5, y + 0.5, -0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                    elif a == Action.RIGHT:
                        ax.arrow(x + 0.5, y + 0.5, 0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        d_pi = compute_marginal_distribution(mdp, pi).reshape(mdp.S, mdp.A)
        d_pi_s = np.sum(d_pi, axis=1)
        
        return d_pi_s 
        
def draw_weight(mdp, pi, weight, filepath, color='Blues'):
    COLOR_WALL = '#990000'
    COLOR_INITIAL_STATE = '#ff9966'
    COLOR_GOAL_STATE = '#99ff99'
    COLOR_ARROW = '#bd534d'
    
    COLOR_OOD_INITIAL_STATE_1 = '#9999ff'
    COLOR_OOD_INITIAL_STATE_2 = '#ff99ff'

    # Draw background
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, aspect='equal')
    for x in range(-1, WIDTH + 1):
        ax.add_patch(patches.Rectangle((x, -1), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((x, HEIGHT), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for y in range(-1, HEIGHT + 1):
        ax.add_patch(patches.Rectangle((-1, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((WIDTH, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for x, y in wall:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))

    # plot deterministic policy
    # pi_det = pi[:, np.argmax(pi, axis=1)]
    # pi = pi_det
    pi_max_idx = np.argmax(pi, -1)
    pi_det = np.zeros( (mdp.S, mdp.A) )
    pi_det[np.arange(mdp.S), pi_max_idx ] = 1.
    # pi_det = pi[:, np.argmax(pi, axis=1)]
    pi = pi_det
    
    d_pi_s = weight
    # vmax = weight.max()
    
    cm = plt.get_cmap(color) 
    cNorm  = colors.Normalize(vmin=0, vmax=1.)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # d_pi = compute_marginal_distribution(mdp, pi).reshape(mdp.S, mdp.A)
    # d_pi_s = np.sum(d_pi, axis=1)
    
    
    # d_pi_s = visit_counts

    # Draw d_pi
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
        # x, y, absorb = state_to_pos[s]
        if (x, y) in wall: continue
        # if absorb == 1: continue 
        
        d_s = d_pi_s[s] #/ np.max(d_pi_s[:S-1])  # 0 ~ 1
        color_s = d_pi_s[s] / np.max(d_pi_s[:S-1])
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=scalarMap.to_rgba(1 - np.power(1 - color_s, 2))))
        ax.annotate(f'{d_s:.2f}', (x + 0.5, y + 0.5), color='black', fontsize=8, ha='center', va='center')

    ax.add_patch(patches.Rectangle(initial_state_pos, 1, 1, linewidth=3, edgecolor=COLOR_INITIAL_STATE, fill=False))
    ax.add_patch(patches.Rectangle((WIDTH - 2, 1), 1, 1, linewidth=3, edgecolor=COLOR_GOAL_STATE, fill=False))
    # ax.add_patch(patches.Rectangle(ood_initial_state_pos_1, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_1, fill=False))
    # ax.add_patch(patches.Rectangle(ood_initial_state_pos_2, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_2, fill=False))

    # Draw policy
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
        # x, y, absorb = state_to_pos[s]
        if (x, y) in wall or (x, y) == (WIDTH - 2, 1): continue
        # if absorb == 1: continue 
        
        for a in range(mdp.A):
            if pi[s, a] > 1e-10:
                if a == Action.UP:
                    # ax.arrow(x + 0.5, y + 0.5, 0, +0.45 * pi[s, a], head_width=0.05, head_length=0.05, fc='k', ec='k') alpha=min(pi[s, a] * 2, 1)
                    ax.arrow(x + 0.5, y + 0.5, 0, 0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.DOWN:
                    ax.arrow(x + 0.5, y + 0.5, 0, -0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.LEFT:
                    ax.arrow(x + 0.5, y + 0.5, -0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.RIGHT:
                    ax.arrow(x + 0.5, y + 0.5, 0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def draw_expert(mdp, pi, filepath, room_infos=None):
    COLOR_WALL = '#990000'  # red
    COLOR_INITIAL_STATE = '#ff9966'
    COLOR_GOAL_STATE = '#99ff99'
    COLOR_ARROW = '#bd534d'
    
    COLOR_OOD_INITIAL_STATE_1 = '#9999ff'
    COLOR_OOD_INITIAL_STATE_2 = '#ff99ff'
    
    room_colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffccff']

    # Draw background
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, aspect='equal')
    for x in range(-1, WIDTH + 1):
        ax.add_patch(patches.Rectangle((x, -1), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((x, HEIGHT), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for y in range(-1, HEIGHT + 1):
        ax.add_patch(patches.Rectangle((-1, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
        ax.add_patch(patches.Rectangle((WIDTH, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))
    for x, y in wall:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=COLOR_WALL))

    cm = plt.get_cmap('Blues') 
    cNorm  = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # # Draw d_pi
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
    #     # x, y, absorb = state_to_pos[s]
        if (x, y) in wall: continue
        # if absorb == 1: continue 
        
        if x <= 5 and y > 5:
            room_idx = 0
        elif x > 5 and y > 4:
            room_idx = 1
        elif x <= 5 and y <= 5:
            room_idx = 2
        elif x > 5 and y <= 4:
            room_idx = 3
            
        ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=room_colors[room_idx]))
        
    #     # d_s = d_pi_s[s] / np.max(d_pi_s[:S-1])  # 0 ~ 1
    #     # ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=None, facecolor=scalarMap.to_rgba(1 - np.power(1 - d_s, 2))))
    #     # ax.annotate(f'{d_s:.2f}', (x + 0.5, y + 0.5), color='black', fontsize=8, ha='center', va='center')

    ax.add_patch(patches.Rectangle(initial_state_pos, 1, 1, linewidth=3, edgecolor=COLOR_INITIAL_STATE, fill=False))
    ax.add_patch(patches.Rectangle((WIDTH - 2, 1), 1, 1, linewidth=3, edgecolor=COLOR_GOAL_STATE, fill=False))
    # ax.add_patch(patches.Rectangle(ood_initial_state_pos_1, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_1, fill=False))
    # ax.add_patch(patches.Rectangle(ood_initial_state_pos_2, 1, 1, linewidth=3, edgecolor=COLOR_OOD_INITIAL_STATE_2, fill=False))

    # Draw policy
    for s in range(mdp.S - 1):
        x, y = state_to_pos[s]
        # x, y, absorb = state_to_pos[s]
        if (x, y) in wall or (x, y) == (WIDTH - 2, 1): continue
        # if absorb == 1: continue 
        
        for a in range(mdp.A):
            if pi[s, a] > 1e-10:
                if a == Action.UP:
                    # ax.arrow(x + 0.5, y + 0.5, 0, +0.45 * pi[s, a], head_width=0.05, head_length=0.05, fc='k', ec='k') alpha=min(pi[s, a] * 2, 1)
                    ax.arrow(x + 0.5, y + 0.5, 0, 0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.DOWN:
                    ax.arrow(x + 0.5, y + 0.5, 0, -0.225, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.LEFT:
                    ax.arrow(x + 0.5, y + 0.5, -0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])
                elif a == Action.RIGHT:
                    ax.arrow(x + 0.5, y + 0.5, 0.225, 0, head_width=0.35, head_length=0.25, fc=COLOR_ARROW, ec=COLOR_ARROW, alpha=pi[s, a])

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    d_pi = compute_marginal_distribution(mdp, pi).reshape(mdp.S, mdp.A)
    d_pi_s = np.sum(d_pi, axis=1)
    
    return d_pi_s       
    
    
imbalance_ratio_list = [1.0] #, 0.95, 0.9, 0.8]
# num_balanced_inits = [1, 2, 3, 4, 5]
seed_offset = 0
num_seeds = 50

# state_suboptimality_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
state_suboptimality_list = [0]
# alpha_list = [10., 5.,  1.0, 0.5, 0.1, 0.05, 0.01]
room_fraction = 0.4

dominant_rooms = [0, 1, 2, 3]
alpha_list = [100., 50., 20., 10., 5., 2.0, 1.0, 0.5, 0.2, 0.1]
num_train_episodes = 100
num_transitions = 1000
is_plot = False
overwrite = False

if is_plot:
    draw_expert(true_mdp, pi_expert, 'fourrooms/expert.png')

# ret_mean_list = np.zeros( (len(imbalance_ratio_list), num_seeds))
# tmse_mean_list = np.zeros( (len(imbalance_ratio_list), num_seeds))

result_dict = {}
method_list = ['BC', 'DEMODICE', 'ADWBC', 'OPTIDICEBC', 'DRBC', 'DICEBC'] #, 'BC']
# method_list = ['BC', 'DRBC'] #'DICEBC-TV', 'OPTIDICEBC-TV'] #, 'BC']
num_seeds = 100
alpha_list_per_method = {
    'BC':            [0.],
    'DEMODICE':      [0.],
    'ADWBC':         [0.],
    'DRBC':          [0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    'OPTIDICEBC':    [100.,   50.,    20.,    10.,   5.,    2.0,   1.0,  0.5,  0.2,  0.1],
    'DrilDICE':        [100.,   50.,    20.,    10.,   5.,    2.0,   1.0,  0.5,  0.2,  0.1],   
}


result_dict = {}
for method in method_list:
    result_dict[method] = {
        'ret_mean_list': np.zeros( (len(dominant_rooms), num_seeds)),
        'tmse_mean_list': np.zeros( (len(dominant_rooms), num_seeds)),
        'cvar25_list': np.zeros( (len(dominant_rooms), num_seeds)),
        't01_mean_list': np.zeros( (len(dominant_rooms), num_seeds)),
    }

pi_behavior = pi_opt
num_points_per_axis = 5
features_global, actions_global, next_features_global, true_states_global, dones_global = generate_full_state_space_transitions(true_mdp, pi_behavior, num_features_per_axis=num_points_per_axis)

base_path = ''
# dict_path = f'{base_path}fourrooms/dominant_room_trajectory_dict_{room_fraction}.pkl'
dict_path = f'fourrooms/dominant_room_trajectory_dict_{room_fraction}.pkl'
    
if os.path.exists(dict_path) and not overwrite:
    with open(dict_path, 'rb') as f:
        trajectory_dict = pickle.load(f)
else:
    trajectory_dict = {}
    for dominant_room in dominant_rooms:
        trajectory_dict[dominant_room] = {}
        # os.makedirs(f'{base_path}dominant-room{dominant_room}', exist_ok=True)
        for seed in range(num_seeds):
            os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}', exist_ok=True)
            
            np.random.seed(seed_offset + seed)
            # * (1 - state_suboptimality) + pi_dir * state_suboptimality
            # Collect trajectories
            trajectory_all = generate_trajectory(seed_offset + seed, true_mdp, pi_behavior, pi_expert, state_to_pos, num_episodes=num_train_episodes, max_timesteps=50, num_features_per_axis=num_points_per_axis)
            
            states = []
            actions = []
            next_states = []
            dones = []
            room_infos = []
            true_states = []
            
            for traj in trajectory_all:
                for _, _, state, action, done, next_state, room_info, true_state in traj:
                    states.append(state)
                    actions.append(action)
                    next_states.append(next_state)
                    dones.append(done)
                    room_infos.append(room_info)
                    true_states.append(true_state)
            
            room_infos = np.array(room_infos)
            subsample_idx = subsample_idx_state_dependent(room_infos, n_transitions=num_transitions, room_fraction=room_fraction, dominant_room=dominant_room)
            # subsample_idx = np.arange(len(states))
            
            features = np.array(states)[subsample_idx]
            actions = np.array(actions)[subsample_idx]
            next_features = np.array(next_states)[subsample_idx]
            dones = np.array(dones)[subsample_idx]
            true_states = np.array(true_states)[subsample_idx]
            
            # Add one perfect global transitions
            features = np.vstack([features, features_global])
            actions = np.vstack([actions, actions_global])
            next_features = np.vstack([next_features, next_features_global])
            true_states = np.vstack([true_states, true_states_global])
            dones = np.concatenate([dones, dones_global])
            
            num_states = len(state_space)
            num_datasets = features.shape[0] 
            num_features = features.shape[-1]
            num_actions = actions.shape[-1]
            
            state_visitation = true_states.sum(axis=0) / true_states.sum()
            if is_plot:
                filepath = f'{base_path}dominant-room{dominant_room}/seed{seed}/data_distribution.png'
                draw_visitation(true_mdp, pi_behavior, state_visitation, filepath)
            
            trajectory_dict[dominant_room][seed] = {
                'features': features,
                'actions': actions,
                'next_features': next_features,
                'dones': dones,
                'd_D': state_visitation
            }
            
    # with open(dict_path, 'wb') as f:
    #     pickle.dump(trajectory_dict, f)

for kk, dominant_room in enumerate(dominant_rooms):
    print('Dominant room:', dominant_room)
    
    bc_weight_list = []
    bc_policy_list = []
    
    for seed in range(num_seeds):
        np.random.seed(seed_offset + seed)
        
        # Collect trajectories
        # trajectory_all = generate_trajectory(seed_offset + seed, true_mdp, pi_behavior, pi_expert, state_to_pos, num_episodes=num_train_episodes, max_timesteps=50, num_features_per_axis=5)
    
        # pi_bc, ret_mean, tmse_mean, ret_stderr, tmse_stderr, cvar25 = \
        pi_bc, stats, W_bc_pi =\
            train_BC_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, verbose=False, num_points_per_axis=num_points_per_axis)

        bc_weight_list.append(W_bc_pi)
        bc_policy_list.append(pi_bc)
            
        if is_plot:
            os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/BC', exist_ok=True)
            plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/BC/behavior.png'
            draw_state_marginal(true_mdp, pi_bc, plot_path)
        
        ret_mean = stats['ret_mean']
        tmse_mean = stats['tmse_mean']
        ret_stderr = stats['ret_stderr']
        tmse_stderr = stats['tmse_stderr']
        cvar25 = stats['cvar25']
        t01_mean = stats['t01_mean']
        t01_stderr = stats['t01_stderr']
        
        with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
            f.write(f'[BC,{seed}]{ret_mean}+-{ret_stderr},{tmse_mean}+-{tmse_stderr},{cvar25},{t01_mean}+-{t01_stderr}\n')
        
        # print(f'[{'BC'}, best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
        
        result_dict['BC']['ret_mean_list'][kk][seed] = ret_mean
        result_dict['BC']['tmse_mean_list'][kk][seed] = tmse_mean
        result_dict['BC']['cvar25_list'][kk][seed] = cvar25
        result_dict['BC']['t01_mean_list'][kk][seed] = t01_mean
        
    ret_mean = result_dict['BC']['ret_mean_list'][kk].mean()
    ret_stderr = result_dict['BC']['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
    tmse_mean = result_dict['BC']['tmse_mean_list'][kk].mean()
    tmse_stderr = result_dict['BC']['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
    cvar25_mean = result_dict['BC']['cvar25_list'][kk].mean()
    cvar25_stderr = result_dict['BC']['cvar25_list'][kk].std() / np.sqrt(num_seeds)
    t01_mean = result_dict['BC']['t01_mean_list'][kk].mean()
    t01_stderr = result_dict['BC']['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
    
    # print(f'[{'BC'}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
    print(f'[BC] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}, best_t01={t01_mean}')
    with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
        f.write(f'[BC,average({num_seeds})]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
        
    for method in method_list:
        if method == 'DrilDICE':
            best_alpha = None
            best_performance = -np.inf
            best_stats = None
            alpha_list = alpha_list_per_method[method]
            for alpha in alpha_list:
                performance_list = []
                cvar25_list = []
                tmse_list = []
                t01_list = []
                
                for seed in range(num_seeds):
                    np.random.seed(seed_offset + seed)
                    
                    W_bc_pi = bc_weight_list[seed]
        
                    pi, stats, w_nu, pi_pre =\
                        train_DrilDICE_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, W_bc_pi=W_bc_pi, state_to_pos=state_to_pos, initial_state_pos=initial_state_pos, gamma=gamma, alpha=alpha, verbose=False, num_points_per_axis=num_points_per_axis)
                    
                    performance_list.append(stats['ret_mean'])
                    cvar25_list.append(stats['cvar25'])
                    tmse_list.append(stats['tmse_mean'])
                    t01_list.append(stats['t01_mean'])
                    
                    ret_mean = stats['ret_mean']
                    ret_stderr = stats['ret_stderr']
                    tmse_mean = stats['tmse_mean']
                    tmse_stderr = stats['tmse_stderr']
                    cvar25 = stats['cvar25']
                    t01_mean = stats['t01_mean']
                    t01_stderr = stats['t01_stderr']
                    
                    with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                        f.write(f'[{method}(alpha={alpha}),{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                    
                    print(f'[{method}(alpha={alpha}), best_performance={ret_mean:.3f}, best_tmse={tmse_mean:.3f}, best_cvar25={cvar25:.3f}, best_t01={t01_mean}')
                    
                    if is_plot:
                        performance = ret_mean
                        performance_pre = result_dict['BC']['ret_mean_list'][kk].mean()
                        
                        os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}', exist_ok=True)
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/post-weight({ret_mean:.2f}).png'
                        draw_state_marginal(true_mdp, pi, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/pre-weight({performance_pre:.2f}).png'
                        draw_state_marginal(true_mdp, pi_pre, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/weight.png'
                        draw_weight(true_mdp, pi_pre, w_nu, plot_path, color='Purples')
                        
                        d_D = trajectory_dict[dominant_room][seed]['d_D']
                        d_W = d_D[:true_mdp.S] * w_nu
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/dW.png'
                        draw_weight(true_mdp, pi_pre, d_W, plot_path, color='Greens')
                            
                
                ret_mean = np.mean(performance_list)
                ret_stderr = np.std(performance_list) / np.sqrt(num_seeds)
                cvar25_mean = np.mean(cvar25_list)
                cvar25_stderr = np.std(cvar25_list) / np.sqrt(num_seeds)
                tmse_mean = np.mean(tmse_list)
                tmse_stderr = np.std(tmse_list) / np.sqrt(num_seeds)
                t01_mean = np.mean(t01_list)
                t01_stderr = np.std(t01_list) / np.sqrt(num_seeds)
                
                with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                    f.write(f'[{method}(alpha={alpha}),summary]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
                if np.mean(performance_list) > best_performance:
                    best_alpha = alpha
                    best_performance = np.mean(performance_list)
                    best_stats = {
                        'ret_mean': performance_list,
                        # 'ret_stderr': np.std(performance_list) / np.sqrt(num_seeds),
                        'tmse_mean': tmse_list,
                        # 'tmse_stderr': np.std(tmse_list) / np.sqrt(num_seeds),
                        'cvar25': cvar25_list,
                        't01_mean': t01_list
                    }
            
            for seed in range(num_seeds):
                result_dict[method]['ret_mean_list'][kk][seed] = best_stats['ret_mean'][seed]
                result_dict[method]['tmse_mean_list'][kk][seed] = best_stats['tmse_mean'][seed]
                result_dict[method]['cvar25_list'][kk][seed] = best_stats['cvar25'][seed]
                result_dict[method]['t01_mean_list'][kk][seed] = best_stats['t01_mean'][seed]
                
            ret_mean = result_dict[method]['ret_mean_list'][kk].mean()
            ret_stderr = result_dict[method]['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
            tmse_mean = result_dict[method]['tmse_mean_list'][kk].mean()
            tmse_stderr = result_dict[method]['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
            cvar25_mean = result_dict[method]['cvar25_list'][kk].mean()
            cvar25_stderr = result_dict[method]['cvar25_list'][kk].std() / np.sqrt(num_seeds)
            t01_mean = result_dict[method]['t01_mean_list'][kk].mean()
            t01_stderr = result_dict[method]['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
            
            print(f'[{method}] best_alpha={best_alpha}, best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}, best_t01={t01_mean}') 
            with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
                f.write(f'[{method},average({num_seeds})]best_alpha={best_alpha},{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
        elif method == 'DRBC':
            best_alpha = None
            best_performance = -np.inf
            best_stats = None
            
            alpha_list = alpha_list_per_method[method]
            for alpha in alpha_list:
                performance_list = []
                cvar25_list = []
                tmse_list = []
                t01_list = []
                
                for seed in range(num_seeds):
                    np.random.seed(seed_offset + seed)
                    
                    W_bc_pi = bc_weight_list[seed]
        
                    pi, stats, w_nu, pi_pre =\
                        train_DRBC_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, W_bc_pi=W_bc_pi, state_to_pos=state_to_pos, initial_state_pos=initial_state_pos, gamma=gamma, alpha=alpha, verbose=False, num_points_per_axis=num_points_per_axis)
                    
                    performance_list.append(stats['ret_mean'])
                    cvar25_list.append(stats['cvar25'])
                    tmse_list.append(stats['tmse_mean'])
                    t01_list.append(stats['t01_mean'])
                    
                    ret_mean = stats['ret_mean']
                    ret_stderr = stats['ret_stderr']
                    tmse_mean = stats['tmse_mean']
                    tmse_stderr = stats['tmse_stderr']
                    cvar25 = stats['cvar25']
                    t01_mean = stats['t01_mean']
                    t01_stderr = stats['t01_stderr']
                    
                    with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                        f.write(f'[{method}(alpha={alpha}),{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                    
                    print(f'[{method}(alpha={alpha}), best_performance={ret_mean:.3f}, best_tmse={tmse_mean:.3f}, best_cvar25={cvar25:.3f}, best_t01={t01_mean}')
                    
                    if is_plot:
                        performance = ret_mean
                        performance_pre = result_dict['BC']['ret_mean_list'][kk].mean()
                        
                        os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}', exist_ok=True)
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/post-weight({ret_mean:.2f}).png'
                        draw_state_marginal(true_mdp, pi, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/pre-weight({performance_pre:.2f}).png'
                        draw_state_marginal(true_mdp, pi_pre, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/weight.png'
                        draw_weight(true_mdp, pi_pre, w_nu, plot_path, color='Purples')
                        
                        d_D = trajectory_dict[dominant_room][seed]['d_D']
                        d_W = d_D[:true_mdp.S] * w_nu
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/dW.png'
                        draw_weight(true_mdp, pi_pre, d_W, plot_path, color='Greens')
                            
                
                ret_mean = np.mean(performance_list)
                ret_stderr = np.std(performance_list) / np.sqrt(num_seeds)
                cvar25_mean = np.mean(cvar25_list)
                cvar25_stderr = np.std(cvar25_list) / np.sqrt(num_seeds)
                tmse_mean = np.mean(tmse_list)
                tmse_stderr = np.std(tmse_list) / np.sqrt(num_seeds)
                t01_mean = np.mean(t01_list)
                t01_stderr = np.std(t01_list) / np.sqrt(num_seeds)
                
                with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                    f.write(f'[{method}(alpha={alpha}),summary]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
                if np.mean(performance_list) > best_performance:
                    best_alpha = alpha
                    best_performance = np.mean(performance_list)
                    best_stats = {
                        'ret_mean': performance_list,
                        # 'ret_stderr': np.std(performance_list) / np.sqrt(num_seeds),
                        'tmse_mean': tmse_list,
                        # 'tmse_stderr': np.std(tmse_list) / np.sqrt(num_seeds),
                        'cvar25': cvar25_list,
                        't01_mean': t01_list
                    }
            
            for seed in range(num_seeds):
                result_dict[method]['ret_mean_list'][kk][seed] = best_stats['ret_mean'][seed]
                result_dict[method]['tmse_mean_list'][kk][seed] = best_stats['tmse_mean'][seed]
                result_dict[method]['cvar25_list'][kk][seed] = best_stats['cvar25'][seed]
                result_dict[method]['t01_mean_list'][kk][seed] = best_stats['t01_mean'][seed]
                
            ret_mean = result_dict[method]['ret_mean_list'][kk].mean()
            ret_stderr = result_dict[method]['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
            tmse_mean = result_dict[method]['tmse_mean_list'][kk].mean()
            tmse_stderr = result_dict[method]['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
            cvar25_mean = result_dict[method]['cvar25_list'][kk].mean()
            cvar25_stderr = result_dict[method]['cvar25_list'][kk].std() / np.sqrt(num_seeds)
            t01_mean = result_dict[method]['t01_mean_list'][kk].mean()
            t01_stderr = result_dict[method]['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
            
            print(f'[{method}] best_alpha={best_alpha}, best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}, best_t01={t01_mean}') 
            with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
                f.write(f'[{method},average({num_seeds})]best_alpha={best_alpha},{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
             
        elif method == 'DEMODICE':
            for seed in range(num_seeds):
                np.random.seed(seed_offset + seed)
                
                # pi_bc, ret_mean, tmse_mean, ret_stderr, tmse_stderr, cvar25 = \
                pi, stats, w_nu =\
                    train_DEMODICE_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, state_to_pos, initial_state_pos=initial_state_pos, verbose=False, num_points_per_axis=num_points_per_axis)
                    
                ret_mean = stats['ret_mean']
                tmse_mean = stats['tmse_mean']
                ret_stderr = stats['ret_stderr']
                tmse_stderr = stats['tmse_stderr']
                cvar25 = stats['cvar25']
                t01_mean = stats['t01_mean']
                t01_stderr = stats['t01_stderr']
                
                # print(f'[{method}, best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
                result_dict[method]['ret_mean_list'][kk][seed] = ret_mean
                result_dict[method]['tmse_mean_list'][kk][seed] = tmse_mean
                result_dict[method]['cvar25_list'][kk][seed] = cvar25
                result_dict[method]['t01_mean_list'][kk][seed] = t01_mean
                
                with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                    f.write(f'[{method},{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25:.3f},{t01_mean}+-{t01_stderr}\n')
                
                if is_plot:
                    # performance = ret_mean
                    performance_pre = result_dict['BC']['ret_mean_list'][kk].mean()
                    pi_pre = bc_policy_list[seed]
                    
                    os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}', exist_ok=True)
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/post-weight({ret_mean:.2f}).png'
                    draw_state_marginal(true_mdp, pi, plot_path)
                    
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/pre-weighting({performance_pre:.2f}).png'
                    draw_state_marginal(true_mdp, pi_pre, plot_path)
                    
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/weight.png'
                    draw_weight(true_mdp, pi_pre, w_nu, plot_path, color='Purples')
                    
                    d_D = trajectory_dict[dominant_room][seed]['d_D']
                    d_W = d_D[:true_mdp.S] * w_nu
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/dW.png'
                    draw_weight(true_mdp, pi_pre, d_W, plot_path, color='Greens')
                
            ret_mean = result_dict[method]['ret_mean_list'][kk].mean()
            ret_stderr = result_dict[method]['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
            tmse_mean = result_dict[method]['tmse_mean_list'][kk].mean()
            tmse_stderr = result_dict[method]['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
            cvar25_mean = result_dict[method]['cvar25_list'][kk].mean()
            cvar25_stderr = result_dict[method]['cvar25_list'][kk].std() / np.sqrt(num_seeds)
            t01_mean = result_dict[method]['t01_mean_list'][kk].mean()
            t01_stderr = result_dict[method]['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
            
            print(f'[{method}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}, best_t01={t01_mean}')
            with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
                f.write(f'[{method},average({num_seeds})]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
        elif method == 'ADWBC':
            for seed in range(num_seeds):
                np.random.seed(seed_offset + seed)
                
                W_bc_pi = bc_weight_list[seed]
                
                # pi_bc, ret_mean, tmse_mean, ret_stderr, tmse_stderr, cvar25 = \
                pi_bc, stats, w_s =\
                    train_ADWBC_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, W_bc_pi=W_bc_pi, verbose=False, num_points_per_axis=num_points_per_axis)
                    
                ret_mean = stats['ret_mean']
                tmse_mean = stats['tmse_mean']
                ret_stderr = stats['ret_stderr']
                tmse_stderr = stats['tmse_stderr']
                cvar25 = stats['cvar25']
                t01_mean = stats['t01_mean']
                t01_stderr = stats['t01_stderr']
                
                # print(f'[{method}, best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
                
                result_dict[method]['ret_mean_list'][kk][seed] = ret_mean
                result_dict[method]['tmse_mean_list'][kk][seed] = tmse_mean
                result_dict[method]['cvar25_list'][kk][seed] = cvar25
                result_dict[method]['t01_mean_list'][kk][seed] = t01_mean
                
                with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                    f.write(f'[{method},{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25:.3f},{t01_mean}+-{t01_stderr}\n')
                    
                if is_plot:
                    pi_pre = bc_policy_list[seed]
                    performance_pre = result_dict['BC']['ret_mean_list'][kk].mean()
                    
                    os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}', exist_ok=True)
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/post-weight({ret_mean:.2f}).png'
                    draw_state_marginal(true_mdp, pi_bc, plot_path)
                    
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/pre-weight({performance_pre:.2f}).png'
                    draw_state_marginal(true_mdp, pi_pre, plot_path)
                    
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/weight.png'
                    draw_weight(true_mdp, pi_pre, w_nu, plot_path, color='Purples')
                    
                    d_D = trajectory_dict[dominant_room][seed]['d_D']
                    d_W = d_D[:true_mdp.S] * w_nu
                    plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/dW.png'
                    draw_weight(true_mdp, pi_pre, d_W, plot_path, color='Greens')
                
            ret_mean = result_dict[method]['ret_mean_list'][kk].mean()
            ret_stderr = result_dict[method]['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
            tmse_mean = result_dict[method]['tmse_mean_list'][kk].mean()
            tmse_stderr = result_dict[method]['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
            cvar25_mean = result_dict[method]['cvar25_list'][kk].mean()
            cvar25_stderr = result_dict[method]['cvar25_list'][kk].std() / np.sqrt(num_seeds)
            t01_mean = result_dict[method]['t01_mean_list'][kk].mean()
            t01_stderr = result_dict[method]['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
            
            # print(f'[{method}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
            print(f'[{method}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
            with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
                f.write(f'[{method},average({num_seeds})]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
        elif method == 'OPTIDICEBC':
            best_alpha = None
            best_performance = -np.inf
            best_stats = None
            
            alpha_list = alpha_list_per_method[method]
            for alpha in alpha_list:
                performance_list = []
                cvar25_list = []
                tmse_list = []
                t01_list = []
                
                for seed in range(num_seeds):
                    np.random.seed(seed_offset + seed)
                    
                    W_bc_pi = bc_weight_list[seed]
        
                    pi_bc, stats, w_nu, pi_pre =\
                        train_OPTIDICEBC_agent(true_mdp, seed, trajectory_dict[dominant_room][seed], state_space, pi_expert, W_bc_pi=W_bc_pi, state_to_pos=state_to_pos, initial_state_pos=initial_state_pos, gamma=gamma, alpha=alpha, verbose=False, num_points_per_axis=num_points_per_axis)
                    
                    performance_list.append(stats['ret_mean'])
                    cvar25_list.append(stats['cvar25'])
                    tmse_list.append(stats['tmse_mean'])
                    t01_list.append(stats['t01_mean'])
                    
                    ret_mean = stats['ret_mean']
                    ret_stderr = stats['ret_stderr']
                    tmse_mean = stats['tmse_mean']
                    tmse_stderr = stats['tmse_stderr']
                    cvar25 = stats['cvar25']
                    t01_mean = stats['t01_mean']
                    t01_stderr = stats['t01_stderr']
                    
                    with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                        f.write(f'[{method}(alpha={alpha}),{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                    
                    print(f'[{method}(alpha={alpha}),{seed}]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25:.3f},{t01_mean}+-{t01_stderr}')
                    
                    if is_plot:
                        performance_pre = result_dict['BC']['ret_mean_list'][kk].mean()
                        
                        os.makedirs(f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}', exist_ok=True)
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/post-weight({ret_mean:.2f}).png'
                        draw_state_marginal(true_mdp, pi, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/pre-weight({performance_pre:.2f}).png'
                        draw_state_marginal(true_mdp, pi_pre, plot_path)
                        
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/weight.png'
                        draw_weight(true_mdp, pi_pre, w_nu, plot_path, color='Purples')
                        
                        d_D = trajectory_dict[dominant_room][seed]['d_D']
                        d_W = d_D[:true_mdp.S] * w_nu
                        plot_path = f'{base_path}dominant-room{dominant_room}/seed{seed}/{method}/alpha{alpha}/dW.png'
                        draw_weight(true_mdp, pi_pre, d_W, plot_path, color='Greens')
                
                ret_mean = np.mean(performance_list)
                ret_stderr = np.std(performance_list) / np.sqrt(num_seeds)
                cvar25_mean = np.mean(cvar25_list)
                cvar25_stderr = np.std(cvar25_list) / np.sqrt(num_seeds)
                tmse_mean = np.mean(tmse_list)
                tmse_stderr = np.std(tmse_list) / np.sqrt(num_seeds)
                t01_mean = np.mean(t01_list)
                t01_stderr = np.std(t01_list) / np.sqrt(num_seeds)
                
                with open(f'{base_path}dominant-room{dominant_room}/logs_indiv.txt', 'a') as f:
                    f.write(f'[{method}(alpha={alpha}),summary]{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
                
                        
                if np.mean(performance_list) > best_performance:
                    best_alpha = alpha
                    best_performance = np.mean(performance_list)
                    best_stats = {
                        'ret_mean': performance_list,
                        # 'ret_stderr': np.std(performance_list) / np.sqrt(num_seeds),
                        'tmse_mean': tmse_list,
                        # 'tmse_stderr': np.std(tmse_list) / np.sqrt(num_seeds),
                        'cvar25': cvar25_list,
                        't01_mean': t01_list
                    }
                        
            tmse_mean = np.mean(best_stats["tmse_mean"])
            cvar25 = np.mean(best_stats["cvar25"])
            
            # print(f'[{method}(alpha={best_alpha}), best_performance={best_performance}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
            
            for seed in range(num_seeds):
                result_dict[method]['ret_mean_list'][kk][seed] = best_stats['ret_mean'][seed]
                result_dict[method]['tmse_mean_list'][kk][seed] = best_stats['tmse_mean'][seed]
                result_dict[method]['cvar25_list'][kk][seed] = best_stats['cvar25'][seed]
                result_dict[method]['t01_mean_list'][kk][seed] = best_stats['t01_mean'][seed]
                
            ret_mean = result_dict[method]['ret_mean_list'][kk].mean()
            ret_stderr = result_dict[method]['ret_mean_list'][kk].std() / np.sqrt(num_seeds)
            tmse_mean = result_dict[method]['tmse_mean_list'][kk].mean()
            tmse_stderr = result_dict[method]['tmse_mean_list'][kk].std() / np.sqrt(num_seeds)
            cvar25_mean = result_dict[method]['cvar25_list'][kk].mean()
            cvar25_stderr = result_dict[method]['cvar25_list'][kk].std() / np.sqrt(num_seeds)
            t01_mean = result_dict[method]['t01_mean_list'][kk].mean()
            t01_stderr = result_dict[method]['t01_mean_list'][kk].std() / np.sqrt(num_seeds)
            
            # print(f'[{method}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25}')
            print(f'[{method}] best_performance={ret_mean}, best_tmse={tmse_mean}, best_cvar25={cvar25},{t01_mean}+-{t01_stderr}')
            with open(f'{base_path}dominant-room{dominant_room}/logs_summary.txt', 'a') as f:
                f.write(f'[{method},average({num_seeds})]best_alpha={best_alpha},{ret_mean:.3f}+-{ret_stderr:.3f},{tmse_mean:.3f}+-{tmse_stderr:.3f},{cvar25_mean:.3f}+-{cvar25_stderr:.3f},{t01_mean}+-{t01_stderr}\n')
 
with open('fourrooms/result_dict.pickle', 'wb') as f:
    pickle.dump(result_dict, f)
