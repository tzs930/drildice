import math
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv



class Continuous_MountainCarEnv_v1(Continuous_MountainCarEnv):
    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with continuous actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |

    ### Action Space

    The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
    The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward

    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
    taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
    is added to the negative reward for that timestep.

    ### Starting State

    The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
    The starting velocity of the car is always assigned to 0.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.

    ### Arguments

    ```
    gym.make('MountainCarContinuous-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    * v1: Add goal condition-related rewards
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    # def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
    def __init__(self, **kwargs):
        super(Continuous_MountainCarEnv_v1, self).__init__(**kwargs)

    def step(self, action: np.ndarray):
        state, reward, terminated, info = super(Continuous_MountainCarEnv_v1, self).step(action)
        reward += -0.01 * abs(self.goal_position - state[0])

        return state, reward, terminated, info

        # position = self.state[0]
        # velocity = self.state[1]
        # force = min(max(action[0], self.min_action), self.max_action)

        # velocity += force * self.power - 0.0025 * math.cos(3 * position)
        # if velocity > self.max_speed:
        #     velocity = self.max_speed
        # if velocity < -self.max_speed:
        #     velocity = -self.max_speed
        # position += velocity
        # if position > self.max_position:
        #     position = self.max_position
        # if position < self.min_position:
        #     position = self.min_position
        # if position == self.min_position and velocity < 0:
        #     velocity = 0

        # # Convert a possible numpy bool to a Python bool.
        # terminated = bool(
        #     position >= self.goal_position and velocity >= self.goal_velocity
        # )

        # reward = 0
        # if terminated:
        #     reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1

        # self.state = np.array([position, velocity], dtype=np.float32)

        # if self.render_mode == "human":
        #     self.render()
        

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        init_state = super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        # self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        # if self.render_mode == "human":
        #     self.render()
        return init_state #np.array(self.state, dtype=np.float32), {}