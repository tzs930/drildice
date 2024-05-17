from gym.envs.registration import register

register(
    id='MountainCarContinuous-v1',
    entry_point='envs.mountaincarcont:Continuous_MountainCarEnv_v1'
)