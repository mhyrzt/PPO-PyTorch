import gymnasium as gym


def env_properties(env: gym.Env):
    state_dim = env.observation_space.shape[0]
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    action_dim = env.action_space.shape[0] if is_continuous else env.action_space.n
    return is_continuous, state_dim, action_dim
