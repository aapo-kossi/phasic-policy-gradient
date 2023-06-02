import numpy as np
import gymnasium as gym
from eldorado import EldoradoVenv

def get_eldorado_venv(*, env_id, num_envs, rendering=False, **env_kwargs):
    if rendering:
        env_kwargs["render_human"] = True

    env = EldoradoVenv(num=num_envs, env_name=env_id, **env_kwargs)

    if rendering:
        env = gym.wrappers.HumanRendering(env, info_key="rgb")
    return env


def get_venv(num_envs, env_name, **env_kwargs):
    venv = get_eldorado_venv(num_envs=num_envs, env_id=env_name, **env_kwargs)

    return venv