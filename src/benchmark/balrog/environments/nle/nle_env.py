from typing import Optional

import gymnasium
import nle  # NOQA: F401

from benchmark.balrog.environments.nle import AutoMore, NLELanguageWrapper
from benchmark.balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

NETHACK_ENVS = []
for env_id in gymnasium.envs.registry:
    if "NetHack" in env_id:
        NETHACK_ENVS.append(env_id)


class _GymnasiumToGymV0:
    """Adapts a gymnasium env to old gym (v21) 4-tuple step / single-obs reset.

    nle >= 1.0 registers environments with gymnasium, but the existing wrapper
    chain (AutoMore, NLELanguageWrapper, NLETimeLimit, GymV21CompatibilityV0)
    still uses the old gym 4-tuple API.  This adapter bridges the gap.
    """

    def __init__(self, env):
        self._gymnasium_env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        result = self._gymnasium_env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def step(self, action):
        result = self._gymnasium_env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated or truncated, info
        return result

    def close(self):
        self._gymnasium_env.close()

    @property
    def unwrapped(self):
        return self._gymnasium_env.unwrapped

    def __getattr__(self, name):
        # gymnasium wrappers (e.g. TimeLimit) don't forward NLE-specific
        # attributes like ``actions``, so fall back to the unwrapped env.
        if hasattr(self._gymnasium_env, name):
            return getattr(self._gymnasium_env, name)
        return getattr(self._gymnasium_env.unwrapped, name)


def make_nle_env(env_name, task, config, render_mode: Optional[str] = None):
    nle_kwargs = dict(config.envs.nle_kwargs)
    skip_more = nle_kwargs.pop("skip_more", False)
    vlm = False
    # nle >= 1.0 registers with gymnasium; nle < 1.0 registers with old gym
    try:
        gymnasium_env = gymnasium.make(task, disable_env_checker=True, **nle_kwargs)
        env = _GymnasiumToGymV0(gymnasium_env)
    except (gymnasium.error.NameNotFound, gymnasium.error.DeprecatedEnv):
        env = gym.make(task, **nle_kwargs)
        env = _GymnasiumToGymV0(env)
    if skip_more:
        env = AutoMore(env)
    include_ascii_map = getattr(config.envs, "include_ascii_map", False)
    env = NLELanguageWrapper(env, vlm=vlm, include_ascii_map=include_ascii_map)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
