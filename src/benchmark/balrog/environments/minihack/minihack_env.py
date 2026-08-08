from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import gym
import gym.spaces
import gymnasium.spaces as _gymnasium_spaces
import minihack  # NOQA: F401

from benchmark.balrog.environments.nle import AutoMore, NLELanguageWrapper
from benchmark.balrog.environments.nle.nle_env import _GymnasiumToGymV0
from benchmark.balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

# ---------------------------------------------------------------------------
# Compatibility patch: nle 1.2.0 returns gymnasium.spaces.Box for its
# observation spaces, but gym 0.23.0's Dict.__init__ asserts
# isinstance(value, gym.Space).  Relax the check to accept both.
# ---------------------------------------------------------------------------
_dict_orig_init = gym.spaces.Dict.__init__


def _dict_compat_init(self, spaces=None, seed=None, **kwargs):
    assert (spaces is None) or (not kwargs), \
        "Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)"
    if spaces is None:
        spaces = kwargs
    if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
        try:
            spaces = OrderedDict(sorted(spaces.items()))
        except TypeError:
            spaces = OrderedDict(spaces.items())
    if isinstance(spaces, Sequence):
        spaces = OrderedDict(spaces)
    assert isinstance(spaces, OrderedDict), "spaces must be a dictionary"
    self.spaces = spaces
    for space in spaces.values():
        assert isinstance(space, (gym.spaces.Space, _gymnasium_spaces.Space)), \
            "Values of the dict should be instances of gym.Space"
    gym.spaces.Space.__init__(self, None, None, seed)


gym.spaces.Dict.__init__ = _dict_compat_init
# ---------------------------------------------------------------------------
# Compatibility patch: minihack's BoxOHack passes wizkit_items=None to
# super().reset(), but nle 1.2.0's NLE.reset() doesn't accept that kwarg.
# ---------------------------------------------------------------------------
import minihack.base as _minihack_base

_minihack_orig_reset = _minihack_base.MiniHack.reset


def _minihack_compat_reset(self, *args, sample_seed=True, **kwargs):
    if "wizkit_items" in kwargs and kwargs["wizkit_items"] is None:
        kwargs.pop("wizkit_items")
    return _minihack_orig_reset(self, *args, sample_seed=sample_seed, **kwargs)


_minihack_base.MiniHack.reset = _minihack_compat_reset
# ---------------------------------------------------------------------------
# Compatibility patch: BoxoHack._is_episode_end accesses self._goal_pos_set
# during NLE.reset() -> _get_end_status() -> _is_episode_end(), but
# _goal_pos_set is only set after super().reset() returns in BoxoHack.reset().
# Guard against the missing attribute by returning RUNNING until initialized.
# ---------------------------------------------------------------------------
from minihack.envs.boxohack import BoxoHack as _BoxoHack

_boxohack_orig_is_episode_end = _BoxoHack._is_episode_end


def _boxohack_safe_is_episode_end(self, observation):
    if not hasattr(self, "_goal_pos_set"):
        return self.StepStatus.RUNNING
    return _boxohack_orig_is_episode_end(self, observation)


_BoxoHack._is_episode_end = _boxohack_safe_is_episode_end
# ---------------------------------------------------------------------------

MINIHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "MiniHack":
        MINIHACK_ENVS.append(id)


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
    minihack_kwargs = dict(config.envs.minihack_kwargs)
    skip_more = minihack_kwargs.pop("skip_more", False)
    vlm = False
    env = gym.make(
        task,
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
        **minihack_kwargs,
    )
    env = _GymnasiumToGymV0(env)
    if skip_more:
        env = AutoMore(env)
    include_ascii_map = getattr(config.envs, "include_ascii_map", False)
    env = NLELanguageWrapper(env, vlm=vlm, include_ascii_map=include_ascii_map)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
