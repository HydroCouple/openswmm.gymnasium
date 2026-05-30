"""
Action-mask wrappers for the Dict({design, runtime}) action contract.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces


def _sample_midpoint(space: spaces.Space) -> Any:
    """Return a deterministic mid-range value for C{space}.

    For Box: midpoint of C{low} and C{high}. For Dict: recurse.
    Other space types fall back to C{space.sample()} (non-deterministic).
    """
    if isinstance(space, spaces.Box):
        return ((space.low + space.high) * 0.5).astype(space.dtype)
    if isinstance(space, spaces.Dict):
        return {k: _sample_midpoint(v) for k, v in space.spaces.items()}
    if isinstance(space, spaces.Tuple):
        return tuple(_sample_midpoint(s) for s in space.spaces)
    return space.sample()


class _DictHalfMask(gym.Wrapper):
    """Internal base — exposes one half of C{Dict({"design", "runtime"})}.

    Subclasses set L{_kept_key} and L{_other_key}. The wrapper's
    C{action_space} is the kept half's Dict; on C{step} we splice the
    kept agent action with a filler for the other half.
    """

    _kept_key: str = ""
    _other_key: str = ""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Dict):
            raise TypeError(
                "Mask wrappers require a Dict action space with 'design' and 'runtime' keys"
            )
        if not {self._kept_key, self._other_key}.issubset(env.action_space.spaces):
            raise TypeError(
                f"Action space must contain both '{self._kept_key}' and "
                f"'{self._other_key}'; got {list(env.action_space.spaces)}"
            )
        self.action_space = env.action_space.spaces[self._kept_key]
        self._other_space = env.action_space.spaces[self._other_key]

    def _splice(self, kept_action: Any, other_action: Any) -> dict[str, Any]:
        return {self._kept_key: kept_action, self._other_key: other_action}


class MaskRuntimeAction(_DictHalfMask):
    """Hide runtime — agent acts only on the design portion.

    Each step the runtime portion is filled with the per-Box midpoint
    of the env's runtime subspace (deterministic, identity-like for
    typical settings in [0, 1]).
    """

    _kept_key = "design"
    _other_key = "runtime"

    def step(self, action: Any):
        runtime_filler = _sample_midpoint(self._other_space)
        return self.env.step(self._splice(action, runtime_filler))


class MaskDesignAction(_DictHalfMask):
    """Hide design — agent acts only on the runtime portion.

    The design is sampled once at C{reset()} (from the env's design
    subspace) and frozen for the episode. The chosen design appears in
    C{info["frozen_design_action"]}.
    """

    _kept_key = "runtime"
    _other_key = "design"

    def __init__(self, env: gym.Env, frozen_design: Any | None = None) -> None:
        """
        @param env: The wrapped env (must have C{"design"} subspace).
        @type env: L{gymnasium.Env}
        @param frozen_design: Optional design action to use for every
            episode. If C{None}, a new design is sampled at each reset.
        @type frozen_design: dict or C{None}
        """
        super().__init__(env)
        self._frozen_design = frozen_design
        self._current_design: Any | None = None
        # Use a deterministic RNG for sampling so a wrapped env still
        # behaves reproducibly under seeding.
        self._rng_state: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            # Re-seed the design subspace so design sampling is reproducible.
            self._other_space.seed(seed)
        if self._frozen_design is not None:
            self._current_design = self._frozen_design
        else:
            self._current_design = self._other_space.sample()
        obs, info = self.env.reset(seed=seed, options=options)
        info["frozen_design_action"] = self._current_design
        return obs, info

    def step(self, action: Any):
        if self._current_design is None:
            raise RuntimeError("MaskDesignAction.step() called before reset()")
        return self.env.step(self._splice(action, self._current_design))
