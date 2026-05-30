"""
JSONL trajectory recorder.

Writes one JSONL file per episode to a user-supplied directory. Each
line is a JSON object representing either a C{"reset"} or C{"step"}
event. The viz module (§5.5) and regression-test goldens (§8.3) both
consume this format.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy / non-JSON-friendly values into JSON-safe ones."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    # Unknown type — best-effort string fallback so recording never
    # crashes mid-rollout because of a stray info entry.
    return repr(value)


class RecordTrajectory(gym.Wrapper):
    """Persist one JSONL file per episode under C{output_dir}.

    File naming: C{episode_<00000>.jsonl}, counter incrementing across
    resets within the wrapper's lifetime. Each file contains:

      - One C{"reset"} record at the top with the initial observation
        and reset info.
      - One C{"step"} record per env step with action, observation,
        reward, terminated, truncated, and info.

    @ivar output_dir: Directory where episode files are written.
    @type output_dir: L{pathlib.Path}
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str | os.PathLike,
    ) -> None:
        """
        @param env: The wrapped env.
        @type env: L{gymnasium.Env}
        @param output_dir: Directory to write JSONL files into.
            Created if it does not exist.
        @type output_dir: str or C{os.PathLike}
        """
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_idx: int = -1
        self._step_idx: int = 0
        self._fh = None

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._close_file()
        self._episode_idx += 1
        self._step_idx = 0
        path = self.output_dir / f"episode_{self._episode_idx:05d}.jsonl"
        self._fh = open(path, "w", encoding="utf-8")  # noqa: SIM115
        obs, info = self.env.reset(seed=seed, options=options)
        self._write(
            {
                "event": "reset",
                "episode": self._episode_idx,
                "obs": _jsonable(obs),
                "info": _jsonable(info),
            }
        )
        return obs, info

    def step(self, action: Any):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        self._step_idx += 1
        self._write(
            {
                "event": "step",
                "t": self._step_idx,
                "action": _jsonable(action),
                "obs": _jsonable(obs),
                "reward": _jsonable(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": _jsonable(info),
            }
        )
        if terminated or truncated:
            self._close_file()
        return result

    def close(self) -> None:
        self._close_file()
        super().close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("RecordTrajectory: no active file handle")
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def _close_file(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
