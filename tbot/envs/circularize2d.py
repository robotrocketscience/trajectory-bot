#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circularize-2D: single-body, planar, continuous-thrust circularization.

The spacecraft starts at periapsis of an elliptical Earth orbit and must reach a
circular orbit at the apoapsis radius using as little Δv as possible. The
analytically optimal maneuver is a single prograde burn at apoapsis
(``orbital.circularize_apoapsis_dv``) — that is the baseline the agent must match.

This is the first, deliberately minimal milestone: 2-D, one central body, no
ephemeris/network. It exists to get the RL loop — action space, reward,
termination — correct with the fewest confounds, before extending to full 3-D
attitude control and multi-body transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from ..dynamics import rk4_step
from ..orbital import (
    MU_EARTH,
    R_EARTH,
    Elements,
    circularize_apoapsis_dv,
    orbital_elements,
    speed_circular,
    vis_viva,
)

Obs = npt.NDArray[np.float32]
Act = npt.NDArray[np.float32]
Vec = npt.NDArray[np.float64]


@dataclass
class Circularize2DConfig:
    mu: float = MU_EARTH               # [km^3/s^2]
    r_body: float = R_EARTH            # [km] central body radius (crash floor)
    dt: float = 10.0                   # [s] integration/decision step
    max_steps: int = 2000              # episode step cap (~ a few orbits)
    thrust_acc_max: float = 2e-2       # [km/s^2] max thrust acceleration (~2g)
    dv_budget: float = 3.0             # [km/s] usable Δv (fuel), forces efficiency
    alt_peri_range: tuple[float, float] = (400.0, 800.0)   # [km] periapsis altitude
    ra_over_rp_range: tuple[float, float] = (1.3, 2.5)     # apoapsis/periapsis ratio
    e_tol: float = 0.02                # success: eccentricity below this
    a_tol: float = 0.02                # success: |a - r_target| / r_target below this
    w_shape: float = 10.0              # potential-shaping scale
    w_e: float = 1.0                   # eccentricity weight in the potential
    k_fuel: float = 10.0               # penalty per km/s of Δv spent
    b_success: float = 100.0
    b_crash: float = 100.0
    gamma: float = 0.999               # shaping discount (match training γ)


class Circularize2DEnv(gym.Env[Obs, Act]):
    """See module docstring."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: Circularize2DConfig | None = None) -> None:
        super().__init__()
        self.cfg: Circularize2DConfig = config or Circularize2DConfig()

        # action: 2-D thrust direction * magnitude, each component in [-1, 1].
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: 8-D normalized state (finite bounds; values are clipped).
        high: Obs = np.full(8, 10.0, dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self._state: Vec = np.zeros(4, dtype=np.float64)   # [x, y, vx, vy]
        self._r_target: float = 0.0        # target circular radius (= initial apoapsis)
        self._fuel: float = 0.0            # remaining Δv budget [km/s]
        self._dv_used: float = 0.0         # cumulative Δv spent [km/s]
        self._steps: int = 0
        self._prev_potential: float = 0.0

    # ---- helpers ---------------------------------------------------------
    def _potential(self, el: Elements) -> float:
        a_err = abs(el.a - self._r_target) / self._r_target
        return -(a_err + self.cfg.w_e * el.e)

    def _obs(self, el: Elements) -> Obs:
        length = self._r_target
        vscale = speed_circular(self._r_target, self.cfg.mu)
        o: Obs = np.array([
            float(self._state[0]) / length,
            float(self._state[1]) / length,
            float(self._state[2]) / vscale,
            float(self._state[3]) / vscale,
            el.a / self._r_target - 1.0,
            el.e,
            el.r / length - 1.0,
            self._fuel / self.cfg.dv_budget,
        ], dtype=np.float32)
        return np.clip(o, -10.0, 10.0)

    # ---- gymnasium API ---------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        r_p = self.cfg.r_body + float(rng.uniform(*self.cfg.alt_peri_range))
        r_a = r_p * float(rng.uniform(*self.cfg.ra_over_rp_range))
        a = 0.5 * (r_p + r_a)
        v_p = vis_viva(r_p, a, self.cfg.mu)       # periapsis speed (prograde)

        theta = float(rng.uniform(0.0, 2.0 * np.pi))   # random argument of periapsis
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        # position at periapsis along r_hat; velocity prograde along t_hat (CCW)
        self._state = np.array(
            [r_p * ct, r_p * st, v_p * (-st), v_p * ct], dtype=np.float64)

        self._r_target = r_a
        self._fuel = self.cfg.dv_budget
        self._dv_used = 0.0
        self._steps = 0

        el = orbital_elements(self._state, self.cfg.mu)
        self._prev_potential = self._potential(el)
        info: dict[str, Any] = {
            "r_target": r_a,
            "baseline_dv": circularize_apoapsis_dv(r_p, r_a, self.cfg.mu),
            "elements": el,
        }
        return self._obs(el), info

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        cfg = self.cfg
        act: Vec = np.asarray(action, dtype=np.float64)
        norm = float(np.hypot(float(act[0]), float(act[1])))
        if norm > 1.0:
            act = act / norm
        thrust_acc: Vec = act * cfg.thrust_acc_max
        dv_step = float(np.hypot(float(thrust_acc[0]), float(thrust_acc[1]))) * cfg.dt

        # fuel gate: no thrust once the Δv budget is exhausted.
        if self._fuel <= 0.0:
            thrust_acc = np.zeros(2, dtype=np.float64)
            dv_step = 0.0
        elif dv_step > self._fuel:
            thrust_acc = thrust_acc * (self._fuel / dv_step)
            dv_step = self._fuel
        self._fuel -= dv_step
        self._dv_used += dv_step

        self._state = rk4_step(self._state, cfg.dt, thrust_acc, cfg.mu)
        self._steps += 1
        el = orbital_elements(self._state, cfg.mu)

        # --- reward: true objective + potential-based shaping ---
        potential = self._potential(el)
        shaping = cfg.w_shape * (cfg.gamma * potential - self._prev_potential)
        self._prev_potential = potential
        reward = shaping - cfg.k_fuel * dv_step

        terminated = False
        truncated = False
        success = (el.e < cfg.e_tol
                   and abs(el.a - self._r_target) / self._r_target < cfg.a_tol)
        if success:
            reward += cfg.b_success
            terminated = True
        elif el.r <= cfg.r_body:                 # crashed into the planet
            reward -= cfg.b_crash
            terminated = True
        elif self._steps >= cfg.max_steps:
            truncated = True

        info: dict[str, Any] = {
            "elements": el,
            "dv_used": self._dv_used,
            "dv_step": dv_step,
            "fuel": self._fuel,
            "r_target": self._r_target,
            "success": success,
        }
        return self._obs(el), reward, terminated, truncated, info
