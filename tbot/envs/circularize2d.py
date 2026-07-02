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
)

Obs = npt.NDArray[np.float32]
Act = npt.NDArray[np.float32]
Vec = npt.NDArray[np.float64]


@dataclass
class Circularize2DConfig:
    mu: float = MU_EARTH               # [km^3/s^2]
    r_body: float = R_EARTH            # [km] central body radius (crash floor)
    dt: float = 10.0                   # [s] physics integration substep
    decision_repeat: int = 10          # physics substeps per agent decision (the
    #                                    agent decides every dt*decision_repeat s;
    #                                    keeps the burn from being a needle in a
    #                                    ~2000-step horizon while physics stay fine)
    max_steps: int = 250               # episode cap in DECISIONS (~ a few orbits)
    thrust_acc_max: float = 5e-2       # [km/s^2] near-impulsive thrust for this
    #                                    milestone; low-thrust spiral is a later one
    dv_budget: float = 3.0             # [km/s] usable Δv (fuel), forces efficiency
    alt_peri_range: tuple[float, float] = (400.0, 800.0)   # [km] periapsis altitude
    ra_over_rp_range: tuple[float, float] = (1.3, 2.5)     # apoapsis/periapsis ratio
    e_tol: float = 0.05                # success: eccentricity below this (near-circular)
    a_tol: float = 0.05                # success: |a - r_target| / r_target below this
    w_shape: float = 20.0              # potential-shaping scale
    w_e: float = 1.0                   # eccentricity weight in the potential
    k_fuel: float = 1.0                # penalty per km/s of Δv spent
    w_time: float = 0.05               # per-step living cost so dawdling/hovering
    #                                    near the target is worse than committing
    b_success: float = 200.0           # large: terminating in-tol >> hovering near it
    b_crash: float = 100.0
    b_proximity: float = 20.0          # small graded bonus (bootstrap, not a crutch)
    gamma: float = 0.999               # shaping discount (match training γ)


class Circularize2DEnv(gym.Env[Obs, Act]):
    """See module docstring."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: Circularize2DConfig | None = None) -> None:
        super().__init__()
        self.cfg: Circularize2DConfig = config or Circularize2DConfig()

        # action: velocity-frame throttle [tangential, radial], each in [-1, 1].
        self.action_space: spaces.Space[Act] = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # observation: 8-D normalized state (finite bounds; values are clipped).
        high: Obs = np.full(8, 10.0, dtype=np.float32)
        self.observation_space: spaces.Space[Obs] = spaces.Box(
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
            self._fuel / self.cfg.dv_budget if self.cfg.dv_budget > 0.0 else 0.0,
        ], dtype=np.float32)
        return np.clip(o, -10.0, 10.0)

    # ---- gymnasium API ---------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        mu = self.cfg.mu
        r_p = self.cfg.r_body + float(rng.uniform(*self.cfg.alt_peri_range))
        r_a = r_p * float(rng.uniform(*self.cfg.ra_over_rp_range))
        a = 0.5 * (r_p + r_a)
        e = (r_a - r_p) / (r_a + r_p)
        p = a * (1.0 - e * e)                      # semi-latus rectum
        h = float(np.sqrt(mu * p))                 # specific angular momentum

        # Start at a RANDOM true anomaly on the ellipse (not always periapsis), so
        # the apoapsis burn opportunity comes soon and varies across episodes —
        # far more learnable than always facing a full half-orbit coast first.
        nu = float(rng.uniform(0.0, 2.0 * np.pi))
        r = p / (1.0 + e * float(np.cos(nu)))
        # perifocal position/velocity, then rotate by a random argument of periapsis
        pf_r = (r * float(np.cos(nu)), r * float(np.sin(nu)))
        pf_v = ((mu / h) * (-float(np.sin(nu))), (mu / h) * (e + float(np.cos(nu))))
        w = float(rng.uniform(0.0, 2.0 * np.pi))
        cw, sw = float(np.cos(w)), float(np.sin(w))
        self._state = np.array([
            cw * pf_r[0] - sw * pf_r[1], sw * pf_r[0] + cw * pf_r[1],
            cw * pf_v[0] - sw * pf_v[1], sw * pf_v[0] + cw * pf_v[1],
        ], dtype=np.float64)

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
        # The action is held over `decision_repeat` physics substeps. It is
        # interpreted in the VELOCITY frame: act[0] = tangential (prograde), act[1]
        # = radial (outward) throttle — so "burn prograde" is just act ≈ (+1, 0).
        # The thrust direction is recomputed each substep so it tracks prograde as
        # the craft moves.
        dv_decision = 0.0
        crashed = False
        for _ in range(cfg.decision_repeat):
            x, y, vx, vy = (float(self._state[i]) for i in range(4))
            r_mag = float(np.hypot(x, y))
            v_mag = float(np.hypot(vx, vy))
            t_hat = (vx / v_mag, vy / v_mag)        # prograde
            r_hat = (x / r_mag, y / r_mag)          # radial outward
            thrust_acc: Vec = np.array([
                (act[0] * t_hat[0] + act[1] * r_hat[0]) * cfg.thrust_acc_max,
                (act[0] * t_hat[1] + act[1] * r_hat[1]) * cfg.thrust_acc_max,
            ], dtype=np.float64)
            dv_sub = float(np.hypot(float(thrust_acc[0]), float(thrust_acc[1]))) * cfg.dt
            if self._fuel <= 0.0:                    # fuel gate
                thrust_acc = np.zeros(2, dtype=np.float64)
                dv_sub = 0.0
            elif dv_sub > self._fuel:
                thrust_acc = thrust_acc * (self._fuel / dv_sub)
                dv_sub = self._fuel
            self._fuel -= dv_sub
            self._dv_used += dv_sub
            dv_decision += dv_sub
            self._state = rk4_step(self._state, cfg.dt, thrust_acc, cfg.mu)
            if float(np.hypot(float(self._state[0]), float(self._state[1]))) <= cfg.r_body:
                crashed = True
                break

        self._steps += 1
        el = orbital_elements(self._state, cfg.mu)

        # --- reward: true objective + potential-based shaping (per decision) ---
        potential = self._potential(el)
        shaping = cfg.w_shape * (cfg.gamma * potential - self._prev_potential)
        self._prev_potential = potential
        reward = shaping - cfg.k_fuel * dv_decision - cfg.w_time

        terminated = False
        truncated = False
        a_err = abs(el.a - self._r_target) / self._r_target
        success = el.e < cfg.e_tol and a_err < cfg.a_tol
        if success:
            reward += cfg.b_success
            terminated = True
        elif crashed:                            # hit the planet during the decision
            reward -= cfg.b_crash
            terminated = True
        elif self._steps >= cfg.max_steps:
            truncated = True
            # graded terminal bonus, gentle in the RAW element errors (not scaled by
            # the tight tolerance) so there is a usable gradient from far away.
            reward += cfg.b_proximity * float(np.exp(-(a_err + cfg.w_e * el.e)))

        info: dict[str, Any] = {
            "elements": el,
            "dv_used": self._dv_used,
            "dv_step": dv_decision,
            "fuel": self._fuel,
            "r_target": self._r_target,
            "success": success,
        }
        return self._obs(el), reward, terminated, truncated, info
