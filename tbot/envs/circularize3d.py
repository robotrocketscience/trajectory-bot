#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circularize-3D: single-body circularization with FULL quaternion attitude control.

Same maneuver as the 2-D milestone (circularize an elliptical orbit at its
apoapsis radius, minimum Δv) but now the orbit is in an arbitrary plane and the
agent must *fly the spacecraft*: it commands body torque to slew the vehicle
(quaternion kinematics + Euler's equations) so the body thrust axis points
prograde, then throttles the main engine. Action = ``[τx, τy, τz, throttle]``.

This is the first 3-D milestone and the attitude-control test. It re-validates
circularization once the vehicle also has to point itself — the step up from the
2-D env where thrust was vectored directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from .. import quaternion as quat
from ..dynamics3d import Spacecraft, rk4_step
from ..orbital import MU_EARTH, R_EARTH, speed_circular
from ..orbital3d import Elements3D, orbital_elements3d

Obs = npt.NDArray[np.float32]
Act = npt.NDArray[np.float32]
Vec = npt.NDArray[np.float64]


@dataclass
class Circularize3DConfig:
    mu: float = MU_EARTH
    r_body: float = R_EARTH
    dt: float = 10.0
    decision_repeat: int = 20
    max_steps: int = 120
    dv_budget: float = 2.0
    alt_peri_range: tuple[float, float] = (400.0, 800.0)
    ra_over_rp_range: tuple[float, float] = (1.3, 2.5)
    inc_max: float = np.radians(40.0)          # random orbit-plane tilt range
    e_tol: float = 0.05
    a_tol: float = 0.05
    w_shape: float = 20.0
    w_e: float = 1.0
    w_point: float = 2.0                        # reward pointing thrust prograde
    k_fuel: float = 1.0
    w_time: float = 0.05
    b_success: float = 200.0
    b_crash: float = 100.0
    b_proximity: float = 20.0
    r_escape_factor: float = 5.0
    gamma: float = 0.999
    sc: Spacecraft = field(default_factory=Spacecraft)


class Circularize3DEnv(gym.Env[Obs, Act]):
    """See module docstring."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: Circularize3DConfig | None = None) -> None:
        super().__init__()
        self.cfg: Circularize3DConfig = config or Circularize3DConfig()
        # action: [torque_x, torque_y, torque_z, throttle], each in [-1, 1]
        self.action_space: spaces.Space[Act] = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # observation: 16-D (see _obs)
        high: Obs = np.full(16, 10.0, dtype=np.float32)
        self.observation_space: spaces.Space[Obs] = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self._state: Vec = np.zeros(13, dtype=np.float64)
        self._r_target: float = 0.0
        self._fuel: float = 0.0
        self._dv_used: float = 0.0
        self._steps: int = 0
        self._prev_potential: float = 0.0

    # ---- helpers ---------------------------------------------------------
    def _potential(self, el: Elements3D) -> float:
        a_err = min(abs(el.a - self._r_target) / self._r_target, 5.0)
        return -(a_err + self.cfg.w_e * min(el.e, 2.0))

    def _thrust_dir(self) -> Vec:
        return quat.rotate(self._state[6:10],
                           np.array(self.cfg.sc.thrust_axis, dtype=np.float64))

    def _obs(self, el: Elements3D) -> Obs:
        length = self._r_target
        vscale = speed_circular(self._r_target, self.cfg.mu)
        r = self._state[0:3]
        v = self._state[3:6]
        w = self._state[10:13]
        tdir = self._thrust_dir()
        o: Obs = np.array([
            float(r[0]) / length, float(r[1]) / length, float(r[2]) / length,
            float(v[0]) / vscale, float(v[1]) / vscale, float(v[2]) / vscale,
            el.a / self._r_target - 1.0, el.e, el.r / length - 1.0,
            float(tdir[0]), float(tdir[1]), float(tdir[2]),
            float(w[0]), float(w[1]), float(w[2]),
            self._fuel / self.cfg.dv_budget if self.cfg.dv_budget > 0 else 0.0,
        ], dtype=np.float32)
        return np.clip(o, -10.0, 10.0)

    # ---- gymnasium API ---------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self.cfg
        r_p = cfg.r_body + float(rng.uniform(*cfg.alt_peri_range))
        r_a = r_p * float(rng.uniform(*cfg.ra_over_rp_range))
        a = 0.5 * (r_p + r_a)
        e = (r_a - r_p) / (r_a + r_p)
        p = a * (1 - e * e)
        h = float(np.sqrt(cfg.mu * p))
        nu = float(rng.uniform(0.0, 2 * np.pi))
        r = p / (1 + e * float(np.cos(nu)))
        # planar (perifocal) state, then tilt into a random 3-D plane
        pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0], dtype=np.float64)
        pfv = np.array([(cfg.mu / h) * (-np.sin(nu)),
                        (cfg.mu / h) * (e + np.cos(nu)), 0.0], dtype=np.float64)
        inc = float(rng.uniform(0.0, cfg.inc_max))
        raan = float(rng.uniform(0.0, 2 * np.pi))
        q_inc = quat.multiply(quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), raan),
                              quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), inc))
        r_vec = quat.rotate(q_inc, pf)
        v_vec = quat.rotate(q_inc, pfv)
        # random initial attitude and small tumble
        q0 = quat.normalize(rng.standard_normal(4))
        w0 = rng.uniform(-0.01, 0.01, size=3)
        self._state = np.concatenate([r_vec, v_vec, q0, w0])

        self._r_target = r_a
        self._fuel = cfg.dv_budget
        self._dv_used = 0.0
        self._steps = 0
        el = orbital_elements3d(r_vec, v_vec, cfg.mu)
        self._prev_potential = self._potential(el)
        info: dict[str, Any] = {"r_target": r_a, "elements": el}
        return self._obs(el), info

    def step(self, action: Act) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        cfg = self.cfg
        act: Vec = np.asarray(action, dtype=np.float64)
        torque: Vec = np.clip(act[0:3], -1.0, 1.0) * cfg.sc.max_torque
        throttle = float(np.clip(act[3], 0.0, 1.0))

        dv_decision = 0.0
        crashed = False
        for _ in range(cfg.decision_repeat):
            thr = throttle
            dv_sub = thr * cfg.sc.a_thrust * cfg.dt
            if self._fuel <= 0.0:
                thr = 0.0
                dv_sub = 0.0
            elif dv_sub > self._fuel:
                thr = thr * (self._fuel / dv_sub)
                dv_sub = self._fuel
            self._fuel -= dv_sub
            self._dv_used += dv_sub
            dv_decision += dv_sub
            self._state = rk4_step(self._state, cfg.dt, torque, thr, cfg.sc)
            rnow = float(np.linalg.norm(self._state[0:3]))
            if rnow <= cfg.r_body:
                crashed = True
                break

        self._steps += 1
        el = orbital_elements3d(self._state[0:3], self._state[3:6], cfg.mu)

        potential = self._potential(el)
        shaping = cfg.w_shape * (cfg.gamma * potential - self._prev_potential)
        self._prev_potential = potential
        # pointing reward: cos angle between thrust axis and prograde direction
        v = self._state[3:6]
        vmag = float(np.linalg.norm(v))
        tdir = self._thrust_dir()
        point = float(np.dot(tdir, v) / vmag) if vmag > 0 else 0.0
        reward = shaping + cfg.w_point * point - cfg.k_fuel * dv_decision - cfg.w_time

        terminated = False
        truncated = False
        a_err = abs(el.a - self._r_target) / self._r_target
        success = el.e < cfg.e_tol and a_err < cfg.a_tol
        escaped = (el.a <= 0.0 or el.e >= 1.0
                   or el.r > cfg.r_escape_factor * self._r_target)
        if success:
            reward += cfg.b_success
            terminated = True
        elif crashed or escaped:
            reward -= cfg.b_crash
            terminated = True
        elif self._steps >= cfg.max_steps:
            truncated = True
            reward += cfg.b_proximity * float(np.exp(-(a_err + cfg.w_e * el.e)))

        info: dict[str, Any] = {
            "elements": el, "dv_used": self._dv_used, "dv_step": dv_decision,
            "fuel": self._fuel, "r_target": self._r_target, "success": success,
        }
        return self._obs(el), reward, terminated, truncated, info
