"""Gymnasium environments for TrajectoryBot.

The env is registered lazily so importing ``tbot.envs`` does not require
gymnasium for physics-only use.
"""
# gymnasium's `register` is imperfectly typed upstream (kwargs: dict[Unknown,
# Unknown]); relax that one rule here rather than across the strict library.
# pyright: reportUnknownVariableType=false

from __future__ import annotations


def register_envs() -> None:
    """Register TrajectoryBot envs with Gymnasium (idempotent)."""
    from gymnasium.envs.registration import register, registry

    if "TBot-Circularize2D-v0" not in registry:
        register(
            id="TBot-Circularize2D-v0",
            entry_point="tbot.envs.circularize2d:Circularize2DEnv",
        )


__all__ = ["register_envs"]
