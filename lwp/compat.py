"""Shim old lunar_lander.src.* module paths for loading pre-migration checkpoints.

The lwg->lwp repo migration (2026-03-22) renamed modules:
    lunar_lander.src.visual_backbones -> lwp.agents.visual_backbones
    lunar_lander.src.aux_ppo -> lwp.agents.aux_ppo

SB3 model.zip files saved before the migration contain cloudpickle blobs
that reference the old paths. This module registers sys.modules aliases
so pickle can resolve them. Import this module before calling PPO.load()
on old checkpoints:

    import lwp.compat  # noqa: F401 -- registers old module paths
    model = PPO.load("old_model.zip")

Only visual_rl_agents checkpoints are affected (4 old references across
2 modules). All other checkpoint types (encoder .pt, pixel world model .pt,
state-vector RL .zip, world-model-ladder .pt) load without this shim.
"""
import sys
import types

import lwp.agents.visual_backbones as _vb
import lwp.agents.aux_ppo as _ap

# Create the old module hierarchy and point it at the new modules.
_ll = types.ModuleType("lunar_lander")
_ll_src = types.ModuleType("lunar_lander.src")
_ll_src.visual_backbones = _vb  # type: ignore[attr-defined]
_ll_src.aux_ppo = _ap  # type: ignore[attr-defined]
_ll.src = _ll_src  # type: ignore[attr-defined]

# setdefault avoids clobbering if the old package is actually installed.
sys.modules.setdefault("lunar_lander", _ll)
sys.modules.setdefault("lunar_lander.src", _ll_src)
sys.modules.setdefault("lunar_lander.src.visual_backbones", _vb)
sys.modules.setdefault("lunar_lander.src.aux_ppo", _ap)
