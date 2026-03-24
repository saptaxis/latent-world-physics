"""Tests for legacy module path compatibility shim."""
import sys
import pytest

from lwp.compat import register_compat_modules


class TestCompat:
    def test_registers_visual_backbones(self):
        """Calling register_compat_modules makes lunar_lander.src.* resolvable."""
        for key in list(sys.modules.keys()):
            if key.startswith("lunar_lander"):
                del sys.modules[key]

        register_compat_modules()

        assert "lunar_lander" in sys.modules
        assert "lunar_lander.src" in sys.modules
        assert "lunar_lander.src.visual_backbones" in sys.modules
        assert "lunar_lander.src.aux_ppo" in sys.modules

    def test_visual_backbones_points_to_lwp(self):
        """The shim maps to the actual lwp modules, not empty stubs."""
        register_compat_modules()
        import lwp.agents.visual_backbones as real_vb

        shimmed = sys.modules["lunar_lander.src.visual_backbones"]
        assert shimmed is real_vb

    def test_aux_ppo_points_to_lwp(self):
        """The aux_ppo shim maps correctly."""
        register_compat_modules()
        import lwp.agents.aux_ppo as real_ap

        shimmed = sys.modules["lunar_lander.src.aux_ppo"]
        assert shimmed is real_ap

    def test_classes_resolvable_via_old_path(self):
        """Can import actual classes through the old path."""
        register_compat_modules()

        from lunar_lander.src.visual_backbones import ImpalaCNN  # type: ignore[import]
        from lwp.agents.visual_backbones import ImpalaCNN as RealImpalaCNN

        assert ImpalaCNN is RealImpalaCNN

    def test_idempotent(self):
        """Calling register twice doesn't break anything."""
        register_compat_modules()
        register_compat_modules()

        assert "lunar_lander.src.visual_backbones" in sys.modules

    def test_setdefault_doesnt_clobber(self):
        """If lunar_lander is already registered, register doesn't overwrite."""
        import types
        fake = types.ModuleType("lunar_lander")
        sys.modules["lunar_lander"] = fake

        register_compat_modules()

        assert sys.modules["lunar_lander"] is fake

        del sys.modules["lunar_lander"]
