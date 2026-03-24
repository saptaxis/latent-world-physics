"""Tests for legacy module path compatibility shim."""
import sys
import pytest


class TestCompat:
    def test_registers_visual_backbones(self):
        """Importing compat makes lunar_lander.src.visual_backbones resolvable."""
        for key in list(sys.modules.keys()):
            if key.startswith("lunar_lander"):
                del sys.modules[key]

        import lwp.compat  # noqa: F401

        assert "lunar_lander" in sys.modules
        assert "lunar_lander.src" in sys.modules
        assert "lunar_lander.src.visual_backbones" in sys.modules
        assert "lunar_lander.src.aux_ppo" in sys.modules

    def test_visual_backbones_points_to_lwp(self):
        """The shim maps to the actual lwp modules, not empty stubs."""
        import lwp.compat  # noqa: F401
        import lwp.agents.visual_backbones as real_vb

        shimmed = sys.modules["lunar_lander.src.visual_backbones"]
        assert shimmed is real_vb

    def test_aux_ppo_points_to_lwp(self):
        """The aux_ppo shim maps correctly."""
        import lwp.compat  # noqa: F401
        import lwp.agents.aux_ppo as real_ap

        shimmed = sys.modules["lunar_lander.src.aux_ppo"]
        assert shimmed is real_ap

    def test_classes_resolvable_via_old_path(self):
        """Can import actual classes through the old path."""
        import lwp.compat  # noqa: F401

        from lunar_lander.src.visual_backbones import ImpalaCNN  # type: ignore[import]
        from lwp.agents.visual_backbones import ImpalaCNN as RealImpalaCNN

        assert ImpalaCNN is RealImpalaCNN

    def test_idempotent(self):
        """Importing compat twice doesn't break anything."""
        import lwp.compat  # noqa: F401
        import lwp.compat  # noqa: F401

        assert "lunar_lander.src.visual_backbones" in sys.modules

    def test_setdefault_doesnt_clobber(self):
        """If lunar_lander is already registered, compat doesn't overwrite."""
        import types
        fake = types.ModuleType("lunar_lander")
        sys.modules["lunar_lander"] = fake

        if "lwp.compat" in sys.modules:
            del sys.modules["lwp.compat"]
        import lwp.compat  # noqa: F401

        assert sys.modules["lunar_lander"] is fake

        del sys.modules["lunar_lander"]
