"""Tests for FrontEnd._apply_pose_jump_guard.

Two properties matter here:

1. With ``pose_jump_guard_relax_when_lost`` unset (the default, and what every
   ECCV config uses) the guard must behave exactly as it did before the
   lost-recovery relaxation was added -- in particular an object's ``lost``
   state must not change a single decision.
2. With the relaxation enabled, an object that has been lost for
   ``max_lost_frames`` frames may re-acquire across an arbitrarily large pose
   jump, but only on strictly stronger support than normal tracking requires.
"""

import numpy as np
import pytest

from point2pose.pipeline.components.front_end import FrontEnd


DEFAULTS = dict(
    pose_jump_guard_enable=True,
    pose_jump_guard_trans_thres=0.08,
    pose_jump_guard_rot_deg_thres=25.0,
    pose_jump_guard_min_inliers=10,
    pose_jump_guard_min_inlier_ratio=0.15,
    pose_jump_guard_single_cluster_min_inliers=12,
    pose_jump_guard_relax_when_lost=False,
    max_lost_frames=5,
    pose_jump_guard_lost_min_inliers=13,
    pose_jump_guard_lost_min_inlier_ratio=0.15,
)


def make_front_end(**overrides):
    """A FrontEnd with only the guard attributes set (__init__ builds models)."""
    fe = FrontEnd.__new__(FrontEnd)
    for key, value in {**DEFAULTS, **overrides}.items():
        setattr(fe, key, value)
    return fe


class FakeObject:
    def __init__(self, lost=False, lost_streak=0):
        self.lost = lost
        self.lost_streak = lost_streak


def pose(trans=0.0, rot_deg=0.0):
    T = np.eye(4)
    a = np.deg2rad(rot_deg)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1]]
    T[0, 3] = trans
    return T


def stats(num_inliers, num_total, num_clusters=3):
    inliers = np.zeros(num_total, dtype=bool)
    inliers[:num_inliers] = True
    return {
        "inliers": inliers,
        "residuals": np.full(num_total, 0.002),
        "clusters": [None] * num_clusters,
    }


def guard(fe, obj, candidate, reg_stats):
    _, rejected, info = fe._apply_pose_jump_guard(candidate, np.eye(4), reg_stats, obj)
    return rejected, info


class TestLegacyBehavior:
    """pose_jump_guard_relax_when_lost unset -> pre-existing ECCV behavior."""

    def setup_method(self):
        self.fe = make_front_end()

    def test_small_jump_is_accepted_even_on_weak_support(self):
        assert guard(self.fe, FakeObject(), pose(0.01, 2), stats(3, 40))[0] is False

    def test_big_jump_with_weak_support_is_rejected(self):
        assert guard(self.fe, FakeObject(), pose(0.5, 40), stats(3, 40))[0] is True

    def test_big_jump_with_strong_support_is_accepted(self):
        assert guard(self.fe, FakeObject(), pose(0.5, 40), stats(30, 40))[0] is False

    def test_big_jump_with_single_weak_cluster_is_rejected(self):
        reg = stats(11, 40, num_clusters=1)
        assert guard(self.fe, FakeObject(), pose(0.5, 40), reg)[0] is True

    @pytest.mark.parametrize(
        "obj", [FakeObject(), FakeObject(lost=True, lost_streak=99)]
    )
    def test_lost_state_changes_no_decision(self, obj):
        """The property that keeps the ECCV numbers reproducible."""
        for reg in (stats(3, 40), stats(30, 40), stats(11, 40, num_clusters=1)):
            live = guard(make_front_end(), FakeObject(), pose(0.5, 40), reg)[0]
            assert guard(self.fe, obj, pose(0.5, 40), reg)[0] == live

    def test_never_reports_relaxation(self):
        obj = FakeObject(lost=True, lost_streak=99)
        assert guard(self.fe, obj, pose(0.5, 40), stats(30, 40))[1][
            "relaxed_when_lost"
        ] is False


class TestRelaxWhenLost:
    """pose_jump_guard_relax_when_lost enabled -> re-acquisition can succeed."""

    def setup_method(self):
        self.fe = make_front_end(pose_jump_guard_relax_when_lost=True)

    def test_tracking_object_still_uses_legacy_rule(self):
        obj = FakeObject(lost=False, lost_streak=99)
        assert guard(self.fe, obj, pose(0.5, 40), stats(3, 40))[0] is True

    def test_relaxation_waits_for_max_lost_frames(self):
        obj = FakeObject(lost=True, lost_streak=DEFAULTS["max_lost_frames"] - 1)
        assert guard(self.fe, obj, pose(0.5, 40), stats(30, 40))[1][
            "relaxed_when_lost"
        ] is False

    def test_relaxation_engages_at_max_lost_frames(self):
        obj = FakeObject(lost=True, lost_streak=DEFAULTS["max_lost_frames"])
        assert guard(self.fe, obj, pose(0.5, 40), stats(30, 40))[1][
            "relaxed_when_lost"
        ] is True

    def test_well_supported_reacquisition_breaks_the_deadlock(self):
        """A frozen pose makes any true re-acquisition look like a huge jump."""
        obj = FakeObject(lost=True, lost_streak=9)
        assert guard(self.fe, obj, pose(2.0, 170), stats(30, 40))[0] is False

    def test_relaxed_guard_still_rejects_too_few_inliers(self):
        obj = FakeObject(lost=True, lost_streak=9)
        assert guard(self.fe, obj, pose(2.0, 170), stats(12, 40))[0] is True

    def test_relaxed_guard_still_rejects_low_inlier_ratio(self):
        obj = FakeObject(lost=True, lost_streak=9)
        assert guard(self.fe, obj, pose(2.0, 170), stats(14, 200))[0] is True

    def test_reacquisition_bar_is_stricter_than_tracking(self):
        reg = stats(12, 40)
        assert guard(self.fe, FakeObject(), pose(0.5, 40), reg)[0] is False
        assert guard(self.fe, FakeObject(True, 9), pose(0.5, 40), reg)[0] is True


def test_disabled_guard_never_rejects():
    fe = make_front_end(
        pose_jump_guard_enable=False, pose_jump_guard_relax_when_lost=True
    )
    obj = FakeObject(lost=True, lost_streak=99)
    assert guard(fe, obj, pose(2.0, 170), stats(1, 40))[0] is False
