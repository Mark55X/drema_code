import ctypes

import numpy as np
import pytest

from panda_global_ik import FRANKA_HAND_TCP, PandaIK, Q_MAX, Q_MIN

RNG = np.random.default_rng(0)


def random_q(n, pad=0.02):
    return RNG.uniform(Q_MIN + pad, Q_MAX - pad, size=(n, 7))


def test_forward_kinematics_matches_geofik():
    ik = PandaIK()
    out = np.zeros(16)
    dp = ctypes.POINTER(ctypes.c_double)
    for q in random_q(200):
        qc = np.ascontiguousarray(q)
        ik._lib.geofik_fk(qc.ctypes.data_as(dp), out.ctypes.data_as(dp))
        assert np.allclose(ik.fk(q), out.reshape(4, 4), atol=1e-9)


def test_true_configuration_is_found():
    ik = PandaIK()
    hits = 0
    Q = random_q(300)
    for q in Q:
        assert len(ik.solutions(ik.fk(q))) > 0
        exact = ik.solutions(ik.fk(q), q7=q[6])
        hits += len(exact) > 0 and np.min(np.linalg.norm(exact - q, axis=1)) < 1e-4
    assert hits / len(Q) > 0.95


def test_every_solution_reaches_the_target():
    ik = PandaIK()
    for q in random_q(100):
        T = ik.fk(q)
        for s in ik.solutions(T):
            assert np.linalg.norm(ik.fk(s)[:3, 3] - T[:3, 3]) < 1e-3
            assert np.all(s >= Q_MIN) and np.all(s <= Q_MAX)


def test_out_of_reach_gives_nothing():
    ik = PandaIK()
    for _ in range(50):
        d = RNG.normal(size=3)
        T = np.eye(4)
        T[:3, 3] = d / np.linalg.norm(d) * 1.5
        assert len(ik.solutions(T)) == 0
        assert ik.ik(T, np.zeros(7)) is None


def test_custom_tcp_base_and_joint_offsets():
    tcp = np.eye(4)
    tcp[:3, 3] = [0.002, -0.001, 0.12]
    base = np.eye(4)
    base[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    base[:3, 3] = [-0.27, 0.01, 0.75]
    ik = PandaIK(tcp_offset=tcp, base=base, joint_offsets=[0, 0, 0, 0.0683, 0, 0, 0])
    hits = 0
    Q = random_q(100, pad=0.1)
    for q in Q:
        assert len(ik.solutions(ik.fk(q))) > 0
        exact = ik.solutions(ik.fk(q), q7=q[6])
        hits += len(exact) > 0 and np.min(np.linalg.norm(exact - q, axis=1)) < 1e-4
    assert hits / len(Q) > 0.95


def test_best_prefers_margin_then_closeness():
    ik = PandaIK()
    q = random_q(1, pad=0.3)[0]
    sols = ik.solutions(ik.fk(q))
    pick = ik.best(sols, q, min_margin=0.2)
    assert ik.margin(pick) >= 0.2 or all(ik.margin(s) < 0.2 for s in sols)
    good = [s for s in sols if ik.margin(s) >= 0.2] or list(sols)
    assert np.linalg.norm(pick - q) <= min(np.linalg.norm(s - q) for s in good) + 1e-12


def test_default_tcp_is_the_franka_hand():
    assert np.allclose(PandaIK().tcp, FRANKA_HAND_TCP)


@pytest.mark.parametrize('q7_samples', [9, 33])
def test_every_reachable_pose_is_solved(q7_samples):
    ik = PandaIK(q7_samples=q7_samples)
    Q = random_q(400)
    assert all(len(ik.solutions(ik.fk(q))) > 0 for q in Q)
