#!/usr/bin/env python
import numpy as np
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.controller.franka_kinematics import FrankaKinematics

def test_twin_collision_timings():
    # 1. PyBullet
    pb_twin = PyBulletDigitalTwin(visualize=False, table_z=0.75)
    pb_twin.load_robot(base_position=(0, 0, 0.75))
    kin = FrankaKinematics(base_position=(0, 0, 0.75))
    
    K = 10
    H = 5
    Q = np.zeros((K, H, 7), dtype=np.float32)
    QD = np.zeros((K, H, 7), dtype=np.float32)
    
    coll_p, coll_gvm = pb_twin.compute_trajectory_collision_costs(
        Q=Q, QD=QD, sigma_1=0.03, sigma_2=0.20, kappa=15.0, rho=0.8, kin_helper=kin, dt=0.05
    )
    print("PyBullet coll_p shape:", coll_p.shape)
    print("PyBullet last_collision_timings:", pb_twin.last_collision_timings)
    assert 'backend' in pb_twin.last_collision_timings
    assert pb_twin.last_collision_timings['backend'] == 'PyBullet'
    pb_twin.shutdown()
    
    # 2. MuJoCo
    mj_twin = MuJoCoDigitalTwin(visualize=False, table_z=0.75)
    mj_twin.load_robot(base_position=(0, 0, 0.75))
    coll_p_mj, coll_gvm_mj = mj_twin.compute_trajectory_collision_costs(
        Q=Q, QD=QD, sigma_1=0.03, sigma_2=0.20, kappa=15.0, rho=0.8, kin_helper=kin, dt=0.05
    )
    print("MuJoCo coll_p shape:", coll_p_mj.shape)
    print("MuJoCo last_collision_timings:", mj_twin.last_collision_timings)
    assert 'backend' in mj_twin.last_collision_timings
    assert mj_twin.last_collision_timings['backend'] == 'MuJoCo'
    mj_twin.shutdown()
    
    print("ALL COLLISION TIMING TESTS PASSED!")

if __name__ == '__main__':
    test_twin_collision_timings()
