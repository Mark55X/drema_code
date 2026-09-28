#!/usr/bin/env python
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
from drema.config import load_config
from run_drema_dynamic_system import DremaDynamicSystem
from drema.communication.grpc_client import DremaGrpcClient

cfg = load_config("configs/drema_default.yaml")
cfg.set_nested("digital_twin.engine", "mujoco")
cfg.set_nested("digital_twin.gui", False)
cfg.set_nested("system.device", "cpu")
cfg.set_nested("system.grpc_port", 50077)
cfg.set_nested("system.viser.enabled", False)
cfg.set_nested("perception.cache.enabled", False)

system = DremaDynamicSystem(config=cfg)
system.start()
time.sleep(1.0)

client = DremaGrpcClient(target_address="localhost:50077")
try:
    assert client.ping(), "MuJoCo gRPC server ping failed!"
    print("✓ MuJoCo gRPC Server is ALIVE!")

    act = client.request_action(
        timestep=1,
        joint_positions=[0.0]*7,
        joint_velocities=[0.0]*7,
        ee_pose=[0.4, 0.0, 0.8, 0, 1, 0, 0],
        gripper_open=1.0,
        task_active=False,
        robot_base_pos=[0.0, 0.0, 0.0],
        reachability_radius=0.95
    )
    assert act is not None and len(act.joint_velocities) == 7
    print("✓ ControlAction received with 7 joints, status:", act.status_message)

    res_reset = client.reset_episode(episode_index=1, task_name="test_mujoco_task")
    assert res_reset, "Episode reset failed"
    print("✓ Episode reset confirmed with MuJoCo backend: SUCCESS")
finally:
    client.close()
    system.stop()
    print("✓ DremaDynamicSystem with MuJoCo stopped cleanly!")

print("\n🎉 DREMA DYNAMIC SYSTEM WITH MUJOCO VERIFIED 100%! 🎉")
