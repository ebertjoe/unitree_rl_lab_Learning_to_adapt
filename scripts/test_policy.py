import torch
import numpy as np

policy = torch.jit.load(
    "/home/yulong/student_projects/jonasebert-thesis/unitree_rl_lab_Learning-to-adapt/logs/rsl_rl/unitree_go2_locomotion_paper/2026-05-10_12-44-29/exported/policy.pt",
    map_location="cpu"
)
policy.eval()

# MuJoCo step 0 obs from latest run
proj_grav      = [-0.034,  0.000, -0.999]
joint_pos      = [-0.095,  0.816, -1.540,  0.096,  0.816, -1.540,
                  -0.156,  0.991, -1.541,  0.155,  0.990, -1.543]
ang_vel        = [ 0.000,  0.001,  0.000]
joint_vel      = [ 0.000,  0.005,  0.001,  0.000,  0.005,  0.001,
                   0.000,  0.003,  0.005,  0.000,  0.003,  0.005]
lin_vel        = [ 0.002,  0.000,  0.000]
vel_cmd        = [ 0.000,  0.000,  0.000]
torques        = [ 0.000] * 12
foot_contact   = [ 1.000,  1.000,  1.000,  1.000]
base_height    = [ 0.311]
desFeetContact = [ 1.000,  1.000,  1.000,  1.000]
refFootZ       = [-0.314, -0.314, -0.326, -0.326]
refFootX       = [ 0.194,  0.194, -0.172, -0.172]
refFootY       = [-0.122,  0.122, -0.122,  0.122]

obs = np.array(
    proj_grav + joint_pos + ang_vel + joint_vel + lin_vel +
    vel_cmd + torques + foot_contact + base_height +
    desFeetContact + refFootZ + refFootX + refFootY,
    dtype=np.float32
)

print(f"Obs shape: {obs.shape}")

with torch.no_grad():
    action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

print(f"Action:  {np.round(action, 3).tolist()}")
print(f"Max abs: {np.abs(action).max():.4f}")

default = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])
targets = default + action * 0.25
print(f"Targets: {np.round(targets, 3).tolist()}")