"""Robustness sweep of a Go2 locomotion policy in MuJoCo (thesis §4.1, Table 4.1).

Sim-to-sim counterpart of ``run_robustness_sweep_isaac.py``: same grid, same
episode length, same settle windows, same termination criteria and the same CSV
schema, so the two ``episodes.csv`` files can be concatenated and compared.

The policy is taken from the TorchScript export of the run
(``<run>/exported/policy.pt``), which already contains the observation
normaliser. If a run has no export yet, produce one with

    python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity --checkpoint <ckpt> --headless

Requirements: ``mujoco`` (``pip install mujoco imageio imageio-ffmpeg``) and a Go2
MJCF scene, e.g. ``unitree_mujoco/unitree_robots/go2/scene.xml``.

Examples
--------
    python run_robustness_sweep_mujoco.py --mjcf ~/Github/unitree_mujoco/unitree_robots/go2/scene.xml
    python run_robustness_sweep_mujoco.py --mjcf .../scene.xml --video --seeds 1 --gaits trot,stand
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_sweep import grid as G  # noqa: E402
from robustness_sweep.checkpoints import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_LOG_ROOT,
    resolve_policy,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MuJoCo robustness sweep (thesis section 4.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mjcf", required=True, help="Path to the Go2 MJCF scene (with a floor).")

    pol = p.add_argument_group("policy")
    pol.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    pol.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    pol.add_argument("--run", default="latest", help="Run folder, 'latest', or a path.")
    pol.add_argument("--checkpoint", default="latest",
                     help="Used for provenance and to locate the run; the rollout uses the export.")
    pol.add_argument("--policy_jit", default=None,
                     help="Override the TorchScript policy (default <run>/exported/policy.pt).")

    swp = p.add_argument_group("sweep")
    swp.add_argument("--seeds", type=int, default=G.N_SEEDS)
    swp.add_argument("--base_seed", type=int, default=0)
    swp.add_argument("--gaits", default="all", help="Comma separated gait names/ids, or 'all'.")
    swp.add_argument("--out_dir", default=None,
                     help="Output directory (default <run>/robustness_sweep/mujoco).")
    swp.add_argument("--resume", action="store_true")

    prot = p.add_argument_group("protocol")
    prot.add_argument("--domain_rand", dest="domain_rand", action="store_true", default=False,
                      help="Randomise base mass and foot friction per episode.")
    prot.add_argument("--deterministic_reset", action="store_true", default=False,
                      help="Zero the reset joint-velocity randomisation.")
    prot.add_argument("--torch_threads", type=int, default=1,
                      help="Torch CPU threads; 1 is fastest for this small MLP.")
    prot.add_argument("--joint_friction", type=float, default=0.01,
                      help="Joint friction loss; matches UNITREE_GO2_CFG.")
    prot.add_argument("--joint_damping", type=float, default=0.0,
                      help="Simulated joint damping; 0 because the PD is explicit, as in Isaac.")
    prot.add_argument("--joint_armature", type=float, default=0.0,
                      help="Rotor armature; 0 because Go2HV configures none.")
    prot.add_argument("--ground_friction", type=float, default=1.0,
                      help="Floor friction; matches the Isaac terrain material.")
    prot.add_argument("--keep_scene_obstacles", action="store_true", default=False,
                      help="Keep non-floor world geoms; by default the scene is flattened.")

    vid = p.add_argument_group("video")
    vid.add_argument("--video", action="store_true", default=False)
    vid.add_argument("--video_every", type=int, default=4,
                     help="Capture a frame every N control steps (4 -> 25 fps real time).")
    vid.add_argument("--video_fps", type=int, default=25)
    vid.add_argument("--video_per_gait", type=int, default=8,
                     help="Episodes filmed per gait (first seed only).")
    vid.add_argument("--video_resolution", type=int, nargs=2, default=(1280, 720))
    vid.add_argument("--video_distance", type=float, default=2.5)
    vid.add_argument("--video_azimuth", type=float, default=130.0)
    vid.add_argument("--video_elevation", type=float, default=-20.0)
    vid.add_argument("--mujoco_gl", default="egl",
                     help="MUJOCO_GL backend used for offscreen rendering (egl/osmesa/glfw).")
    return p


def parse_gaits(spec: str) -> list[int] | None:
    if spec.strip().lower() in ("all", ""):
        return None
    out = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token.isdigit() and int(token) in G.GAIT_NAMES:
            out.append(int(token))
        elif token in G.NAME_TO_GAIT_ID:
            out.append(G.NAME_TO_GAIT_ID[token])
        else:
            raise SystemExit(f"unknown gait '{token}'; known: {', '.join(G.NAME_TO_GAIT_ID)}")
    return out


def main() -> None:
    args = build_parser().parse_args()
    args.gaits = parse_gaits(args.gaits)

    if args.video:
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    try:
        import mujoco  # noqa: F401
    except ImportError:
        raise SystemExit(
            "mujoco is not installed in this interpreter.\n"
            "  pip install mujoco imageio imageio-ffmpeg\n"
            "The Isaac Lab sweep (run_robustness_sweep_isaac.py) does not need it."
        )

    import torch

    torch.set_num_threads(max(args.torch_threads, 1))

    from robustness_sweep.mujoco_backend import (
        OBS_DIM,
        Go2MujocoSim,
        MujocoRobustnessSweep,
    )
    from robustness_sweep.summarise import summarise

    policy_ref = resolve_policy(
        run=args.run,
        checkpoint=args.checkpoint,
        log_root=args.log_root,
        experiment=args.experiment,
    )
    jit_path = Path(args.policy_jit) if args.policy_jit else policy_ref.jit_policy
    if jit_path is None or not Path(jit_path).is_file():
        raise SystemExit(
            f"no TorchScript policy for run {policy_ref.run}.\n"
            f"Expected {policy_ref.run_dir / 'exported' / 'policy.pt'}; export it with "
            "scripts/rsl_rl/play.py, or pass --policy_jit."
        )
    print(f"[sweep] policy    : {policy_ref.tag}")
    print(f"[sweep] TorchScript: {jit_path}")

    policy = torch.jit.load(str(jit_path), map_location="cpu")
    policy.eval()

    mjcf = Path(args.mjcf).expanduser()
    if not mjcf.is_file():
        raise SystemExit(f"MJCF not found: {mjcf}")
    sim = Go2MujocoSim(
        str(mjcf),
        joint_friction=args.joint_friction,
        joint_damping=args.joint_damping,
        joint_armature=args.joint_armature,
        ground_friction=args.ground_friction,
        flatten_scene=not args.keep_scene_obstacles,
    )
    sim.snapshot_nominal()
    print(f"[sweep] model     : {mjcf}")
    print(f"[sweep] feet geoms: {sim.foot_geoms}  base geoms: {sim.base_geoms}")
    if sim.disabled_geoms:
        print(f"[sweep] flattened : disabled {len(sim.disabled_geoms)} non-floor world geom(s)")

    # Fail fast on an observation-layout mismatch rather than 4000 episodes later.
    with torch.inference_mode():
        probe = torch.zeros(1, OBS_DIM)
        try:
            action = policy(probe)
        except Exception as exc:
            raise SystemExit(
                f"the exported policy rejected a {OBS_DIM}-dimensional observation "
                f"({exc}).\nThe observation layout in robustness_sweep.mujoco_backend."
                "build_observation must match mdp.observations.robot_state_s of the "
                "checkpoint being evaluated."
            )
    if action.shape[-1] != 12:
        raise SystemExit(f"expected 12 actions, the policy returned {action.shape[-1]}")
    print(f"[sweep] obs dim   : {OBS_DIM} -> {action.shape[-1]} actions")

    sweep = MujocoRobustnessSweep(args, policy_ref, policy, sim)
    sweep.run()
    summarise(sweep.csv_path, sweep.out_dir)


if __name__ == "__main__":
    main()
