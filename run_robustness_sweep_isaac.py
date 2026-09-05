"""Robustness sweep of a Go2 locomotion policy in Isaac Lab (thesis §4.1, Table 4.1).

Evaluates a policy over the full velocity grid for every gait, five seeds, and
records per episode: survival and termination reason, mean tracking errors and
achieved values in vx / vy / wz, mean and standard deviation of the base height,
mean roll and pitch derived from projected gravity, the per-foot contact fraction,
the gait contact accuracy against the Raibert schedule, and - for zero-command
episodes - the planar drift at the end of the episode. A single annotated mp4 of
the sweep plus a seekable manifest are written alongside the CSV.

Examples
--------
    # latest checkpoint of the latest run, no video
    python run_robustness_sweep_isaac.py --headless

    # a specific run + checkpoint, with the sweep video
    python run_robustness_sweep_isaac.py --headless --video \
        --run 2026-08-29_10-25-57 --checkpoint model_9999.pt

    # quick shakedown: one seed, two gaits, small grid pass
    python run_robustness_sweep_isaac.py --headless --seeds 1 --gaits trot,stand --num_envs 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "rsl_rl"))

from robustness_sweep import grid as G  # noqa: E402  (no isaac import, safe here)
from robustness_sweep.checkpoints import DEFAULT_EXPERIMENT, DEFAULT_LOG_ROOT, resolve_policy  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Isaac Lab robustness sweep (thesis section 4.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="Unitree-Go2-Velocity", help="Isaac Lab task id.")

    pol = p.add_argument_group("policy")
    pol.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT), help="Root of the rsl_rl logs.")
    pol.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Experiment folder name.")
    pol.add_argument("--run", default="latest", help="Run folder, 'latest', or a path.")
    pol.add_argument("--policy_jit", default=None,
                     help="TorchScript policy to roll out (default: <run>/exported/policy.pt). "
                          "Its input width selects the observation layout.")
    pol.add_argument("--checkpoint", default="latest",
                     help="'latest', 'model_9999.pt', an iteration number, or a path.")

    swp = p.add_argument_group("sweep")
    swp.add_argument("--seeds", type=int, default=G.N_SEEDS, help="Number of seeds (Table 4.1).")
    swp.add_argument("--base_seed", type=int, default=0, help="First seed value.")
    swp.add_argument("--gaits", default="all",
                     help="Comma separated gait names or ids to sweep, or 'all'.")
    swp.add_argument("--num_envs", type=int, default=None,
                     help="Episodes simulated in parallel (default 512, or 32 with --video).")
    swp.add_argument("--out_dir", default=None,
                     help="Output directory (default <run>/robustness_sweep/isaac).")
    swp.add_argument("--resume", action="store_true",
                     help="Append to an existing episodes.csv and skip the rows it already has.")

    prot = p.add_argument_group("protocol")
    prot.add_argument("--domain_rand", dest="domain_rand", action="store_true", default=True,
                      help="Keep the start-up mass/friction randomisation (default).")
    prot.add_argument("--no_domain_rand", dest="domain_rand", action="store_false",
                      help="Evaluate the nominal robot only.")
    prot.add_argument("--push_robot", action="store_true", default=False,
                      help="Keep the interval push disturbance (off by default).")
    prot.add_argument("--obs_noise", action="store_true", default=False,
                      help="Keep the observation corruption (off by default).")
    prot.add_argument("--deterministic_reset", action="store_true", default=False,
                      help="Zero the reset joint-velocity randomisation.")
    prot.add_argument("--quiet_env", dest="quiet_env", action="store_true", default=True,
                      help="Suppress the environment's periodic debug prints (default).")
    prot.add_argument("--verbose_env", dest="quiet_env", action="store_false",
                      help="Let the environment print its debug output.")

    vid = p.add_argument_group("video")
    vid.add_argument("--video", action="store_true", default=False,
                     help="Record one annotated mp4 of the sweep plus a manifest.")
    vid.add_argument("--video_every", type=int, default=4,
                     help="Capture a frame every N control steps (4 -> 25 fps real time).")
    vid.add_argument("--video_fps", type=int, default=25, help="Frame rate of the mp4.")
    vid.add_argument("--video_stride", type=int, default=1,
                     help="Film one episode every N batches.")
    vid.add_argument("--video_resolution", type=int, nargs=2, default=(1280, 720))
    vid.add_argument("--video_eye", type=float, nargs=3, default=(2.2, 2.2, 1.1),
                     help="Camera offset from the filmed robot.")

    AppLauncher.add_app_launcher_args(p)
    return p


def parse_gaits(spec: str) -> list[int] | None:
    if spec.strip().lower() in ("all", ""):
        return None
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            gid = int(token)
            if gid not in G.GAIT_NAMES:
                raise SystemExit(f"unknown gait id {gid}")
        elif token in G.NAME_TO_GAIT_ID:
            gid = G.NAME_TO_GAIT_ID[token]
        else:
            raise SystemExit(
                f"unknown gait '{token}'; known: {', '.join(G.NAME_TO_GAIT_ID)}"
            )
        out.append(gid)
    return out


args_cli = build_parser().parse_args()
args_cli.gaits = parse_gaits(args_cli.gaits)
if args_cli.num_envs is None:
    args_cli.num_envs = 32 if args_cli.video else 512
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Isaac imports (only valid once the app is running)
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from robustness_sweep.isaac_backend import (  # noqa: E402
    IsaacRobustnessSweep,
    check_gait_table,
    make_env_cfg,
)


def load_agent_cfg(task: str, args):
    """Agent config from the registry; ``cli_args`` is only used if its flags are present."""
    from isaaclab_tasks.utils import load_cfg_from_registry

    try:
        import cli_args  # scripts/rsl_rl/cli_args.py

        return cli_args.parse_rsl_rl_cfg(task, args)
    except (ImportError, AttributeError):
        # cli_args expects the train/play flag set, which this parser does not define
        return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")


def main() -> None:
    policy_ref = resolve_policy(
        run=args_cli.run,
        checkpoint=args_cli.checkpoint,
        log_root=args_cli.log_root,
        experiment=args_cli.experiment,
    )
    print(f"[sweep] policy: {policy_ref.checkpoint}")

    check_gait_table(args_cli.task)

    # The exported TorchScript carries its own observation normalizer and its own
    # input width, so it is the authority on which observation layout to build.
    # Loading the raw checkpoint through OnPolicyRunner cannot do this: the runner
    # sizes the network from the *current* env, which fails for older runs.
    jit_path = Path(args_cli.policy_jit) if args_cli.policy_jit else policy_ref.jit_policy
    if jit_path is None or not Path(jit_path).is_file():
        raise SystemExit(
            f"no TorchScript policy found for this run.\n"
            f"Expected {policy_ref.run_dir / 'exported' / 'policy.pt'}."
        )
    policy_jit = torch.jit.load(str(jit_path), map_location="cpu")
    obs_dim = None
    for _n, _t in policy_jit.named_parameters():
        if _t.dim() == 2:
            obs_dim = int(_t.shape[1])
            break
    if obs_dim is None:
        raise SystemExit(f"could not infer the observation width of {jit_path}")
    print(f"[sweep] TorchScript: {jit_path}")
    print(f"[sweep] obs layout : {obs_dim}")
    args_cli.obs_dim = obs_dim  # recorded in config.json for provenance

    env_cfg = make_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device, args_cli)
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    device = env.unwrapped.device
    policy = policy_jit.to(device).eval()
    got = env.unwrapped.observation_manager.compute()["policy"].shape[-1]
    if got != obs_dim:
        raise SystemExit(
            f"observation width mismatch: env builds {got}, policy expects {obs_dim}"
        )
    print("[sweep] policy loaded.")

    sweep = IsaacRobustnessSweep(args_cli, policy_ref, env, policy, device)
    try:
        sweep.run()
    finally:
        env.close()

    from robustness_sweep.summarise import summarise

    summarise(sweep.csv_path, sweep.out_dir)
    print(f"[sweep] done: {sweep.out_dir}")


if __name__ == "__main__":
    try:
        with torch.inference_mode(False):
            main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
