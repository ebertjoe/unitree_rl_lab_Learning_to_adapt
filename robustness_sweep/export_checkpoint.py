"""Export one RSL-RL checkpoint to TorchScript, without touching ``<run>/exported/``.

The robustness sweep rolls out the TorchScript export, not the ``.pt`` checkpoint
(``--checkpoint`` only selects provenance), so evaluating a checkpoint other than
the one already exported requires producing its own export first.

``scripts/rsl_rl/play.py`` can do this, but it always writes to
``<run>/exported/`` and would overwrite the export the earlier sweeps were run
with. This tool writes wherever it is told and leaves ``<run>/exported/`` alone.

The network is sized from the *current* environment, so the observation code in
the working tree must be the code the checkpoint was trained with. The width is
asserted against the checkpoint's own first layer before anything is written.

Example
-------
    python robustness_sweep/export_checkpoint.py --headless \
        --run 2026-08-17_20-51-38 --checkpoint model_14800.pt \
        --out_dir logs/rsl_rl/unitree_go2_locomotion_paper/2026-08-17_20-51-38/exported_it14800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "rsl_rl"))

from robustness_sweep.checkpoints import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    DEFAULT_LOG_ROOT,
    resolve_policy,
)

from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser(description="Export one checkpoint to TorchScript.")
p.add_argument("--task", default="Unitree-Go2-Velocity")
p.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
p.add_argument("--run", required=True, help="Run folder, 'latest', or a path.")
p.add_argument("--checkpoint", required=True, help="'model_14800.pt', an iteration, or a path.")
p.add_argument("--out_dir", required=True, help="Directory to write policy.pt / policy.onnx into.")
p.add_argument("--num_envs", type=int, default=8,
               help="Only used to size the network; keep it small.")
AppLauncher.add_app_launcher_args(p)
args_cli = p.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from robustness_sweep.isaac_backend import make_env_cfg  # noqa: E402


class _ProtocolDefaults:
    """The env-cfg switches ``make_env_cfg`` reads.

    None of them affect the exported weights - the env exists only to tell the
    runner how wide the observation is - but they must all be present.
    """
    obs_noise = False
    push_robot = False
    domain_rand = True
    deterministic_reset = False
    quiet_env = True
    video = False
    video_resolution = (1280, 720)
    video_eye = (2.2, 2.2, 1.1)


def load_agent_cfg(task: str):
    from isaaclab_tasks.utils import load_cfg_from_registry

    return load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")


def main() -> None:
    ref = resolve_policy(
        run=args_cli.run,
        checkpoint=args_cli.checkpoint,
        log_root=args_cli.log_root,
        experiment=args_cli.experiment,
    )
    ckpt = Path(ref.checkpoint)
    print(f"[export] checkpoint: {ckpt}")

    # The checkpoint's own first layer is the authority on the observation width.
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"] if "model_state_dict" in raw else raw
    ckpt_dim = None
    for k, v in state.items():
        if k.startswith("actor.") and getattr(v, "ndim", 0) == 2:
            ckpt_dim = int(v.shape[1])
            break
    print(f"[export] checkpoint expects obs width: {ckpt_dim}")

    env_cfg = make_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device, _ProtocolDefaults)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    built = env.unwrapped.observation_manager.compute()["policy"].shape[-1]
    print(f"[export] environment builds obs width : {built}")
    if ckpt_dim is not None and built != ckpt_dim:
        env.close()
        raise SystemExit(
            f"observation width mismatch: the working tree builds {built}, the checkpoint "
            f"expects {ckpt_dim}.\nRestore observations.py to the state this run was trained "
            f"with before exporting."
        )

    agent_cfg = load_agent_cfg(args_cli.task)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(out), filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(out), filename="policy.onnx")
    env.close()

    # Read the artefact back the way the sweep will read it.
    jit = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
    jit_dim = None
    for _n, t in jit.named_parameters():
        if t.dim() == 2:
            jit_dim = int(t.shape[1])
            break
    print(f"[export] wrote {out / 'policy.pt'}  (obs width {jit_dim})")
    if jit_dim != built:
        raise SystemExit(f"export width {jit_dim} != env width {built}")
    print("[export] ok")


if __name__ == "__main__":
    # simulation_app.close() hard-exits the process, which discards any pending
    # exception and its traceback, so the error is printed before we get there.
    try:
        with torch.inference_mode(False):
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
        raise SystemExit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
