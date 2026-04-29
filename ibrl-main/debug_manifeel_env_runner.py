import argparse
import os
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def _ensure_import_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal ManiFeel env_runner debug harness.")
    parser.add_argument("--manifeel_root", required=True)
    parser.add_argument("--isaacgym_envs_path", required=True)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--step_count", type=int, default=1)
    parser.add_argument("--output_dir", default="debug_env_runner_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _ensure_import_path(args.manifeel_root)
    _ensure_import_path(args.isaacgym_envs_path)

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    config_dir = os.path.join(os.path.abspath(args.manifeel_root), "manifeel", "config")
    overrides = [
        "task=vision_wrist",
        "isaacgym_cfg_name=isaacgym_config_bulb.yaml",
        f"training.seed={args.seed}",
        f"n_obs_steps={args.n_obs_steps}",
        f"n_action_steps={args.n_action_steps}",
        f"task.env_runner.n_test={args.num_envs}",
        "task.env_runner.n_test_vis=1",
        f"task.env_runner.max_steps={args.max_steps}",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="train_diffusion_workspace", overrides=overrides)
        runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=str(Path(args.output_dir).resolve()),
        )

    env = runner.env
    print("runner created")
    print("num_envs:", env.num_envs)
    print("action_space shape:", env.action_space.shape)
    print("observation_space keys:", list(env.observation_space.spaces.keys()))

    obs = env.reset()
    print("reset ok")
    print("obs keys:", list(obs.keys()))
    for key, value in obs.items():
        print(f"{key}: {value.shape}")

    action_shape = (env.num_envs,) + tuple(env.action_space.shape)
    action = np.zeros(action_shape, dtype=np.float32)
    print("dummy action shape:", action.shape)

    for step_idx in range(args.step_count):
        obs, reward, done, info = env.step(action)
        print(f"step {step_idx} ok")
        print("reward shape:", np.asarray(reward).shape)
        print("done shape:", np.asarray(done).shape)
        print("all done:", bool(np.all(done)))

    video_paths = env.render()
    print("render ok")
    print("video paths:", video_paths)


if __name__ == "__main__":
    main()
