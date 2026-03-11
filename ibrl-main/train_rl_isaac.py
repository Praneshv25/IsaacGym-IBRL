"""
train_rl_isaac.py
=================
Vectorized IBRL training script for the IsaacGym bulb-insertion task.

Key differences from the original ``train_rl.py`` (Robosuite / MuJoCo):

* Uses ``IsaacGymBulbEnv`` – N parallel simulation environments running
  on GPU simultaneously.
* Uses ``VecReplayBuffer`` – N independent ``rela.Episode`` accumulators
  sharing one ``SingleStepTransitionReplay`` for uniform random sampling.
* Training loop increments ``global_step`` by ``num_envs`` every env step,
  so logging / update frequencies are measured in total *transitions*
  (consistent with the single-env script).
* State-only observations (14-D): no image encoder, no BC camera stack.
* Evaluation reuses the training env (IsaacGym only allows one simulator
  per process) by resetting all envs and collecting episodes in eval mode.

Usage
-----
::

    python train_rl_isaac.py \\
        --isaacgym_envs_path ../manifeel-isaacgymenvs-tacsl-manifeel-rl \\
        --num_envs 64 \\
        --save_dir exps/rl_isaac/run1

or with pyrallis config file::

    python train_rl_isaac.py --config_path my_cfg.yaml
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyrallis
import torch
import yaml

import common_utils
from common_utils import ibrl_utils as utils
from env.isaac_gym_wrapper import ACTION_DIM, STATE_DIM, IsaacGymBulbEnv
from evaluate.eval_isaac import run_eval_isaac
from rl.q_agent import QAgent, QAgentConfig
from rl.vec_replay import VecReplayBuffer, add_demos_from_pkl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MainConfig(common_utils.RunConfig):
    """Top-level configuration for vectorized IBRL on the IsaacGym bulb task."""

    seed: int = 1

    # ── environment ──────────────────────────────────────────────────────────
    # Path to the manifeel-isaacgymenvs repo root (contains isaacgymenvs/)
    isaacgym_envs_path: str = "../manifeel-isaacgymenvs-tacsl-manifeel-rl"
    num_envs: int = 64  # parallel envs during training
    sim_device: str = "cuda:0"
    rl_device: str = "cuda:0"
    graphics_device_id: int = -1
    headless: bool = True
    env_reward_scale: float = 1.0

    # ── observations / actions ───────────────────────────────────────────────
    obs_dim: int = STATE_DIM  # 14  (fixed – matches IsaacGymBulbEnv)
    action_dim: int = ACTION_DIM  # 7  (fixed)

    # ── RL agent ─────────────────────────────────────────────────────────────
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    # Exploration noise schedule: linearly decays from stddev_max → stddev_min
    # over stddev_step total transitions.
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500_000

    # ── n-step returns & discount ────────────────────────────────────────────
    nstep: int = 3
    discount: float = 0.99

    # ── replay buffer ────────────────────────────────────────────────────────
    # replay_buffer_size is in *episodes* (how rela counts capacity)
    replay_buffer_size: int = 2_000
    batch_size: int = 256

    # ── training schedule ────────────────────────────────────────────────────
    num_critic_update: int = 1
    # How many total transitions between each agent update.
    # (Effectively: update every update_freq // num_envs env steps)
    update_freq: int = 128
    # Total transitions to train for
    num_train_step: int = 500_000

    # ── warm-up ──────────────────────────────────────────────────────────────
    # Number of env steps (each = num_envs transitions) with random actions
    # before training begins.
    num_warm_up_steps: int = 200

    # ── demo preloading (optional) ───────────────────────────────────────────
    # Path to a MarsLab .pkl transition file.  Leave empty to skip.
    preload_pkl: str = ""

    # ── evaluation ───────────────────────────────────────────────────────────
    num_eval_episodes: int = 50
    # Log / eval every this many total transitions
    log_per_step: int = 10_000

    # ── logging ──────────────────────────────────────────────────────────────
    save_dir: str = "exps/rl_isaac/run1"
    use_wb: int = 0

    def __post_init__(self) -> None:
        # Clamp stddev so min ≤ max
        self.stddev_min = min(self.stddev_max, self.stddev_min)

    @property
    def stddev_schedule(self) -> str:
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """Encapsulates env, agent, replay and the training loop."""

    def __init__(self, cfg: MainConfig) -> None:
        self.cfg = cfg
        self.work_dir = cfg.save_dir
        os.makedirs(self.work_dir, exist_ok=True)

        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        # Dump config to disk so experiments are reproducible
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(common_utils.wrap_ruler("config"))
        with open(cfg.cfg_path) as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path))

        # ── counters ─────────────────────────────────────────────────────
        self.global_step = 0  # total transitions seen
        self.global_episode = 0  # total episodes completed
        self.train_step = 0  # number of gradient update steps

        # ── environment ───────────────────────────────────────────────────
        self._setup_env()

        # ── agent ─────────────────────────────────────────────────────────
        # Force state-based mode: FcActor + MultiFcQ, no image encoder.
        # QAgentConfig.act_method defaults to "rl"; we leave it as-is
        # (no BC policy available for this prototype).
        print(common_utils.wrap_ruler("building QAgent (use_state=True)"))
        self.agent = QAgent(
            use_state=1,
            obs_shape=(cfg.obs_dim,),
            prop_shape=(cfg.obs_dim,),
            action_dim=cfg.action_dim,
            rl_camera="",  # unused when use_state=True
            cfg=cfg.q_agent,
        )
        # Reference agent: always acts with pure RL (used in actor updates)
        self.ref_agent = copy.deepcopy(self.agent)
        self.ref_agent.cfg.act_method = "rl"

        # ── replay buffer ─────────────────────────────────────────────────
        self._setup_replay()

    # ──────────────────────────────────────────────────────────────────────
    # Setup helpers
    # ──────────────────────────────────────────────────────────────────────

    def _setup_env(self) -> None:
        cfg = self.cfg
        print(common_utils.wrap_ruler("building IsaacGymBulbEnv"))
        self.train_env = IsaacGymBulbEnv(
            isaacgym_envs_path=cfg.isaacgym_envs_path,
            num_envs=cfg.num_envs,
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            seed=cfg.seed,
            env_reward_scale=cfg.env_reward_scale,
        )
        print(self.train_env)
        print(f"  observation_shape : {self.train_env.observation_shape}")
        print(f"  action_dim        : {self.train_env.action_dim}")
        print(f"  max_episode_length: {self.train_env.max_episode_length}")

    def _setup_replay(self) -> None:
        cfg = self.cfg
        self.replay = VecReplayBuffer(
            num_envs=cfg.num_envs,
            nstep=cfg.nstep,
            gamma=cfg.discount,
            max_episode_length=self.train_env.max_episode_length,
            replay_size=cfg.replay_buffer_size,
        )

        if cfg.preload_pkl:
            add_demos_from_pkl(
                self.replay,
                cfg.preload_pkl,
                verbose=True,
            )
            print(
                f"After demo preload: replay size = {self.replay.size()} episodes, "
                f"successes = {self.replay.num_success}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Warm-up
    # ──────────────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        """Fill the replay buffer with *num_warm_up_steps* env steps of
        uniformly random actions before training begins.

        Each env step produces ``num_envs`` transitions, so the warm-up
        adds ``num_warm_up_steps * num_envs`` transitions in total.
        """
        print(common_utils.wrap_ruler("warm-up (random actions)"))
        cfg = self.cfg

        obs = self.train_env.reset()
        self.replay.new_episodes(obs)

        for step in range(cfg.num_warm_up_steps):
            # Sample uniformly random actions in [-1, 1]
            actions = torch.zeros(
                cfg.num_envs, cfg.action_dim, device=cfg.rl_device
            ).uniform_(-1.0, 1.0)

            obs, rewards, dones, successes = self.train_env.step(actions)
            self.replay.add_step(obs, actions, rewards, dones, successes)

            n_done = int(dones.sum().item())
            if n_done > 0:
                self.global_episode += n_done

        print(
            f"Warm-up done. "
            f"Replay: {self.replay.size()} episodes, "
            f"{self.replay.num_success} successes, "
            f"{self.replay.num_episode} total episodes added."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def eval(self) -> float:
        """Run evaluation episodes and return the mean success rate.

        The training environment is reused (IsaacGym does not support two
        simultaneous simulators per process).  After eval the env is reset
        again so training can resume cleanly.
        """
        scores = run_eval_isaac(
            env=self.train_env,
            agent=self.agent,
            num_episodes=self.cfg.num_eval_episodes,
            verbose=False,
        )
        return float(np.mean(scores))

    # ──────────────────────────────────────────────────────────────────────
    # RL update
    # ──────────────────────────────────────────────────────────────────────

    def _rl_train(self, stat: common_utils.MultiCounter) -> None:
        """Perform one round of critic (and optionally actor) updates."""
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            batch = self.replay.sample(self.cfg.batch_size, self.cfg.rl_device)
            update_actor = i == self.cfg.num_critic_update - 1
            metrics = self.agent.update(
                batch,
                stddev,
                update_actor,
                bc_batch=None,  # no BC policy in state-only prototype
                ref_agent=self.ref_agent,
            )
            stat.append(metrics)
            stat["data/discount"].append(float(batch.bootstrap.mean().item()))

    # ──────────────────────────────────────────────────────────────────────
    # Logging & saving
    # ──────────────────────────────────────────────────────────────────────

    def _log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ) -> None:
        elapsed = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / max(elapsed, 1e-6))
        stat["other/elapsed_time"].append(elapsed)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["data/replay_episodes"].append(self.replay.size())
        stat["data/replay_successes"].append(self.replay.num_success)

        # ── evaluation ────────────────────────────────────────────────────
        eval_score = self.eval()
        stat["score/eval_success_rate"].append(eval_score)
        print(f"[step {self.global_step:,}] eval success rate: {eval_score:.4f}")

        # ── save checkpoint ───────────────────────────────────────────────
        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)

        stat.summary(self.global_step, reset=True)
        print(f"  checkpoint saved: {saved}")
        stopwatch.summary(reset=True)
        print("  total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Vectorized IBRL training loop.

        Loop structure (per iteration)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. **Act**: query ``agent.act()`` with batched obs ``(N, 14)`` →
           batched actions ``(N, 7)``.
        2. **Step**: advance all N environments simultaneously.
        3. **Add**: push the N transitions into ``VecReplayBuffer``.
        4. **Train**: every ``update_freq`` transitions, perform one
           critic update (and one actor update on the last critic step).
        5. **Log/eval**: every ``log_per_step`` transitions, run eval and
           save a checkpoint.

        ``global_step`` counts *transitions* (not env steps), so it
        increments by ``num_envs`` each iteration.
        """
        cfg = self.cfg

        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(cfg.use_wb),
            wb_exp_name=cfg.wb_exp,
            wb_run_name=cfg.wb_run,
            wb_group_name=cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        # ── warm-up if needed ──────────────────────────────────────────────
        if not self.replay.ready(min_episodes=cfg.num_envs):
            self.warm_up()

        # ── initial env reset ──────────────────────────────────────────────
        obs = self.train_env.reset()
        self.replay.new_episodes(obs)

        stopwatch = common_utils.Stopwatch()
        last_log_step = 0
        last_update_step = 0

        print(common_utils.wrap_ruler("training"))
        while self.global_step < cfg.num_train_step:
            # ── 1. act ────────────────────────────────────────────────────
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(cfg.stddev_schedule, self.global_step)
                # obs["state"]: (N, 14) → dim==2 → _maybe_unsqueeze_ is a no-op
                # Returns (N, 7) actions on rl_device
                actions = self.agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
                stat["data/stddev"].append(stddev)

            # ── 2. env step ───────────────────────────────────────────────
            with stopwatch.time("env_step"):
                obs, rewards, dones, successes = self.train_env.step(actions)

            # ── 3. add to replay ──────────────────────────────────────────
            with stopwatch.time("replay_add"):
                self.replay.add_step(obs, actions, rewards, dones, successes)
                # Each env step produces num_envs transitions
                self.global_step += cfg.num_envs

            # ── episode stats ─────────────────────────────────────────────
            n_done = int(dones.sum().item())
            if n_done > 0:
                self.global_episode += n_done
                n_success = int(successes[dones].sum().item())
                stat["score/train_success_rate"].append(n_success / n_done)
                stat["data/episode_reward"].append(
                    float(self.train_env.episode_reward[dones].mean().item())
                )

            # ── 4. train ──────────────────────────────────────────────────
            if (
                self.global_step - last_update_step >= cfg.update_freq
                and self.replay.ready(min_episodes=max(1, cfg.batch_size // 10))
            ):
                last_update_step = self.global_step
                with stopwatch.time("train"):
                    self._rl_train(stat)
                    self.train_step += 1

            # ── 5. log & eval ─────────────────────────────────────────────
            if self.global_step - last_log_step >= cfg.log_per_step:
                last_log_step = self.global_step
                self._log_and_save(stopwatch, stat, saver)
                # After eval the env state is undefined; reset all envs and
                # re-synchronise the replay episode trackers before resuming.
                obs = self.train_env.reset()
                self.replay.new_episodes(obs)

        # Final evaluation
        print(common_utils.wrap_ruler("final evaluation"))
        final_score = self.eval()
        print(f"Final success rate: {final_score:.4f}")
        saver.save(self.agent.state_dict(), final_score, save_latest=True)


# ---------------------------------------------------------------------------
# Model loading (for post-hoc evaluation)
# ---------------------------------------------------------------------------


def load_model(weight_file: str, device: str):
    """Load a saved agent checkpoint for evaluation.

    Parameters
    ----------
    weight_file : str
        Path to a ``.pt`` checkpoint saved by ``TopkSaver``.
    device : str
        PyTorch device string, e.g. ``"cuda:0"``.

    Returns
    -------
    agent : QAgent
        The loaded agent in eval mode.
    cfg : MainConfig
        The config used to create the agent.
    env_params : dict
        Keyword arguments that can be forwarded to ``IsaacGymBulbEnv``.
    """
    run_folder = os.path.dirname(weight_file)
    cfg_path = os.path.join(run_folder, "cfg.yaml")

    cfg: MainConfig = pyrallis.load(MainConfig, open(cfg_path))  # type: ignore

    agent = QAgent(
        use_state=1,
        obs_shape=(cfg.obs_dim,),
        prop_shape=(cfg.obs_dim,),
        action_dim=cfg.action_dim,
        rl_camera="",
        cfg=cfg.q_agent,
    )
    agent.load_state_dict(torch.load(weight_file, map_location=device))
    agent.to(device)
    agent.train(False)

    env_params = dict(
        isaacgym_envs_path=cfg.isaacgym_envs_path,
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        seed=cfg.seed,
        env_reward_scale=cfg.env_reward_scale,
    )

    return agent, cfg, env_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import rich.traceback

    rich.traceback.install()
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    cfg: MainConfig = pyrallis.parse(config_class=MainConfig)  # type: ignore

    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main()
