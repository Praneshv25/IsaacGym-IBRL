"""
IsaacGym vectorized environment wrapper for IBRL.

Wraps TacSLTaskBulb (a VecTask with N parallel envs) into an interface
compatible with IBRL's QAgent and training loop.

Observation space (state-only prototype):
    ee_pos   (3)  +  ee_quat   (4)
  + socket_pos(3) +  socket_quat(4)
  = 14-D float32 tensor

Action space: 7-D  (6-DoF task-space delta + gripper width)

For *done* environments the returned obs from step() is the **new
episode's initial observation** (after the internal reset), so callers
can feed it straight into the next episode's trajectory without any
extra book-keeping.
"""

# from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

# IsaacGym must be imported before PyTorch when both are present.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys in TacSLTaskBulb.obs_dict that we concatenate into the state vector
OBS_KEYS: List[str] = ["ee_pos", "ee_quat", "socket_pos", "socket_quat"]

STATE_DIM: int = 3 + 4 + 3 + 4  # = 14
ACTION_DIM: int = 7  # 6-DoF delta + gripper


# ---------------------------------------------------------------------------
# Resolver / hydra helpers
# ---------------------------------------------------------------------------


def _register_omegaconf_resolvers() -> None:
    """Register the custom OmegaConf resolvers used by isaacgymenvs configs.

    Safe to call multiple times – duplicate registrations are silently ignored.
    """
    from omegaconf import OmegaConf

    resolvers = {
        "eq": lambda x, y: x.lower() == y.lower(),
        "contains": lambda x, y: x.lower() in y.lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
        "eval": eval,
    }
    for name, fn in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass  # already registered – harmless


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_tacsl_bulb_task(
    *,
    isaacgym_envs_path: str,
    num_envs: int,
    sim_device: str,
    rl_device: str,
    graphics_device_id: int,
    headless: bool,
    seed: int,
    extra_overrides: Optional[List[str]] = None,
):
    """Compose the TacSLTaskBulb hydra config (cameras / tactile OFF) and
    directly instantiate the VecTask.

    Parameters
    ----------
    isaacgym_envs_path:
        Filesystem path to the root of the ``manifeel-isaacgymenvs-*`` repo
        (i.e. the directory that contains ``isaacgymenvs/``).
    num_envs:
        Number of parallel simulation environments.
    sim_device / rl_device:
        PyTorch device strings, e.g. ``"cuda:0"``.
    graphics_device_id:
        GPU index used for rendering.  ``-1`` means headless / no display.
    headless:
        If ``True`` the simulator runs without a viewer window.
    seed:
        Random seed forwarded to the environment.
    extra_overrides:
        Optional list of additional hydra overrides, e.g.
        ``["task.rl.max_episode_length=200"]``.

    Returns
    -------
    TacSLTaskBulb instance (a ``VecTask`` subclass).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from isaacgymenvs.tasks.tacsl.tacsl_task_bulb import TacSLTaskBulb
    from isaacgymenvs.utils.reformat import omegaconf_to_dict

    _register_omegaconf_resolvers()

    cfg_dir = os.path.abspath(os.path.join(isaacgym_envs_path, "isaacgymenvs", "cfg"))

    # Clear any previous hydra initialisation so we can re-init
    GlobalHydra.instance().clear()

    # Base overrides: disable every sensor so the env is pure state-based
    # ``config.yaml`` defaults to ``train=${task}PPO`` → ``TacSLTaskBulbPPO``,
    # which is not shipped in manifeel-isaacgymenvs-tacsl; use the bundled
    # bulb train config instead (override via ``extra_overrides`` if your fork
    # differs).
    base_overrides: List[str] = [
        "task=TacSLTaskBulb",
        "train=TacSLTaskBulbInsertionPPO_LSTM_dict_AAC",
        f"num_envs={num_envs}",
        f"seed={seed}",
        "headless=true",
        # ── disable cameras & tactile sensors ──────────────────────────────
        "task.env.use_camera_obs=false",
        "task.env.use_isaac_gym_tactile=false",
        "task.env.use_shear_force=false",
        "task.env.use_camera=false",
        "task.env.enableCameraSensors=false",
        # ── dict observations (only the 14-D state keys will be populated) ─
        "task.env.use_dict_obs=true",
    ]

    if extra_overrides:
        base_overrides = base_overrides + extra_overrides

    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(config_name="config", overrides=base_overrides)

    task_config: dict = omegaconf_to_dict(cfg.task)
    # Ensure numEnvs is consistent
    task_config["env"]["numEnvs"] = num_envs

    env = TacSLTaskBulb(
        cfg=task_config,
        rl_device=rl_device,
        sim_device=sim_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        virtual_screen_capture=False,
        force_render=False,
    )
    return env


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


class IsaacGymBulbEnv:
    """IBRL-compatible wrapper around ``TacSLTaskBulb``.

    Runs *num_envs* parallel simulation environments simultaneously and
    exposes a clean vectorised interface.

    State vector
    ~~~~~~~~~~~~
    ``ee_pos`` (3) + ``ee_quat`` (4) + ``socket_pos`` (3) +
    ``socket_quat`` (4) = **14 floats**.

    Interface
    ~~~~~~~~~
    ::

        env = IsaacGymBulbEnv(num_envs=64, ...)

        obs = env.reset()
        # obs = {'state': Tensor(N, 14), 'prop': Tensor(N, 14)}

        obs, rewards, dones, successes = env.step(actions)
        # actions   : Tensor(N, 7)  – must reside on *rl_device*
        # obs       : {'state': Tensor(N, 14), 'prop': Tensor(N, 14)}
        # rewards   : Tensor(N,)
        # dones     : Tensor(N,)  bool – episode ended
        # successes : Tensor(N,)  bool – episode succeeded

    For *done* environments the returned ``obs`` contains the **new
    episode's initial observation** (after the internal ``reset_idx``
    call), so callers can immediately begin the next episode's trajectory.

    IBRL / QAgent compatibility
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Setting ``use_state=True`` in ``QAgentConfig`` makes the agent use
    ``obs["state"]`` directly (no image encoder).  The wrapper exposes the
    three attributes that QAgent's constructor reads::

        env.observation_shape  ->  (14,)
        env.prop_shape         ->  (14,)
        env.action_dim         ->  7
    """

    # ── QAgent compatibility attributes (set in __init__) ─────────────────
    observation_shape: tuple
    prop_shape: tuple
    action_dim: int

    def __init__(
        self,
        *,
        isaacgym_envs_path: str,
        num_envs: int = 64,
        sim_device: str = "cuda:0",
        rl_device: str = "cuda:0",
        graphics_device_id: int = -1,
        headless: bool = True,
        seed: int = 0,
        env_reward_scale: float = 1.0,
        extra_overrides: Optional[List[str]] = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = rl_device
        self.env_reward_scale = env_reward_scale

        self.ig_env = make_tacsl_bulb_task(
            isaacgym_envs_path=isaacgym_envs_path,
            num_envs=num_envs,
            sim_device=sim_device,
            rl_device=rl_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            seed=seed,
            extra_overrides=extra_overrides,
        )

        # ── IBRL / QAgent compatibility ────────────────────────────────────
        self.action_dim = ACTION_DIM
        self.observation_shape = (STATE_DIM,)
        self.prop_shape = (STATE_DIM,)  # prop == state when use_state=True

        # ── per-env episode bookkeeping ────────────────────────────────────
        self._episode_reward = torch.zeros(num_envs, device=rl_device)
        self._episode_step = torch.zeros(num_envs, dtype=torch.long, device=rl_device)
        self._total_episodes = 0
        self._total_successes = 0

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _obs_from_env(self) -> Dict[str, torch.Tensor]:
        """Read the current ``ig_env.obs_dict`` and return an IBRL obs dict.

        The 14-D state is built by concatenating the four keys in
        ``OBS_KEYS`` along the last dimension.  Both ``"state"`` and
        ``"prop"`` point to the same tensor (no copy) so downstream code
        can use either key.
        """
        parts = [self.ig_env.obs_dict[k] for k in OBS_KEYS]
        state = torch.cat(parts, dim=-1).float()  # (N, 14)
        return {"state": state, "prop": state}

    def _apply_reset_for_done_envs(self, done_ids: torch.Tensor) -> None:
        """Call ``reset_idx`` on the done environments and refresh ``obs_dict``.

        After this call, ``_obs_from_env()`` will return the **new
        episode's initial observations** for the reset envs.
        """
        if done_ids.numel() == 0:
            return
        self.ig_env.reset_idx(done_ids)
        # Recompute obs_dict so done envs now show their fresh initial state.
        # Active envs are unaffected (their physics state did not change).
        self.ig_env.compute_observations()

        # Reset per-env counters for the envs that just started a new episode
        self._episode_reward[done_ids] = 0.0
        self._episode_step[done_ids] = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset **all** N environments and return their initial observations.

        Returns
        -------
        obs_dict : ``{'state': Tensor(N, 14), 'prop': Tensor(N, 14)}``
        """
        self._episode_reward.zero_()
        self._episode_step.zero_()

        # VecTask.reset() does bookkeeping but does not always fill obs_dict.
        self.ig_env.reset()
        # Force a fresh observation computation so obs_dict is populated.
        self.ig_env.compute_observations()

        return self._obs_from_env()

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Step all N environments with the provided actions.

        Parameters
        ----------
        actions : ``Tensor(N, 7)`` on *rl_device*
            Batched action tensor.

        Returns
        -------
        obs_dict  : ``{'state': Tensor(N,14), 'prop': Tensor(N,14)}``
            For *done* envs this is the **new episode's initial obs**.
        rewards   : ``Tensor(N,)``  scaled by *env_reward_scale*.
        dones     : ``Tensor(N,)``  bool – ``True`` when an episode ended.
        successes : ``Tensor(N,)``  bool – ``True`` when the episode was a
                    success (plug fully inserted).
        """
        # ── physics step ────────────────────────────────────────────────
        # VecTask.step() runs pre/post physics, computes obs & reward,
        # and fills reset_buf.  It does NOT call reset_idx automatically.
        _, raw_rewards, resets, _ = self.ig_env.step(actions)

        rewards = raw_rewards.float() * self.env_reward_scale
        dones = resets.bool()

        # Capture success flags BEFORE reset_idx clears the plug/socket
        # positions for done environments.
        successes = self.ig_env._check_success().bool()

        # ── per-env bookkeeping ─────────────────────────────────────────
        self._episode_reward += rewards
        self._episode_step += 1

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._total_episodes += int(done_ids.numel())
            self._total_successes += int(successes[done_ids].sum().item())
            # Reset done envs and refresh obs → done envs get new initial obs
            self._apply_reset_for_done_envs(done_ids)

        # ── build and return obs ─────────────────────────────────────────
        # After _apply_reset_for_done_envs, obs_dict holds:
        #   active envs → post-step observations
        #   done   envs → new-episode initial observations
        obs_dict = self._obs_from_env()

        return obs_dict, rewards, dones, successes

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics / properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        """Fraction of completed episodes that were successes."""
        if self._total_episodes == 0:
            return 0.0
        return self._total_successes / self._total_episodes

    @property
    def episode_reward(self) -> torch.Tensor:
        """Current per-env cumulative episode reward (since last reset)."""
        return self._episode_reward.clone()

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length as configured in the YAML."""
        return int(self.ig_env.max_episode_length)

    def __repr__(self) -> str:
        return (
            f"IsaacGymBulbEnv("
            f"num_envs={self.num_envs}, "
            f"obs_dim={STATE_DIM}, "
            f"action_dim={ACTION_DIM}, "
            f"device={self.device})"
        )
