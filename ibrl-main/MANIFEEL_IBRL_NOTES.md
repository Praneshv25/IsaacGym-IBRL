# ManiFeel + IBRL Integration Notes

This note summarizes the main problems encountered while integrating a pretrained ManiFeel Diffusion Policy (DP) checkpoint into `ibrl-main` for online RL on the bulb task, along with the solutions and code changes that were made.

## Goal

Use a pretrained ManiFeel BC/Diffusion Policy checkpoint as the BC branch inside IBRL, then continue improving performance with online RL on the bulb task.

## Core Problem

The ManiFeel bulb environment worked in ManiFeel's own `train.py` / `eval.py` flow, but repeatedly failed when driven through custom `ibrl-main` environment wrappers and replay logic.

The integration issues were not caused by the bulb task itself. They came from mismatches between:

- ManiFeel's expected environment lifecycle
- Isaac Gym GPU pipeline constraints
- `ibrl-main`'s original MuJoCo-oriented replay/training assumptions

## Main Problems We Hit

### 1. GPU pipeline crash with old Isaac Gym API usage

We repeatedly hit:

```text
Function GymGetActorDofStates cannot be used with the GPU pipeline after simulation starts.
```

This happened when the bulb task was launched through the wrong path or when a second Isaac stack was created in-process.

Why it mattered:

- bulb depends on GPU-only features like SDF/contact behavior
- switching to CPU pipeline was not a viable workaround
- the stable ManiFeel path had to be respected more closely

### 2. Separate train and eval env stacks were unstable

Early versions of the RL trainer used:

- one direct train env for online RL
- a separate ManiFeel eval runner/env stack for evaluation

This led to repeated segfaults, especially at eval boundaries, because the process was trying to create a second live Isaac Gym bulb stack after already using one.

Important finding:

- ManiFeel `train.py` does **not** create separate train/eval env stacks
- it keeps one long-lived runner and reuses it

### 3. ManiFeel eval reward was not the dense RL reward

ManiFeel's rollout wrapper is evaluation-oriented and uses a success/reset-style reward path.

That is fine for `test/mean_score`, but not ideal as the training reward for online RL.

Key finding:

- the direct bulb task wrapper returns the real dense reward each step
- this dense reward appears to be the shaped signal RL should train on

### 4. Original `ibrl-main` replay path was a bad fit

The original replay logic in `ibrl-main` was designed around simpler environments and CPU-oriented storage assumptions.

Problems we hit:

- crashes when copying Isaac Gym tensors into replay
- instability around GPU-to-CPU observation materialization
- the replay path did not naturally align with Diffusion Policy horizon semantics

### 5. Diffusion Policy expects history, replay stores single steps

The BC model expects an observation history over `n_obs_steps`.

But `ibrl-main` replay is fundamentally transition-based and stores single-step observations.

That mismatch had to be handled explicitly.

### 6. Image-size mismatch between checkpoint and env

When using a `96x96` BC checkpoint, the trainer was still feeding `256x256` wrist images.

This caused BC encoder assertion failures.

### 7. Subset reset in TacSL bulb was broken

We hit:

```text
RuntimeError: shape mismatch: value tensor of shape [8, 3] cannot be broadcast to indexing result of shape [1, 3]
```

This came from the bulb task's object reset path when trying to reset only a subset of environments.

## Final Architecture We Moved To

### Unified single-stack env design

Instead of separate train/eval env stacks, the current setup uses one long-lived direct bulb env stack:

- total envs = `num_train_envs + num_eval_envs`
- the first subset is used for training
- the second subset is used for evaluation
- no second Isaac stack is created during training

This is much closer to the stable ManiFeel pattern.

### Direct dense-reward train env

Training now uses a direct `TacSLTaskBulb`-based wrapper instead of ManiFeel's eval runner.

Why:

- we need dense reward for RL
- we need stable long-lived env ownership
- we want to avoid ManiFeel's eval-specific reward rewriting during training

### Same stack used for evaluation

Evaluation now runs on the held-out eval subset of the same env stack instead of launching a fresh ManiFeel runner in-process.

This avoids the previous eval-time segfaults caused by train/eval stack swapping.

## Important Code Changes

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/train_rl_manifeel.py`

Main changes:

- introduced ManiFeel-specific RL trainer path
- replaced old separate eval-runner lifecycle with a unified single-stack design
- created train/eval env index partitions
- switched replay to a GPU-native local replay buffer
- added warmup and train/eval progress bars with `tqdm`
- added optional eval step cap (`eval_max_steps`) for smoke testing
- fixed eval video export by explicitly setting codec

Notable architectural changes:

- one env stack for both train and eval
- train steps use only `train_idx`
- eval steps use only `eval_idx`
- `global_step` tracks training env steps

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/env/manifeel_bulb_wrapper.py`

Main changes:

- added direct `TacSLTaskBulb` wrapper for RL training
- compose ManiFeel bulb config under Hydra correctly
- expose dense reward, success, wrist/state observations
- maintain BC history tensors (`bc_wrist`, `bc_state`)
- resize wrist images to the configured target resolution
- build `state = concat(ee_pos, ee_quat)` for BC compatibility

Current workaround:

- because subset reset in the bulb task is unstable, the wrapper currently performs a **full vector reset whenever any env finishes**

This is not ideal RL semantics, but it keeps the pipeline running.

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/bc/diffusion_policy_adapter.py`

Main changes:

- load ManiFeel DP checkpoint for BC action proposals
- support history input using either:
  - explicit `bc_wrist` / `bc_state`
  - or repeated current observations if needed

This is how the BC branch remains compatible with a transition-style RL loop.

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/release/cfgs/manifeel/vision_wrist_bulb_ibrl.yaml`

Main changes:

- added ManiFeel-specific RL config values
- image size aligned with the selected BC checkpoint
- smoke-test parameters were added and later restored to longer-run values
- `gpu_replay_capacity` added for the custom GPU replay buffer
- `eval_max_steps` added for quick end-to-end tests

## Replay Changes

The original CPU/`rela` episode replay path was effectively abandoned for this ManiFeel trainer.

Reason:

- it repeatedly crashed on Isaac Gym observation handling
- it assumed a simpler environment/data flow than this setup provides

Current solution:

- a local GPU replay buffer is used inside `train_rl_manifeel.py`
- it stores:
  - current obs
  - next obs
  - action
  - reward
  - bootstrap mask

The replay stores single-step transitions, not full DP horizon windows.

## Warmup Behavior

Warmup currently:

- uses only the frozen BC/Diffusion Policy to act
- fills replay with initial transitions
- does **not** train the Q function directly

The Q function begins training after warmup, using the replay data warmup collected.

## Evaluation Behavior

Evaluation currently:

- runs on the eval subset of the same env stack
- uses the RL agent in eval mode
- records one wrist-camera video when enabled
- logs `test/mean_score` and video to WandB

For smoke testing, eval was temporarily reduced to one step. That was later restored to full episode-based eval.

## Current Known Limitations

### 1. `num_train_envs` is still effectively limited to 1

The current IBRL action-selection path in `QAgent` still assumes batch size 1 in training mode.

This is the main reason full training remains slow.

If that path is generalized to support batch size >1, training speed should improve substantially.

### 2. Full-reset workaround on any done

Because subset resets in the bulb task are unstable, one env finishing causes a full env-stack reset.

This keeps the system alive but is not ideal.

### 3. Eval metric is now trainer-local

The current eval loop no longer uses ManiFeel's `env_runner.run(policy)` directly.

That was necessary for stability, but it means eval now depends on the unified-stack RL trainer implementation rather than ManiFeel's original runner abstraction.

## Practical Runtime Notes

With the current stable setup:

- warmup is slow because it runs BC rollout in the real bulb env
- training is slow mainly because `num_train_envs = 1`
- a full 200k-step run can take a long time

The single biggest runtime improvement would likely be:

- fixing `QAgent` train-time action selection to support batched envs, e.g. `num_train_envs = 8`

## Summary

The key lessons from this integration were:

1. ManiFeel bulb is stable when the simulator stack stays long-lived.
2. Dense-reward RL should use the direct task env, not ManiFeel's eval reward wrapper.
3. The original CPU replay assumptions in `ibrl-main` were a poor fit for Isaac Gym + ManiFeel.
4. A unified single-stack train/eval architecture was necessary to avoid repeated segfaults.
5. The current pipeline works, but there are still speed and batching limitations worth improving next.
