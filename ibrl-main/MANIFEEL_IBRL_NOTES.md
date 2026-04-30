# ManiFeel + IBRL Integration Notes

This file summarizes the main issues we hit while integrating ManiFeel Diffusion Policy into `ibrl-main` for online RL on the bulb task, and the code changes we made to get to the current setup.

## Goal

Use a pretrained ManiFeel BC / Diffusion Policy checkpoint as the BC branch inside IBRL, then continue improving the policy with online RL on the bulb task.

## Main Problems We Hit

### 1. ManiFeel's stable path did not match `ibrl-main`'s original assumptions

ManiFeel's own `train.py` / `eval.py` flow was stable, but early `ibrl-main` integrations were not.

The main mismatch was:

- ManiFeel expects a long-lived Isaac Gym stack
- original `ibrl-main` code assumed simpler env / replay behavior
- Isaac Gym bulb + camera + tactile tensors are much less forgiving than MuJoCo-style paths

### 2. Separate train and eval env stacks caused crashes

Earlier versions used:

- one direct env for training
- a separate eval runner/env for rollout

That repeatedly led to segfaults when a second Isaac Gym bulb stack was created in-process.

Key lesson:

- ManiFeel stays stable by reusing one live runner
- recreating a fresh Isaac stack during training was the wrong pattern here

### 3. ManiFeel eval reward was not the RL training reward

ManiFeel eval uses a success/reset-style metric path. That is fine for rollout scoring, but it is not the same as the task reward that online RL should train on.

We verified:

- direct bulb task stepping exposes the real dense reward
- ManiFeel eval logic is better treated as evaluation semantics, not training semantics

### 4. The original CPU / `rela` replay path was a bad fit

The original `ibrl-main` replay path assumed simpler env outputs and CPU-friendly observation materialization.

In this integration it caused:

- crashes when copying Isaac Gym observations into replay
- instability around GPU-to-CPU transfers
- awkward mismatch with Diffusion Policy history expectations

We replaced it for this trainer with a local GPU-native replay path.

### 5. Diffusion Policy expects observation history, replay stores single steps

The BC model expects `n_obs_steps` history. The RL trainer stores transitions one step at a time.

We handled that by:

- storing single-step RL observations in replay
- keeping BC history inside the live env wrapper
- reconstructing the BC-compatible observation format at action time

### 6. Image-size mismatch between checkpoint and env

When using a `96x96` checkpoint, the trainer was still feeding `256x256` wrist images at one point.

This caused BC encoder shape assertions.

The fix was:

- make the RL wrapper enforce the configured image size
- keep the live RL / BC observation path aligned to `image_size`

### 7. Subset reset in TacSL bulb was not safe enough for RL

The original bulb task file was written around ManiFeel's assumptions, not this online RL setup.

We hit subset-reset shape bugs and control-path bugs while trying to reset only some envs.

Examples included:

- object reset shape mismatch on subset envs
- methods accidentally assuming full-env tensors
- gripper target shape assumptions breaking with vectorized RL control

### 8. Dense reward produced unstable behavior

With dense reward, the hybrid policy tended to drift into bad RL behavior quickly:

- critic and actor losses were large
- eval success stayed at zero
- videos showed the robot thrashing / wandering

This made the dense setup look numerically active but behaviorally poor.

### 9. Sparse reward was more stable, but replay became dominated by failures

Sparse reward made training much calmer, but then most online transitions had zero reward.

That meant:

- successful BC-like behavior was getting drowned out
- the critic mostly trained on zero-reward failures
- the hybrid policy could still drift away from the strong BC baseline

This motivated the later demo-buffer changes.

### 10. The gripper was a major instability source

Videos showed a common failure pattern:

- the hybrid policy would flicker or perturb the gripper
- the bulb would get dropped
- the resulting state went out of the good BC distribution
- behavior then became erratic

This suggested that letting RL directly control the gripper too early was a bad idea.

## Current Architecture

### Unified single-stack train/eval design

The current trainer uses one long-lived direct bulb env stack:

- total envs = `num_train_envs + num_eval_envs`
- first subset = training envs
- second subset = eval envs
- no separate ManiFeel eval runner is launched during training

This is much closer to ManiFeel's stable “one live stack” pattern.

### Direct RL env wrapper

Training uses a direct bulb wrapper instead of ManiFeel's eval runner:

- reward comes directly from the task wrapper
- BC-compatible `wrist`, `prop`, `state`, `bc_wrist`, `bc_state` are built there
- wrist images are resized to the configured `image_size`

### Separate IBRL-specific bulb task file

To avoid breaking Diffusion Policy training, we split the task file:

- original DP path still uses:
  - `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/Diffusion Policy/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bulb.py`
- RL path now uses:
  - `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/Diffusion Policy/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bulb_ibrl.py`

The RL wrapper imports `TacSLTaskBulbIBRL` explicitly.

## Important Code Changes

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/train_rl_manifeel.py`

Main trainer changes:

- added a ManiFeel-specific online RL trainer
- uses one unified env stack for train and eval
- keeps separate train / eval env index subsets
- replaced the old CPU / `rela` replay path with a local GPU replay buffer
- added `tqdm` progress bars for warmup, train, and eval
- records eval videos directly from the same stack

Replay / data changes:

- added a normal online replay buffer:
  - `self.replay`
- added a separate demo replay buffer:
  - `self.demo_replay`
- warmup BC rollout now fills:
  - online replay with all transitions
  - demo replay with only successful BC episodes
- each update now uses a fixed mixed batch:
  - online replay samples
  - plus demo replay samples according to `demo_batch_ratio`
- eval video now prefers the external/front-style camera key instead of always logging wrist view

Actor-update changes:

- actor BC regularization is enabled through `add_bc_loss`
- BC regularization now samples from the demo replay, not the polluted online replay
- this increases BC influence specifically on successful BC data

Current important config knobs in this trainer:

- `reward_mode`
- `gpu_replay_capacity`
- `demo_replay_capacity`
- `demo_batch_ratio`
- `add_bc_loss`

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/env/manifeel_bulb_wrapper.py`

Main wrapper changes:

- direct ManiFeel bulb RL env wrapper
- imports `TacSLTaskBulbIBRL`
- composes Hydra config correctly for direct task construction
- exposes:
  - `wrist`
  - `prop`
  - `state`
  - BC history tensors
- can also carry an external debug camera such as `client` / `front`
- enforces the configured image size
- supports:
  - `reward_mode="dense"`
  - `reward_mode="sparse"`

Sparse reward behavior:

- sparse reward uses success as the training reward
- dense reward uses the task reward returned by the env

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/rl/q_agent.py`

Main changes:

- made the active `ibrl` path batch-safe for train acting
- removed the old train-time `bsize == 1` assumption from `_act_ibrl()`
- train/eval/bootstrap BC-selection stats now work with batched envs
- gripper stabilization change:
  - in the active `ibrl` path, the final gripper command now always comes from BC
  - the RL branch effectively controls only the first 6 action dims
  - actor loss is trained consistently with the same BC-gripper substitution

This is what enabled moving `num_train_envs` from 1 to 8 for the active `ibrl` path.

The `ibrl_soft` path was intentionally left untouched.

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/Diffusion Policy/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bulb_ibrl.py`

Main RL-only task changes:

- created a separate `TacSLTaskBulbIBRL` class
- fixed subset reset behavior in the RL path
- fixed subset object reset assignment bugs
- made gripper/control target handling accept vectorized RL shapes
- made subset robot/object reset helpers work on selected envs instead of assuming full-env resets

This file is the main place where ManiFeel bulb task behavior was adapted for online vectorized RL.

### `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/ibrl-main/release/cfgs/manifeel/vision_wrist_bulb_ibrl.yaml`

Current important settings:

- `num_train_envs: 8`
- `num_eval_envs: 50`
- `num_eval_episode: 2`
- `episode_length: 800`
- `image_size: 96`
- `reward_mode: "sparse"`
- `gpu_replay_capacity: 2000`
- `demo_replay_capacity: 20000`
- `demo_batch_ratio: 0.5`
- `add_bc_loss: 1`
- `q_agent.bc_loss_coef: 1.0`

Important reminder:

- the YAML still points to an older local checkpoint path
- actual Linux runs should override `--dp_checkpoint` with the correct `96x96` BC checkpoint

## Reward Evolution

### Dense reward phase

Dense reward was tested first because it matches the underlying task reward.

Result:

- RL quickly became behaviorally unstable
- eval success stayed at zero
- videos looked worse than BC

Conclusion:

- dense reward gave a lot of signal, but not the right behavioral pressure for this setup

### Sparse reward phase

Sparse reward was then used, with success as the reward signal.

Result:

- training became much more numerically stable
- BC remained a stronger anchor
- but online replay was still mostly filled with zero-reward failures

Conclusion:

- sparse reward was better than dense reward for stability
- but needed extra help to preserve successful BC behavior

## Demo / Replay Strategy

The current replay strategy is:

- online replay:
  - stores all online transitions
- demo replay:
  - stores only successful BC warmup episodes

Training batches are mixed from both buffers.

Why:

- online replay provides current on-policy-ish coverage
- demo replay preserves what successful behavior looks like
- actor BC loss should pull toward successful BC trajectories, not failed online ones

This was added specifically because the sparse-reward run showed the critic was learning from almost entirely zero-reward online data.

## Gripper Constraint

The current hybrid policy uses a simple stabilization rule:

- BC controls the gripper dimension
- RL controls only the first 6 arm-control dimensions

Why:

- the gripper was the easiest way for early RL to destroy otherwise good BC behavior
- once the bulb was dropped, the policy quickly entered bad out-of-distribution states

This is a practical safety constraint to keep the hybrid policy closer to the strong BC baseline.

## Evaluation Behavior

Evaluation currently:

- runs on the eval subset of the same live env stack
- uses the hybrid IBRL policy in eval mode
- logs `test/mean_score`
- records wrist-camera video
- can run up to 50 eval envs in parallel

For easier debugging, eval video now prefers the configured external camera key, e.g. `client` or `front`, and falls back to `wrist` if needed.

This is no longer ManiFeel's original `env_runner.run(policy)` path. It is trainer-local eval built on the unified env stack.

## Current Known Limitations

### 1. This is still not the exact original ManiFeel eval path

The current eval loop is custom and trainer-local.

It is stable enough for online RL, but it is not identical to ManiFeel's offline eval pipeline.

### 2. The IBRL-specific bulb task file is still the riskiest integration point

Most of the hard environment adaptation work lives in:

- `/Users/PV/Desktop/CS 441/IsaacGym-IBRL/Diffusion Policy/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bulb_ibrl.py`

That file should be treated carefully because it is where most RL-specific reset/control fixes were made.

### 3. BC influence is stronger now, but still heuristic

The current solution uses:

- success-only demo replay
- fixed demo/online batch mixing
- actor BC regularization

This is a practical stabilization strategy, not a fully new RL algorithm.

## Practical Runtime Notes

With the current setup:

- warmup is expensive because it uses real BC rollout in the bulb env
- 8 train envs substantially improve throughput over 1 env
- 50 eval envs make eval more expensive, but much faster in env-step terms than serial eval
- sparse reward is currently the more promising training mode

## Summary

The integration moved from an unstable “bolt ManiFeel onto old `ibrl-main` assumptions” approach to a ManiFeel-specific online RL path with:

- one unified live Isaac stack
- a separate RL-specific bulb task file
- batch-safe `ibrl` acting
- sparse reward support
- GPU-native replay
- a success-only demo replay buffer
- mixed demo/online training batches
- stronger BC influence in actor updates

The main current strategy is:

- keep BC behavior alive explicitly
- let RL learn on top of it more cautiously
- avoid letting online failure data completely wash out the successful BC prior
