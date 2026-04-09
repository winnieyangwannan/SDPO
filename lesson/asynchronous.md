# Asynchronous vs On-Policy Training in verl

## Current Setting: Fully On-Policy (Synchronous)

The default configuration in /home/winnieyangwn/SDPO/experiments/rich_feedback/run_sdpo.sh runs **fully on-policy, synchronous training**.

---

## Training Loop Flow

```
for each epoch:
    for each batch:
        1. Generate rollouts (blocking) using weights θ_{t-1}
        2. Recompute old_log_probs on CURRENT weights θ_t  ← makes it on-policy
        3. Train actor & critic
        4. Sync weights to rollout workers
```

Key code location: `verl/trainer/ppo/ray_trainer.py` - `fit()` method

---

## Three-Policy Setup

| Policy | Weights | Purpose |
|--------|---------|---------|
| π_rollout | θ_{t-1} | Generates trajectories (slightly stale) |
| π_old | θ_t | Recomputed reference for PPO clipping |
| π_θ | θ_t | Policy being trained |

**Why it's on-policy**: Even though rollouts use θ_{t-1}, the `old_log_probs` are recomputed on θ_t before training. This means PPO's ratio `π_θ / π_old` uses matching weights, making it effectively on-policy.

---

## Three Training Modes Supported

| Mode | Description | Config |
|------|-------------|--------|
| **On-policy (Sync)** | Rollout then train serially, colocated | **Default (main_ppo.py)** |
| One-step-off-policy | Parallelize generation & training slightly | Not commonly used |
| Fully async (Decoupled) | Separate trainer/rollout nodes | Experimental |

---

## How to Enable Off-Policy Training

Set `rollout_correction.bypass_mode=True`:

```yaml
rollout_correction:
  bypass_mode: True
  calculate_log_probs: True  # Required
```

This skips recomputation and uses `rollout_log_probs` directly from π_rollout, enabling:
- True off-policy training
- Importance sampling corrections
- 2-policy setup (π_rollout, π_θ) instead of 3-policy

---

## Weight Synchronization

`CheckpointEngine` handles weight sync after training:

1. `abort_all_requests()` - pause rollout
2. `sleep_replicas()` - free KV cache
3. Transfer weights via NCCL/NIXL
4. `wake_up_replicas()` - resume generation

Code: `verl/checkpoint_engine/base.py` - `CheckpointEngineManager.update_weights()`

---

## Summary

| Aspect | Current Behavior |
|--------|------------------|
| Rollout generation | Blocking (sync) |
| Rollout weights | θ_{t-1} (1 step stale) |
| old_log_probs | Recomputed on θ_t |
| Effective policy | **On-policy** |
| Off-policy mode | Available via `bypass_mode=True` |
