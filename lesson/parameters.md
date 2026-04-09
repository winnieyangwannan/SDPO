# GRPO Training Parameters

## Batch Size Parameters

### `data.train_batch_size` (TRAIN_BATCH_SIZES in scripts)

**What it controls:** The global batch size of **prompts** sampled per training step.

From docs/algo/grpo.md:
> `data.train_batch_size`: The global batch size of prompts used to generate a set of sampled trajectories/rollouts.

### `actor_rollout_ref.rollout.n` (ROLLOUT_BATCH_SIZES in scripts)

**What it controls:** The number of **responses generated per prompt** (group sampling).

From docs/algo/grpo.md:
> `actor_rollout.ref.rollout.n`: For each prompt, sample n times. Default to 1. For GRPO, please set it to a value larger than 1 for group sampling.

### `actor_rollout_ref.actor.ppo_mini_batch_size` (MINI_BATCH_SIZES in scripts)

**What it controls:** The mini-batch size for PPO/GRPO actor updates. This is the **effective batch size per gradient update**.

From docs/algo/grpo.md:
> `actor_rollout_ref.actor.ppo_mini_batch_size`: The set of sampled trajectories is split into multiple mini-batches with batch_size=ppo_mini_batch_size for PPO actor updates. The ppo_mini_batch_size is a global size across all workers.

---

## Training Pipeline (per step)

| Step | Operation | Batch Size | Description |
|------|-----------|------------|-------------|
| 1 | **gen** | 32 prompts → 256 responses | Generate responses (train_batch_size × rollout.n) |
| 2 | **reward** | 256 responses | Compute rewards for ALL responses |
| 3 | **old_log_prob** | 256 responses | Compute π_θ log probs on all responses |
| 4 | **ref_log_prob** (optional) | 256 responses | Compute reference model log probs |
| 5 | **adv** | 256 responses | Compute GRPO advantages with group normalization |
| 6 | **update_actor** | 16 responses × 16 mini-batches | Split into mini-batches for gradient updates |

---

## Effective Batch Size Calculation

**Total responses (trajectories) per PPO step:**
```
train_batch_size × rollout.n = total_responses
```

**Number of gradient updates per PPO step:**
```
total_responses ÷ ppo_mini_batch_size = num_mini_batches
```

### Example Calculation

With configuration:
- `train_batch_size = 32` (prompts)
- `rollout.n = 8` (responses per prompt)
- `ppo_mini_batch_size = 16` (responses per mini-batch)

| Metric | Value |
|--------|-------|
| Total responses | 32 × 8 = **256 responses** |
| Effective batch size per gradient update | **16 responses** |
| Number of gradient steps per PPO epoch | 256 ÷ 16 = **16 gradient updates** |

---

## Key Insights

1. **Rewards are computed on ALL 256 responses** before mini-batch splitting
2. **GRPO advantage normalization** groups responses by prompt (8 responses per group), computing mean/std within each group
3. **Mini-batch splitting happens inside `update_actor`** — the 256 responses are split into 16 mini-batches of 16 responses each

---

## Code Evidence

**1. Batch is repeated before reward computation** (`verl/trainer/ppo/ray_trainer.py:1606-1609`):
```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**2. Reward computed on full batch** (`verl/trainer/ppo/ray_trainer.py:1627-1633`):
```python
with marked_timer("reward", timing_raw, color="yellow"):
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)  # All 256
        batch = batch.union(batch_reward)
```

**3. GRPO advantage uses group normalization** (`verl/trainer/ppo/core_algos.py:304-318`):
```python
# Groups responses by prompt (uid), computes mean/std per group
for idx in id2score:
    scores_tensor = torch.stack(id2score[idx])  # 8 responses per prompt
    id2mean[idx] = torch.mean(scores_tensor)
    id2std[idx] = torch.std(scores_tensor)
# Normalizes: (score - group_mean) / group_std
```

---

# Evaluation

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.test_freq` | 5 | Run validation every N training steps |
| `trainer.validation_save_freq` | 50 | Save validation generations every N steps |
| `actor_rollout_ref.rollout.val_kwargs.n` | 16 | Number of samples per validation problem |
| `reward.num_workers` | 8 | Number of parallel Ray actors for reward computation |

### Validation Pipeline

```
Validation Set (e.g., 131 problems)
         │
         ▼
    × val_kwargs.n (16 samples/problem)
         │
         ▼
    = 2,096 total generations to evaluate
         │
         ▼
    Split across reward.num_workers (e.g., 16 workers)
         │
         ▼
    ~131 samples per worker
```

### Parallelism Architecture

**Two levels of parallelism:**

1. **Level 1: `reward.num_workers`** (Ray actors)
   - Samples are distributed across N workers
   - Workers run in **parallel** with each other
   - Each worker processes its samples **sequentially** (naive manager) or in parallel (prime manager)

2. **Level 2: Code execution** (inside `code.py`)
   - Each code solution spawns multiprocessing.Process for test cases
   - LiveCodeBench has **40 test cases per problem**
   - All 40 tests run **in parallel** via `p.start()` before collecting results

```
                    ┌─ Worker 1 ─── sample → [40 test processes] → sample → ...
                    ├─ Worker 2 ─── sample → [40 test processes] → sample → ...
Batch of 2096  ────►├─ Worker 3 ─── sample → [40 test processes] → sample → ...
samples              │    ...
                    └─ Worker 16 ── sample → [40 test processes] → sample → ...
```

### CPU Usage Calculation

```
Peak concurrent processes = reward.num_workers × test_cases_per_problem
                         = 16 workers × 40 tests
                         = 640 processes
```

| num_workers | Test Processes | Total Processes | Recommendation |
|-------------|---------------|-----------------|----------------|
| 8 | 40 | 320 | Conservative |
| 16 | 40 | 640 | Balanced (default) |
| 24 | 40 | 960 | Aggressive |
| 32 | 40 | 1,280 | May cause thrashing |

**Rule of thumb:** `num_workers ≈ CPUs / (test_cases × 0.5)` for I/O-bound workloads

### Reward Managers

| Manager | Parallelism | Use Case |
|---------|-------------|----------|
| `naive` | Sequential within worker | Default; good for code execution (avoids nested parallelism) |
| `prime` | 64 processes per worker via ProcessPoolExecutor | Math/text tasks without subprocess spawning |
| `batch` | Batched scoring | For batched reward functions |

**Warning:** Don't use `prime` with code execution tasks — it causes nested parallelism:
```
prime worker → 64 sub-processes → each spawns 40 test processes = OOM!
```



