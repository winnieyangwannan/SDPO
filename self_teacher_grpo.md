# Self-Teacher GRPO: Implementation Plan

## 1. Executive Summary

**Goal:** Enhance GRPO training by providing some rollouts with "privilege information" — the best solution found so far for each question — while keeping other rollouts unprivileged. This creates a curriculum where the model learns both from guided (privileged) and unguided (unprivileged) attempts.

**Key Insight:** Unlike SDPO which uses solutions from the *current batch* for distillation loss computation *after* generation, Self-Teacher GRPO uses solutions from a *persistent cache* (across training steps) and injects them *before* generation to guide the model's response.

**Addressing Train-Test Mismatch:** Privileged prompts optimize a different distribution (`P(response | prompt + hint)`) than inference (`P(response | prompt)`). To mitigate this, we apply a **privilege penalty** that discounts privileged solutions in cache updates and GRPO advantage computation. The penalty increases over training via the `late_ramp` schedule, creating increasing pressure to succeed without hints. The `cache_privileged_fraction` metric tracks whether transfer is happening.

### Comparison with Existing Methods

| Aspect | **GRPO** | **SDPO** | **Self-Teacher GRPO** (proposed) |
|--------|----------|----------|--------------------------------|
| Solution source | None | Current batch (same step) | Persistent cache (previous steps) |
| When privilege applied | N/A | After generation (distillation loss) | **Before generation** (prompt augmentation) |
| Loss function | GRPO advantage | GRPO + KL distillation | **Pure GRPO** (no distillation) |
| Privilege pattern | N/A | Opportunistic (depends on batch successes) | **Conditional + Fractional** |
| Train-test mismatch mitigation | N/A | N/A | **Privilege penalty** (unprivileged solutions favored) |

---

## 2. System Architecture

### 2.1 High-Level Flow

**IMPORTANT: Tokenization Timing Verification**

The privilege injection works because tokenization happens **inside** `generate_sequences()`, not during `repeat()`. Verified data flow:

```
batch.non_tensor_batch["raw_prompt"]  (NOT tokenized, just message dicts)
    ↓
repeat() → duplicates raw_prompt via np.repeat()  (still NOT tokenized)
    ↓
_inject_privilege_info() → modifies raw_prompt[idx]  ← OUR INJECTION POINT
    ↓
generate_sequences() → for each sample:
    kwargs["raw_prompt"] = batch.non_tensor_batch["raw_prompt"][i]
    _run_agent_loop(**kwargs)
        ↓
    SingleTurnAgentLoop.run():
        messages = list(kwargs["raw_prompt"])  ← reads modified value
        prompt_ids = await self.apply_chat_template(messages)  ← TOKENIZATION HERE
```

**Code references:**
- `verl/experimental/agent_loop/single_turn_agent_loop.py:37` - reads `raw_prompt`
- `verl/experimental/agent_loop/single_turn_agent_loop.py:45` - tokenizes via `apply_chat_template`

**Coupling Warning:** This approach relies on the agent loop architecture. Alternative rollout backends that pre-tokenize prompts would break this approach.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Training Step (Global Step t)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Sample 32 prompts from dataloader                                        │
│                       ↓                                                      │
│  2. Repeat 8x → 256 samples (32 questions × 8 rollouts each)                │
│                       ↓                                                      │
│  3. ★ INJECT PRIVILEGE INFO ★                                               │
│     For each question group:                                                 │
│       - Check if cache has solution with score >= threshold                  │
│       - If yes: modify `privilege_fraction` of rollouts' prompts             │
│       - If no: all rollouts remain unprivileged                              │
│                       ↓                                                      │
│  4. Generate 256 responses (some with privilege info in prompt)              │
│                       ↓                                                      │
│  5. Compute rewards for all 256 responses                                    │
│                       ↓                                                      │
│  6. ★ UPDATE CACHE ★                                                        │
│     For each response: if score > cached score, update cache                 │
│                       ↓                                                      │
│  7. Compute GRPO advantages (privileged & unprivileged compete together)     │
│                       ↓                                                      │
│  8. Split into mini-batches, perform gradient updates                        │
│                       ↓                                                      │
│  9. Log metrics (separate for privileged/unprivileged + combined)            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Layout After `repeat(interleave=True)`

```
Original batch:     [Q0, Q1, Q2, ..., Q31]  (32 questions)

After repeat(8):    [Q0, Q0, Q0, Q0, Q0, Q0, Q0, Q0,   ← indices 0-7   (question 0)
                     Q1, Q1, Q1, Q1, Q1, Q1, Q1, Q1,   ← indices 8-15  (question 1)
                     Q2, Q2, Q2, Q2, Q2, Q2, Q2, Q2,   ← indices 16-23 (question 2)
                     ...]

With privilege (fraction=0.5, interleaved):
                    [Q0✓, Q0✗, Q0✓, Q0✗, Q0✓, Q0✗, Q0✓, Q0✗,   ← 4 privileged, 4 not
                     Q1✓, Q1✗, Q1✓, Q1✗, Q1✓, Q1✗, Q1✓, Q1✗,
                     ...]
                    (only if question has cached solution with score >= threshold)
```

---

## 3. Algorithm Description

### 3.1 Best Solutions Cache

**Purpose:** Persistently store the best solution found for each question across all training steps.

**Metric Clarification:**
| Metric | Range | Source | Description |
|--------|-------|--------|-------------|
| `acc` | 0.0 to 1.0 (continuous) | `non_tensor_batch["acc"]` (after reward) | Fraction of test cases passed |
| `score` (reward) | 0 or 1 (binary) | `batch.batch["rm_scores"]` | 1.0 only if ALL tests pass, else 0.0 |

**Note:** The `acc` field is populated by `compute_score()` and flattened into `non_tensor_batch` by the agent loop after reward computation. It is NOT available before generation.

The cache stores **acc** (continuous), and `accuracy_threshold` filters which solutions qualify for privilege injection.

**Structure:**
```python
self.best_solutions_cache: dict[str, dict] = {
    "42": {
        "response": "def solve(n, k):\n    # Solution code...",
        "acc": 0.85,                            # Fraction of tests passed (continuous)
        "step": 50,
        "was_privileged": False,               # Whether this solution came from a privileged rollout
        "description_hash": "a1b2c3d4...",      # MD5 of first 500 chars
        "description_preview": "You are given..."  # First 100 chars for debugging
    },
    ...
}
```

**Persistence:**
- Saved to: `{checkpoint_dir}/best_solutions_cache.json`
- Loaded on trainer initialization (supports resume)
- Saved on every checkpoint save

**Data Field Reference (Verified Against Codebase):**

| Field | Location | When Available | Source |
|-------|----------|----------------|--------|
| `index` | `non_tensor_batch["index"]` | Always (from dataset) | `verl/utils/dataset/rl_dataset.py:377` |
| `description` | `non_tensor_batch["extra_info"][i]["description"]` | Always (from dataset) | `data/preprocess.py:76` |
| `acc` | `non_tensor_batch["acc"]` | After reward computation | `verl/utils/reward_score/feedback/code.py:862` → flattened by agent loop |
| `score` | `batch.batch["rm_scores"]` (tensor) | After reward computation | `verl/utils/reward_score/feedback/code.py:863` |
| `raw_prompt` | `non_tensor_batch["raw_prompt"]` | Always (from dataset) | `verl/utils/dataset/rl_dataset.py:363` |

**Important:** The `acc` field uses key `"acc"`, NOT `"accuracy"`. This is the return key from `compute_score()` in the reward function.

**`raw_prompt` Structure (Verified):**
```python
# raw_prompt is a list of message dicts:
raw_prompt = [
    {"role": "system", "content": "..."},  # optional
    {"role": "user", "content": "You are given..."}  # last element is always user message
]

# Access patterns:
raw_prompt[-1]["content"]  # user message content (the problem description)
raw_prompt[:-1]            # system messages (everything except last)
```

Code reference: `verl/trainer/ppo/ray_trainer.py:676` shows `system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]`

### 3.2 Privilege Assignment Algorithm

```
ALGORITHM: AssignPrivilege(batch, cache, config)
───────────────────────────────────────────────────────────────────────────────
INPUT:
  - batch: DataProto with 256 samples (32 questions × 8 rollouts)
  - cache: best_solutions_cache
  - config: {privilege_fraction, accuracy_threshold, solution_template}

OUTPUT:
  - modified batch with some prompts augmented
  - privilege_mask: tensor[256] of bool

PROCEDURE:
  privilege_mask ← zeros(256, dtype=bool)
  rollout_n ← config.rollout.n  # 8
  num_privileged ← floor(rollout_n × privilege_fraction)  # 4 if fraction=0.5
  
  FOR group_start IN range(0, 256, rollout_n):  # 0, 8, 16, ...
    question_idx ← batch.non_tensor_batch["index"][group_start]
    
    # Check qualification
    IF question_idx NOT IN cache:
      CONTINUE  # No cached solution → all unprivileged
    
    cached ← cache[question_idx]
    IF cached["acc"] < accuracy_threshold:
      CONTINUE  # Below threshold → all unprivileged
    
    # Verify question match (safety check)
    current_description ← batch.non_tensor_batch["extra_info"][group_start]["description"]
    IF MD5(current_description[:500]) ≠ cached["description_hash"]:
      LOG WARNING "Question mismatch for index {question_idx}"
      CONTINUE
    
    # Assign privilege to interleaved positions: 0, 2, 4, 6
    FOR j IN range(num_privileged):
      idx ← group_start + j × 2  # Interleave pattern
      IF idx < group_start + rollout_n:
        privilege_mask[idx] ← True
        batch.non_tensor_batch["raw_prompt"][idx] ← 
          AddPrivilegeToPrompt(batch.non_tensor_batch["raw_prompt"][idx], cached["response"])
  
  RETURN batch, privilege_mask
───────────────────────────────────────────────────────────────────────────────
```

### 3.3 Prompt Modification

**Template (follows SDPO format):**
```python
reprompt_template = (
    "{prompt}"
    "\n\nCorrect solution:\n\n"
    "{successful_previous_attempt}\n\n"
    "Correctly solve the original question.\n"
)
```

**Example:**
```
# Original prompt (unprivileged):
"You are given three integer sequences of length N..."

# Modified prompt (privileged):
"You are given three integer sequences of length N...

Correct solution:

def solve(n, sequences):
    # ... working solution code ...
    return result

Correctly solve the original question."
```

### 3.4 Cache Update Algorithm

```
ALGORITHM: UpdateCache(batch, reward_tensor, cache, tokenizer)
───────────────────────────────────────────────────────────────────────────────
INPUT:
  - batch: DataProto with 256 samples
  - reward_tensor: tensor[256, response_length] of rewards
  - cache: best_solutions_cache (mutable)
  - tokenizer: for decoding responses

OUTPUT:
  - metrics: {num_new, num_updates, total_cached}

PROCEDURE:
  # Extract acc from non_tensor_batch (continuous: 0.0 to 1.0)
  # Note: "acc" is populated by compute_score() and flattened into non_tensor_batch
  # It is available AFTER reward computation, NOT before generation
  accuracies ← batch.non_tensor_batch["acc"]  # numpy array
  num_new ← 0
  num_updates ← 0
  
  FOR idx IN range(256):
    question_idx ← str(batch.non_tensor_batch["index"][idx])
    acc ← accuracies[idx]
    
    # Get question description for verification (from extra_info dict)
    extra_info ← batch.non_tensor_batch["extra_info"][idx]
    description ← extra_info.get("description", "")
    
    # Get privilege status and compute penalty
    is_privileged ← privilege_mask[idx]
    penalty ← get_cache_penalty(global_steps, total_steps, config) IF is_privileged ELSE 0
    effective_acc ← acc - penalty
    
    should_update ← False
    IF question_idx NOT IN cache:
      should_update ← True
      num_new += 1
    ELIF effective_acc > cache[question_idx]["acc"]:
      # Privileged solutions must exceed cached acc by penalty margin
      should_update ← True
      num_updates += 1
    
    IF should_update:
      cache[question_idx] ← {
        "response": tokenizer.decode(batch.batch["responses"][idx], skip_special_tokens=True),
        "acc": acc,  # Store raw acc, not penalized
        "step": global_steps,
        "was_privileged": is_privileged,
        "description_hash": MD5(description[:500]),
        "description_preview": description[:100]
      }
  
  RETURN {
    "num_new": num_new,
    "num_updates": num_updates,
    "total_cached": len(cache)
  }
───────────────────────────────────────────────────────────────────────────────
```

### 3.5 GRPO Advantage Calculation

**No modification to GRPO algorithm.** Privileged and unprivileged samples compete together in the same group:

```python
# Standard GRPO advantage (unchanged)
for uid in unique_uids:
    group_scores = scores[batch.non_tensor_batch["uid"] == uid]
    group_mean = group_scores.mean()
    group_std = group_scores.std()
    advantages[uid_mask] = (scores[uid_mask] - group_mean) / (group_std + epsilon)
```

**Implication:** If privileged samples consistently score higher, they will:
1. Raise the group mean → unprivileged get lower/negative advantages
2. Model learns that "having a hint helps" but also learns from unprivileged attempts

### 3.6 Privilege Penalty: Addressing Train-Test Mismatch

**Problem:** Privileged samples optimize a different conditional distribution than inference:
- Training with hint: `P(response | prompt + hint)`
- Inference: `P(response | prompt)`

There's no guarantee gradients from hint-assisted generation transfer to hint-free inference. If the model learns "copy the solution when shown one" rather than "internalize the reasoning pattern," the hints are useless at test time.

**Solution:** Discount privileged solutions so unprivileged ones are preferred when accuracy is equal or close.

```
effective_acc = acc - (penalty if is_privileged else 0)
```

| Sample | Raw Acc | Privileged? | Penalty=0.05 | Effective Acc |
|--------|---------|-------------|--------------|---------------|
| A | 0.8 | No | - | **0.8** |
| B | 0.8 | Yes | 0.05 | **0.75** |
| C | 1.0 | Yes | 0.05 | **0.95** |
| D | 0.6 | No | - | **0.6** |

Sample A (unprivileged) now beats Sample B (privileged) in comparisons, even though raw accuracy is equal.

**Penalty Application Points:**

| Context | Effect | Rationale |
|---------|--------|----------|
| **Cache update** | Privileged solutions need higher raw acc to enter/update cache | Curates cache toward hint-free solutions over time |
| **GRPO advantage** | Privileged samples get lower effective scores in advantage computation | Reduces training weight on hint-assisted successes |

**Choosing the Penalty Value:**

The penalty needs to be large enough to always break ties in favor of unprivileged, but small enough not to reject a genuinely better (higher accuracy) privileged solution:

| Penalty | Effect |
|---------|--------|
| Too small (< accuracy resolution) | Doesn't reliably break ties |
| Just right (~0.02–0.05) | Unprivileged always wins ties; better privileged still wins |
| Too large (> typical accuracy gap) | Rejects genuinely better privileged solutions, cache degrades |

Since accuracy is continuous and typical gaps between solutions are on the order of 0.05–0.10 (one or two test cases), a penalty of around 0.02–0.05 seems right.

**Adaptive penalty:** `penalty = 1 / num_test_cases` equals exactly the resolution of one test case. Then the rule becomes: "an unprivileged solution that passes the same number of tests as a privileged one always wins."

### 3.7 Two-Axis Decay: Privilege Fraction + Penalty Schedule

The plan already decays `privilege_fraction` (how many rollouts get hints). Adding a penalty schedule creates a second, independent pressure:

| Decay Axis | Controls | Effect |
|------------|----------|--------|
| `privilege_fraction` decay | How often hints are injected | Fewer rollouts see hints over time |
| `penalty` increase | How strongly unprivileged solutions are preferred in cache | Cache becomes increasingly biased toward hint-free solutions |

These two knobs are complementary. `privilege_fraction` controls the **generation side**; penalty controls the **cache curation side**. Together they squeeze privilege from both ends.

**Penalty Schedule Options:**

```python
def get_cache_penalty(global_step, total_steps, config):
    t = global_step / total_steps  # normalized time in [0, 1]
    p_min = config.penalty_min     # e.g., 0.01  (early: almost no penalty)
    p_max = config.penalty_max     # e.g., 0.10  (late: strong preference for unprivileged)

    schedule = config.penalty_schedule

    if schedule == "linear":
        return p_min + (p_max - p_min) * t

    elif schedule == "cosine":
        # slow start, accelerates in the middle, plateaus at end
        import math
        return p_min + (p_max - p_min) * (1 - math.cos(math.pi * t)) / 2

    elif schedule == "late_ramp":
        # flat early, sharp increase in final third
        if t < 0.66:
            return p_min
        else:
            return p_min + (p_max - p_min) * (t - 0.66) / 0.34

    return p_min  # "none" or unrecognized
```

**The `late_ramp` schedule explained:**

```
penalty
p_max |                          ████
      |                      ████
      |                  ████
p_min |██████████████████
      └─────────────────────────────── t
      0        0.66              1.0
      └── flat ──┘└── ramp ──┘
```

This schedule is specifically designed to align with training phases:
- **First 2/3 (t < 0.66):** Penalty stays low. Let the cache populate freely — privileged solutions are fine when you need *something* in the cache to bootstrap hints.
- **Final 1/3 (t ≥ 0.66):** Penalty ramps to p_max. Cache is mature; start favoring unprivileged solutions to verify real learning happened.

**Interaction with Natural Training Phases:**

| Phase | Time | Privilege Fraction | Penalty | Goal |
|-------|------|-------------------|---------|------|
| **Early** | t ≈ 0 | High but ineffective (cache empty) | Low (p_min) | Populate cache with anything useful |
| **Mid** | t ≈ 0.5 | Active (cache has coverage) | Moderate | Transfer from hinted to unhinted |
| **Late** | t ≈ 1.0 | → 0 (fewer hints) | High (p_max) | Cache reflects true model capability |

The two decays reinforce each other to produce a clean late-training state.

---

## 4. Configuration

### 4.1 New Config File: `self_teacher_grpo.yaml`

```yaml
defaults:
  - ppo_trainer
  - user
  - _self_

max_model_len: 18944  # Larger to accommodate solution in prompt

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 16
  rollout:
    n: 8
    calculate_log_probs: True

algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: False
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0
  
  # NEW: Self-teacher configuration
  self_teacher:
    enable: true
    privilege_fraction: 0.5           # Initial fraction of rollouts that get privilege
    privilege_fraction_decay: linear  # Decay schedule: "linear" or "none"
    # privilege_fraction decays linearly from privilege_fraction → 0 over total training steps.
    # Effective fraction is further modulated by cache coverage (questions with qualifying
    # cached solutions), giving a natural bell-shaped curve: near-zero early (empty cache),
    # rising as cache populates, then declining as configured fraction decays.
    accuracy_threshold: 0.8           # Minimum accuracy (fraction of tests passed) to use cached solution
                                      # Note: accuracy is continuous [0,1], not binary score
    
    # Privilege penalty: discount privileged solutions to favor unprivileged
    privilege_penalty:
      enable: true
      penalty_min: 0.02               # Early training: minimal penalty
      penalty_max: 0.10               # Late training: strong preference for unprivileged
      schedule: late_ramp             # "none", "linear", "cosine", or "late_ramp"
      # late_ramp: flat at p_min for first 66% of training, then ramps to p_max
      apply_to_grpo: true             # Also apply penalty to GRPO advantage computation
    
    solution_template: |

      Correct solution:

      {successful_previous_attempt}

    reprompt_template: "{prompt}{solution}Correctly solve the original question.\n"

data:
  train_batch_size: 32
```

### 4.2 Run Script: `run_self_teacher_grpo.sh`

Key parameters:
```bash
CONFIG_NAME="self_teacher_grpo"
BASE_JOB_NAME="SELF_TEACHER_GRPO"

# Self-teacher specific args
ARGS="...
algorithm.self_teacher.enable=true \
algorithm.self_teacher.privilege_fraction=0.5 \
algorithm.self_teacher.privilege_fraction_decay=linear \
algorithm.self_teacher.accuracy_threshold=0.8 \
..."
```

---

## 5. Implementation Details

### 5.1 Code Location: `verl/trainer/ppo/ray_trainer.py`

**Existing Methods (to be MODIFIED):**

| Method | Current State | Required Modification |
|--------|---------------|----------------------|
| `_update_best_solutions_cache()` | Uses `score` (binary) | Change to use `acc` (continuous); add `description_hash` and `description_preview` |
| `_load_best_solutions_cache()` | Already exists | No change needed |
| `_save_best_solutions_cache()` | Already exists | No change needed |
| `_get_best_solutions_cache_path()` | Already exists | No change needed |
| `__init__` | Initializes `best_solutions_cache = {}` | Already exists, no change needed |

**New Methods to Add:**

| Method | Type | Description |
|--------|------|-------------|
| `_get_current_privilege_fraction()` | **New** | Compute decayed privilege fraction for current step |
| `_get_cache_penalty()` | **New** | Compute penalty for privileged solutions at current step |
| `_apply_privilege_penalty()` | **New** | Apply penalty to scores before GRPO advantage computation |
| `_inject_privilege_info()` | **New** | Assign and inject privilege before generation |
| `_should_give_privilege()` | **New** | Check if question qualifies for privilege |
| `_add_privilege_to_prompt()` | **New** | Modify raw_prompt with cached solution |
| `_verify_question_match()` | **New** | Verify description hash matches |
| `_compute_self_teacher_metrics()` | **New** | Compute privilege/unprivilege metrics |

**Methods to Modify:**

| Method | Required Change |
|--------|----------------|
| `fit()` | Call `_inject_privilege_info()` after `repeat()` but before `generate_sequences()` |

### 5.2 Key Code Snippets

#### 5.2.1 Privilege Fraction Decay

```python
def _get_current_privilege_fraction(self) -> float:
    """
    Compute the privilege fraction at the current training step.

    Linear decay: fraction(t) = fraction_0 * max(0, 1 - t / total_steps)

    Note: the *effective* fraction is further reduced by cache coverage
    (not all questions have qualifying cached solutions), which naturally
    provides a warmup: near-zero early when cache is empty, rising as
    the cache populates, then declining as the configured fraction decays.
    This produces a bell-shaped effective privilege curve without any
    explicit warmup logic.
    """
    self_teacher_cfg = self.config.algorithm.get("self_teacher", {})
    fraction_0 = self_teacher_cfg.get("privilege_fraction", 0.5)
    decay = self_teacher_cfg.get("privilege_fraction_decay", "none")

    if decay == "linear":
        total_steps = self.total_training_steps  # stored on instance, NOT self.config.trainer.total_training_steps
        t = self.global_steps
        return fraction_0 * max(0.0, 1.0 - t / total_steps)

    return fraction_0  # "none" or unrecognized → constant
```

#### 5.2.2 Privilege Penalty Computation

```python
def _get_cache_penalty(self) -> float:
    """
    Compute the privilege penalty at the current training step.
    
    Privileged solutions must exceed cached acc by this margin to enter cache.
    This increasingly favors unprivileged solutions as training progresses.
    """
    self_teacher_cfg = self.config.algorithm.get("self_teacher", {})
    penalty_cfg = self_teacher_cfg.get("privilege_penalty", {})
    
    if not penalty_cfg.get("enable", True):
        return 0.0
    
    p_min = penalty_cfg.get("penalty_min", 0.02)
    p_max = penalty_cfg.get("penalty_max", 0.10)
    schedule = penalty_cfg.get("schedule", "late_ramp")
    
    total_steps = self.total_training_steps
    t = self.global_steps / total_steps if total_steps > 0 else 0.0
    
    if schedule == "linear":
        return p_min + (p_max - p_min) * t
    
    elif schedule == "cosine":
        import math
        return p_min + (p_max - p_min) * (1 - math.cos(math.pi * t)) / 2
    
    elif schedule == "late_ramp":
        # Flat early, sharp increase in final third
        if t < 0.66:
            return p_min
        else:
            return p_min + (p_max - p_min) * (t - 0.66) / 0.34
    
    return p_min  # "none" or unrecognized


def _apply_privilege_penalty(
    self, 
    scores: torch.Tensor,
    privilege_mask: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """
    Discount scores from privileged samples before advantage computation.
    
    This creates incentive to succeed without hints:
    - Same acc without hint > same acc with hint
    - Model learns to internalize reasoning rather than rely on hints
    """
    effective_scores = scores.clone()
    effective_scores[privilege_mask] -= penalty
    return effective_scores
```

#### 5.2.3 Privilege Injection (before generation)

```python
def _inject_privilege_info(self, batch: DataProto) -> tuple[torch.Tensor, int]:
    """
    Inject privilege information into qualifying prompts before generation.
    
    IMPORTANT: This method modifies `batch.non_tensor_batch["raw_prompt"]` which
    must be tokenized AFTER this call. This is verified to work with the agent loop
    architecture where tokenization happens inside `generate_sequences()` via
    `apply_chat_template()`. See Section 2.1 for the verified data flow.
    
    Returns:
        tuple of (privilege_mask, num_questions_with_privilege):
            privilege_mask: Boolean tensor indicating which samples got privilege
            num_questions_with_privilege: Count of questions that had qualifying cached solutions
    """
    self_teacher_cfg = self.config.algorithm.get("self_teacher", {})
    if not self_teacher_cfg.get("enable", False):
        return torch.zeros(len(batch), dtype=torch.bool), 0
    
    # DEFENSIVE CHECK: Ensure prompts are not already tokenized
    # If input_ids exists and has non-trivial content, tokenization already happened
    if batch.batch is not None and "input_ids" in batch.batch:
        if batch.batch["input_ids"].numel() > 0:
            raise RuntimeError(
                "Self-Teacher GRPO: input_ids already exists in batch before privilege injection. "
                "This indicates prompts were pre-tokenized, which would cause privilege injection "
                "to have no effect. Ensure you are using the agent loop rollout backend."
            )
    
    rollout_n = self.config.actor_rollout_ref.rollout.n
    privilege_fraction = self._get_current_privilege_fraction()  # decayed value
    threshold = self_teacher_cfg.get("accuracy_threshold", 0.8)
    num_privileged = int(rollout_n * privilege_fraction)
    
    batch_size = len(batch)
    privilege_mask = torch.zeros(batch_size, dtype=torch.bool)
    
    # Get question indices
    if "index" in batch.non_tensor_batch:
        question_indices = batch.non_tensor_batch["index"]
    else:
        extra_infos = batch.non_tensor_batch.get("extra_info", [{}] * batch_size)
        question_indices = [str(info.get("index", i)) for i, info in enumerate(extra_infos)]
    
    num_questions_with_privilege = 0
    
    for group_start in range(0, batch_size, rollout_n):
        question_idx = str(question_indices[group_start])
        
        # Check if this question qualifies for privilege
        if not self._should_give_privilege(question_idx, threshold):
            continue
        
        # Verify question match
        if not self._verify_question_match(batch, group_start, question_idx):
            continue
        
        num_questions_with_privilege += 1
        cached_solution = self.best_solutions_cache[question_idx]["response"]
        
        # Assign privilege to interleaved positions
        for j in range(num_privileged):
            idx = group_start + j * 2  # Interleave: 0, 2, 4, 6
            if idx < group_start + rollout_n:
                privilege_mask[idx] = True
                self._add_privilege_to_prompt(batch, idx, cached_solution, self_teacher_cfg)
    
    # Log sample privileged prompt for debugging
    if privilege_mask.any() and self.global_steps % 10 == 0:
        first_privileged_idx = privilege_mask.nonzero()[0].item()
        print(f"[Step {self.global_steps}] Sample privileged prompt:\n"
              f"{batch.non_tensor_batch['raw_prompt'][first_privileged_idx][-1]['content'][:500]}...")
    
    return privilege_mask, num_questions_with_privilege

def _should_give_privilege(self, question_idx: str, threshold: float) -> bool:
    """Check if question qualifies for privilege info based on accuracy threshold."""
    if question_idx not in self.best_solutions_cache:
        return False
    return self.best_solutions_cache[question_idx]["acc"] >= threshold

def _verify_question_match(self, batch: DataProto, idx: int, question_idx: str) -> bool:
    """Verify the question description matches the cached entry."""
    import hashlib
    
    cached = self.best_solutions_cache.get(question_idx)
    if not cached or "description_hash" not in cached:
        return True  # No verification possible, allow
    
    extra_info = batch.non_tensor_batch.get("extra_info", [{}] * len(batch))
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.tolist()
    
    current_description = ""
    if idx < len(extra_info) and isinstance(extra_info[idx], dict):
        current_description = extra_info[idx].get("description", "")
    
    if not current_description:
        return True  # No description to verify, allow
    
    current_hash = hashlib.md5(current_description[:500].encode()).hexdigest()
    if current_hash != cached["description_hash"]:
        print(f"WARNING: Question {question_idx} description mismatch!")
        print(f"  Cached preview: {cached.get('description_preview', 'N/A')}")
        print(f"  Current preview: {current_description[:100]}")
        return False
    
    return True

def _add_privilege_to_prompt(
    self, 
    batch: DataProto, 
    idx: int, 
    solution: str, 
    config: dict
) -> None:
    """Modify raw_prompt at idx to include the cached solution."""
    solution_template = config.get("solution_template", 
        "\n\nCorrect solution:\n\n{successful_previous_attempt}\n\n")
    reprompt_template = config.get("reprompt_template",
        "{prompt}{solution}Correctly solve the original question.\n")
    
    # Get current prompt
    raw_prompt = batch.non_tensor_batch["raw_prompt"][idx]
    
    # raw_prompt is list of messages: [{"role": "system", ...}, {"role": "user", "content": "..."}]
    # Modify the last user message
    original_content = raw_prompt[-1]["content"]
    solution_section = solution_template.format(successful_previous_attempt=solution)
    new_content = reprompt_template.format(prompt=original_content, solution=solution_section)
    
    # Create modified prompt (don't mutate original)
    new_raw_prompt = raw_prompt[:-1] + [{"role": "user", "content": new_content}]
    batch.non_tensor_batch["raw_prompt"][idx] = new_raw_prompt
```

#### 5.2.4 Updated Cache Method (with penalty and privilege tracking)

```python
def _update_best_solutions_cache(
    self, 
    batch: DataProto, 
    reward_tensor: torch.Tensor,
    privilege_mask: torch.Tensor = None
) -> dict[str, float]:
    """Update the persistent cache with the best solution for each question.
    
    Uses acc (continuous, 0.0-1.0) rather than binary score for comparison,
    allowing solutions that pass most tests to qualify for privilege injection.
    
    Privileged solutions are penalized: they must exceed the cached acc by
    the current penalty margin to displace an existing entry. This increasingly
    favors unprivileged solutions as training progresses.
    
    Note: "acc" is populated by compute_score() and flattened into non_tensor_batch
    by the agent loop postprocessing. It is available after reward computation.
    """
    import hashlib
    
    responses = batch.batch["responses"]
    
    # Get question indices from non_tensor_batch (populated by dataset)
    question_indices = batch.non_tensor_batch["index"]
    
    # Get acc from non_tensor_batch (populated by compute_score after reward computation)
    # Key is "acc", NOT "accuracy"
    accuracies = batch.non_tensor_batch.get("acc", np.zeros(len(batch)))
    if isinstance(accuracies, list):
        accuracies = np.array(accuracies)
    
    # Get extra_info for descriptions (for verification)
    extra_infos = batch.non_tensor_batch.get("extra_info", [{}] * len(batch))
    if isinstance(extra_infos, np.ndarray):
        extra_infos = extra_infos.tolist()
    
    # Get current penalty for privileged solutions
    penalty = self._get_cache_penalty()
    
    num_updates = 0
    num_new = 0
    
    for idx in range(len(batch)):
        question_idx = str(question_indices[idx])
        
        # Get acc (continuous) from non_tensor_batch
        acc = float(accuracies[idx]) if idx < len(accuracies) else 0.0
        
        # Check if this sample was privileged
        is_privileged = bool(privilege_mask[idx]) if privilege_mask is not None else False
        
        # Apply penalty to privileged samples for cache comparison
        # Unprivileged solutions win ties; privileged must exceed by penalty margin
        effective_acc = acc - (penalty if is_privileged else 0.0)
        
        # Get description from extra_info
        extra_info = extra_infos[idx] if idx < len(extra_infos) and isinstance(extra_infos[idx], dict) else {}
        description = extra_info.get("description", "")
        
        # Check if should update (compare effective_acc against cached raw acc)
        if question_idx not in self.best_solutions_cache:
            num_new += 1
            should_update = True
        elif effective_acc > self.best_solutions_cache[question_idx]["acc"]:
            num_updates += 1
            should_update = True
        else:
            should_update = False
        
        if should_update:
            self.best_solutions_cache[question_idx] = {
                "response": self.tokenizer.decode(responses[idx], skip_special_tokens=True),
                "acc": acc,  # Store raw acc, not penalized
                "step": self.global_steps,
                "was_privileged": is_privileged,  # Track provenance
                "description_hash": hashlib.md5(description[:500].encode()).hexdigest() if description else "",
                "description_preview": description[:100] if description else "",
            }
    
    return {
        "best_solutions_cache/num_new": num_new,
        "best_solutions_cache/num_updates": num_updates,
        "best_solutions_cache/total_cached": len(self.best_solutions_cache),
    }
```

#### 5.2.5 Integration in `fit()` Loop

```python
# In fit() method, after batch repeat and before generation:

# Repeat batch for multiple rollouts per prompt
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
)

# NEW: Inject privilege info for self-teacher GRPO
self_teacher_cfg = self.config.algorithm.get("self_teacher", {})
if self_teacher_cfg.get("enable", False):
    privilege_mask, num_questions_with_privilege = self._inject_privilege_info(gen_batch_output)
    # IMPORTANT: Store as instance variables, NOT on batch
    # because generate_sequences() returns a NEW DataProto and doesn't preserve batch data
    self._current_privilege_mask = privilege_mask
    self._current_num_questions_with_privilege = num_questions_with_privilege
else:
    self._current_privilege_mask = None
    self._current_num_questions_with_privilege = 0

# Generate sequences (returns NEW DataProto - any data stored on input batch is lost)
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

# ... later, after reward computation but BEFORE GRPO advantage calculation:
if self._current_privilege_mask is not None:
    penalty_cfg = self_teacher_cfg.get("privilege_penalty", {})
    if penalty_cfg.get("apply_to_grpo", True):
        # Apply privilege penalty to scores before advantage computation
        penalty = self._get_cache_penalty()
        scores_for_advantage = self._apply_privilege_penalty(
            scores, self._current_privilege_mask, penalty
        )
    else:
        scores_for_advantage = scores
    
    # Update cache with penalty-aware comparison (pass privilege_mask)
    cache_metrics = self._update_best_solutions_cache(
        batch, reward_tensor, self._current_privilege_mask
    )
    metrics.update(cache_metrics)
    
    # Compute self-teacher metrics
    metrics.update(self._compute_self_teacher_metrics(
        batch, reward_tensor, self._current_privilege_mask,
        self._current_num_questions_with_privilege,
        self_teacher_cfg.get("accuracy_threshold", 0.8)
    ))
```

**IMPORTANT Implementation Note:** Data stored on `gen_batch_output.batch` before `generate_sequences()` is **lost** because the function returns a new `DataProto`. Store `privilege_mask` as `self._current_privilege_mask` instead.

---

## 6. Metrics

### 6.1 New Metrics to Track

| Metric | Description |
|--------|-------------|
| `self_teacher/privilege_fraction_actual` | Actual fraction of samples that got privilege (may be < configured if cache empty) |
| `self_teacher/privilege_fraction_configured` | The decayed configured fraction at this step (useful to track decay) |
| `self_teacher/questions_with_privilege` | Number of questions that had qualifying cached solutions |
| `self_teacher/privileged_mean_reward` | Mean reward of privileged samples |
| `self_teacher/unprivileged_mean_reward` | Mean reward of unprivileged samples |
| `self_teacher/privileged_mean_acc` | Mean acc of privileged samples |
| `self_teacher/unprivileged_mean_acc` | Mean acc of unprivileged samples |
| `self_teacher/privileged_above_threshold` | Fraction of privileged samples with acc >= threshold |
| `self_teacher/unprivileged_above_threshold` | Fraction of unprivileged samples with acc >= threshold |
| `self_teacher/privilege_penalty` | Current penalty value (for tracking schedule) |
| `best_solutions_cache/num_new` | New questions added to cache this step |
| `best_solutions_cache/num_updates` | Existing questions improved this step |
| `best_solutions_cache/total_cached` | Total questions in cache |
| `best_solutions_cache/mean_cached_acc` | Mean acc of cached solutions |
| `best_solutions_cache/privileged_fraction` | Fraction of cache entries from privileged rollouts |

### 6.2 Metrics Computation

```python
def _compute_self_teacher_metrics(
    self, 
    batch: DataProto, 
    reward_tensor: torch.Tensor,
    privilege_mask: torch.Tensor,
    num_questions_with_privilege: int,
    threshold: float
) -> dict[str, float]:
    """Compute metrics for self-teacher GRPO."""
    # Get acc from non_tensor_batch (continuous metric)
    # Key is "acc", populated by compute_score() after reward computation
    accuracies = batch.non_tensor_batch.get("acc", np.zeros(len(batch)))
    if isinstance(accuracies, list):
        accuracies = np.array(accuracies)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    
    # Also get binary reward for reference
    seq_scores = reward_tensor.sum(dim=-1).detach().cpu()
    
    privileged_accs = accuracies[privilege_mask]
    unprivileged_accs = accuracies[~privilege_mask]
    privileged_scores = seq_scores[privilege_mask]
    unprivileged_scores = seq_scores[~privilege_mask]
    
    metrics = {
        "self_teacher/privilege_fraction_actual": privilege_mask.float().mean().item(),
        "self_teacher/privilege_fraction_configured": self._get_current_privilege_fraction(),
        "self_teacher/questions_with_privilege": num_questions_with_privilege,
        # Acc metrics (continuous)
        "self_teacher/privileged_mean_acc": privileged_accs.mean().item() if len(privileged_accs) > 0 else 0.0,
        "self_teacher/unprivileged_mean_acc": unprivileged_accs.mean().item() if len(unprivileged_accs) > 0 else 0.0,
        "self_teacher/privileged_above_threshold": (privileged_accs >= threshold).float().mean().item() if len(privileged_accs) > 0 else 0.0,
        "self_teacher/unprivileged_above_threshold": (unprivileged_accs >= threshold).float().mean().item() if len(unprivileged_accs) > 0 else 0.0,
        # Binary reward metrics (for reference)
        "self_teacher/privileged_mean_reward": privileged_scores.mean().item() if len(privileged_scores) > 0 else 0.0,
        "self_teacher/unprivileged_mean_reward": unprivileged_scores.mean().item() if len(unprivileged_scores) > 0 else 0.0,
    }
    
    # Cache statistics
    if self.best_solutions_cache:
        cached_accs = [v["acc"] for v in self.best_solutions_cache.values()]
        metrics["best_solutions_cache/mean_cached_acc"] = sum(cached_accs) / len(cached_accs)
        
        # Track how many cache entries came from privileged rollouts
        # This should decline over training if transfer is working
        num_privileged_entries = sum(
            1 for v in self.best_solutions_cache.values() 
            if v.get("was_privileged", False)
        )
        metrics["best_solutions_cache/privileged_fraction"] = (
            num_privileged_entries / len(self.best_solutions_cache)
        )
    
    # Current penalty value
    metrics["self_teacher/privilege_penalty"] = self._get_cache_penalty()
    
    return metrics
```

---

## 7. Edge Cases & Error Handling

| Scenario | Handling |
|----------|----------|
| Cache is empty (step 0) | All samples unprivileged; cache populates as training progresses |
| Question not in cache | All rollouts for that question are unprivileged |
| Cached acc < threshold | All rollouts for that question are unprivileged |
| Description mismatch | Log warning, skip privilege for that question |
| `rollout_n` is odd with `privilege_fraction=0.5` | Use `floor(rollout_n * fraction)` privileged |
| Resume training | Load cache from disk; continue from where left off |
| Cached solution is very long | May be truncated by tokenizer; `max_model_len` should be large enough |
| Pre-tokenized prompts (wrong rollout backend) | Defensive check in `_inject_privilege_info()` raises RuntimeError if `input_ids` already exists |

---

## 8. Testing Plan

### 8.1 Unit Tests

1. **Cache operations:**
   - Test `_update_best_solutions_cache()` correctly updates only when score improves
   - Test `_update_best_solutions_cache()` applies penalty: privileged with acc=0.8 should NOT displace unprivileged with acc=0.78 when penalty=0.05
   - Test `_load_best_solutions_cache()` and `_save_best_solutions_cache()` round-trip
   - Test `was_privileged` field is correctly stored and loaded

2. **Privilege penalty:**
   - Test `_get_cache_penalty()` returns correct values for each schedule (none, linear, cosine, late_ramp)
   - Test `late_ramp` schedule: penalty should be p_min for t < 0.66, then increase
   - Test `_apply_privilege_penalty()` correctly discounts privileged scores

3. **Privilege assignment:**
   - Test `_should_give_privilege()` respects threshold
   - Test `_inject_privilege_info()` produces correct interleaving pattern
   - Test `_verify_question_match()` catches mismatches
   - Test defensive check raises RuntimeError if `input_ids` already exists

3. **Prompt modification:**
   - Test `_add_privilege_to_prompt()` produces valid prompts
   - Test tokenization doesn't exceed `max_model_len`

4. **Tokenization timing verification:**
   - Verify that `raw_prompt` modifications propagate to actual generated tokens
   - Test with a mock that confirms `apply_chat_template` receives the modified prompt

### 8.2 Integration Tests

1. Run for 10 steps with `privilege_fraction=0.5`, verify:
   - Cache grows over time
   - Privilege fraction increases as cache populates
   - Metrics are logged correctly

2. Resume from checkpoint, verify:
   - Cache is loaded
   - Privilege info consistent with cache state

3. Run full training with `late_ramp` penalty schedule, verify:
   - `best_solutions_cache/privileged_fraction` starts high (most entries from privileged) and declines
   - Late in training, unprivileged solutions can displace privileged ones with same accuracy
   - `self_teacher/privilege_penalty` metric shows expected late_ramp shape

---

## 9. Open Questions / Potential Improvements

1. **Should privileged samples have lower learning weight?** *(ADDRESSED)*
   - **Solution implemented:** Privilege penalty mechanism (Section 3.6-3.7). Privileged samples are discounted in both cache updates and GRPO advantage computation. The penalty increases over training via `late_ramp` schedule, creating increasing pressure to succeed without hints.
   - **Validation metric:** `best_solutions_cache/privileged_fraction` tracks what fraction of cache entries came from privileged rollouts. This should decline over training:
     ```
     cache_privileged_fraction
     1.0 |████████
         |        ████
         |             ████
     0.0 |                  ████▓▓▓░░░
         └────────────────────────────── steps
     ```
     If this curve never declines, the model is not successfully internalizing unprivileged competence — the curriculum isn't transferring. This is an early warning signal to check hyperparameters.

2. **Should cache have a max size or TTL?**
   - Currently: unlimited. Could expire old entries or cap size.

3. **Should privilege_fraction adapt over time?** *(partially addressed)*
   - Implemented as linear decay. Could also try cosine or step schedules if linear proves insufficient.

4. **Should we track "lift" from privilege?**
   - Compare: (score_with_privilege - score_without) for same question over time.

5. **Alternative interleaving patterns:**
   - Current: deterministic `[P, U, P, U, ...]`
   - Alternative: random assignment of privilege within group

6. **Separate GRPO advantage computation per subgroup** *(future work)*
   - **Problem:** When privileged and unprivileged samples compete in the same GRPO group, the group mean is inflated by privileged samples that score higher due to the hint. Unprivileged samples receive unfairly negative advantages — penalized for not having a hint rather than for poor reasoning.
   - **Proposed fix:** Compute GRPO advantages separately within the privileged subgroup and within the unprivileged subgroup for each question:
     ```
     Question Q (8 rollouts, fraction=0.5):
       Privileged group:   [P0, P1, P2, P3] → normalize advantages among these 4 only
       Unprivileged group: [U0, U1, U2, U3] → normalize advantages among these 4 only
     ```
   - **Benefit:** Each subgroup has a clean, uncontaminated learning signal. The model learns "which privileged attempt was best" and "which unprivileged attempt was best" independently.
   - **Trade-off:** Smaller effective group sizes reduce the variance reduction that GRPO relies on. With rollout_n=8 and fraction=0.5, each subgroup has only 4 samples — may require larger rollout_n (e.g., 16) to compensate.
   - **Note:** Linear decay of privilege_fraction mitigates the mixed-group bias over time; separate advantage computation would eliminate it entirely at the cost of implementation complexity.

7. **Per-question dynamic privilege (instead of global decay)** *(potential improvement)*
   - **Problem:** The current plan decays `privilege_fraction` globally and uniformly. This wastes rollout budget: questions the model already solves unprivileged don't need hints, while hard questions may need privilege longer.
   - **Proposed approach:** Track per-question solve rates and adjust privilege locally:
     | Question State | Signal | Privilege Action |
     |----------------|--------|------------------|
     | **Stuck** | `unprivileged_solve_rate ≈ 0%`, `privileged_solve_rate > threshold` | Keep/increase privilege |
     | **Learning** | `unprivileged_solve_rate > 50%` | Reduce privilege |
     | **Mastered** | `unprivileged_solve_rate ≈ privileged_solve_rate` | Remove privilege (fraction=0) |
   - **Benefit:** Localizes curriculum where actually needed. Mastered questions get 100% unprivileged rollouts; stuck questions keep scaffolding.
   - **Implementation sketch:**
     ```python
     # Per-question modifier on top of global decay
     base_fraction = self._get_current_privilege_fraction()  # global decay as fallback
     
     for question_idx in questions:
         stats = self.question_stats.get(question_idx, {})
         unpriv_rate = stats.get("unprivileged_solve_rate", 0.0)
         priv_rate = stats.get("privileged_solve_rate", 0.0)
         
         if unpriv_rate > 0.5:  # "learning"
             question_fraction = base_fraction * 0.5
         elif unpriv_rate < 0.1 and priv_rate > 0.3:  # "stuck"
             question_fraction = min(1.0, base_fraction * 1.5)
         elif abs(unpriv_rate - priv_rate) < 0.1:  # "mastered"
             question_fraction = 0.0
         else:
             question_fraction = base_fraction
     ```
   - **Concerns:**
     - **Statistical noise:** With only 8 rollouts/question/step, solve rates are noisy. Would need exponential moving average across 3-5 appearances of each question.
     - **Cold start:** Early steps lack data; global decay provides a reasonable fallback.
     - **Complexity:** Requires persistent per-question tracking: `privileged_successes`, `unprivileged_successes`, `privileged_attempts`, `unprivileged_attempts`.
   - **Recommendation:** Start with global decay (simpler), add per-question modulation as a follow-up if needed.

---

## 10. Implementation Checklist

- [ ] Create `verl/trainer/config/self_teacher_grpo.yaml`
- [ ] Modify `_update_best_solutions_cache()` to use `acc` instead of `score`; add `description_hash`, `description_preview`, and `was_privileged`
- [ ] Modify `_update_best_solutions_cache()` to apply privilege penalty when comparing accuracies
- [ ] Add `_get_current_privilege_fraction()` method (linear decay)
- [ ] Add `_get_cache_penalty()` method (late_ramp/linear/cosine schedule)
- [ ] Add `_apply_privilege_penalty()` method for GRPO advantage computation
- [ ] Add `_should_give_privilege()` method
- [ ] Add `_verify_question_match()` method
- [ ] Add `_add_privilege_to_prompt()` method
- [ ] Add `_inject_privilege_info()` method (uses decayed fraction)
- [ ] Add `_compute_self_teacher_metrics()` method (with `cache_privileged_fraction`)
- [ ] Modify `fit()` to call privilege injection after `repeat()` but before `generate_sequences()`
- [ ] Modify `fit()` to apply privilege penalty before GRPO advantage computation (if `apply_to_grpo=true`)
- [ ] Modify `fit()` to compute and log self-teacher metrics after reward computation
- [ ] Create `experiments/rich_feedback/run_self_teacher_grpo.sh`
- [ ] Test with dry run
- [ ] Run small-scale experiment to validate

**Note:** The following methods already exist and need NO changes:
- `_load_best_solutions_cache()`
- `_save_best_solutions_cache()`
- `_get_best_solutions_cache_path()`

---

## Appendix A: File Changes Summary

| File | Changes |
|------|---------|
| `verl/trainer/config/self_teacher_grpo.yaml` | **NEW** - config file |
| `verl/trainer/ppo/ray_trainer.py` | Modify 1 existing method (`_update_best_solutions_cache`), add 8 new methods, modify `fit()` |
| `experiments/rich_feedback/run_self_teacher_grpo.sh` | **NEW** - run script |

**Estimated lines of code:** ~280-350 new/modified lines in `ray_trainer.py`

**Existing code that can be reused as-is:**
- `_load_best_solutions_cache()` - already handles loading from JSON
- `_save_best_solutions_cache()` - already handles saving to JSON
- `_get_best_solutions_cache_path()` - returns `{default_local_dir}/best_solutions_cache.json`
- `best_solutions_cache` dict initialized in `__init__`
