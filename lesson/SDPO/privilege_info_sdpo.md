# SDPO Privileged Information: Feedback & Solution Parameters

## Overview

In SDPO, the teacher model receives "privileged information" that the student doesn't have access to at inference time. This information is used to construct a reprompted input for the teacher.

## Two Sources of Privileged Information

### 1. Successful Solution (from sibling rollouts)
- **Source:** Another rollout with the same `uid` that achieved `score >= success_reward_threshold`
- **How it works:** Multiple rollouts are generated for the same question (controlled by `rollout.n`). If any rollout succeeds, its response can be used as a demonstration for failed siblings.

### 2. Environment Feedback (from reward function)
- **Source:** The sample's own reward computation (e.g., test case errors, compiler output)
- **How it works:** The reward function returns a `feedback` field containing diagnostic information about why the response failed.

## Solution Selection (When Multiple Siblings Succeed)

When multiple rollouts for the same question succeed, **the first one found is selected** (effectively random).

From [ray_trainer.py](verl/trainer/ppo/ray_trainer.py) (`_get_solution`):
```python
solution_idx = solution_idxs[0]  # taking the first successful demonstration effectively selects a random one
```

### Selection Behavior

| # Successful Siblings | Selection Method |
|-----------------------|------------------|
| 0 | No solution available |
| 1 | That solution is used |
| 2+ | **First in batch order** (arbitrary/random) |

### What It Does NOT Do
- ❌ Does not select by highest score
- ❌ Does not select shortest solution
- ❌ Does not select by any quality metric

### Potential Improvements
To implement "best" selection, one could modify `_get_solution()` to:
- Sort `solution_idxs` by score before taking `[0]`
- Select shortest solution (fewer tokens)
- Randomly sample (explicit randomness vs. implicit)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `include_environment_feedback` | `False` | Enable/disable using environment feedback |
| `environment_feedback_only_without_solution` | `False` | If `True`, only use feedback when no solution is available |
| `dont_reprompt_on_self_success` | `False` | If `True`, exclude self from solution pool (only use sibling solutions) |
| `success_reward_threshold` | `1.0` | Minimum score to be considered a "successful" solution |

## Scenario Table

With `include_environment_feedback=True` and `environment_feedback_only_without_solution=False`:

| Scenario | Has Solution | Has Feedback | Teacher Input |
|----------|--------------|--------------|---------------|
| Sibling succeeded, self failed with errors | ✓ | ✓ | Prompt + Solution + Feedback |
| Sibling succeeded, self failed (no feedback) | ✓ | ✗ | Prompt + Solution |
| No sibling succeeded, self has feedback | ✗ | ✓ | Prompt + Feedback |
| No sibling succeeded, no feedback | ✗ | ✗ | Original prompt (no distillation applied) |

## Template Format

The reprompted input is constructed using these templates:

```
{prompt}{solution}{feedback}

Correctly solve the original question.
```

Where:
- `solution` = `"\nCorrect solution:\n\n{successful_previous_attempt}\n\n"` (if available)
- `feedback` = `"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"` (if available)

## Code Locations

- Parameter definitions: [verl/workers/config/actor.py](verl/workers/config/actor.py) (`SelfDistillationConfig`)
- Feedback collection: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) (`_collect_feedback`)
- Solution collection: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) (`_collect_solutions_by_uid`)
- Teacher input construction: [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) (`_maybe_build_self_distillation_batch`)
- Feedback generation (code tasks): [verl/utils/reward_score/feedback/code.py](verl/utils/reward_score/feedback/code.py) (`format_test_feedback`)

## Current Settings in `run_sdpo.sh`

**Script:** [experiments/rich_feedback/run_sdpo.sh](experiments/rich_feedback/run_sdpo.sh)

### Privileged Information Parameters

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `include_environment_feedback` | `True` | ✅ Environment feedback **enabled** |
| `dont_reprompt_on_self_success` | `True` | ✅ Only use **sibling** solutions (excludes self-success) |
| `environment_feedback_only_without_solution` | `False` (default) | Use feedback even when solution exists |
| `success_reward_threshold` | `1.0` (default) | Only perfect scores count as "success" |

### What This Means

With these settings:
- **Feedback:** Used for all failed samples that have diagnostic output from the reward function
- **Solutions:** Come only from *other* rollouts with the same question (not self), ensuring the model learns from diverse successful approaches

### Effective Behavior

| Sample's Status | Sibling Succeeded? | Has Feedback? | Teacher Sees |
|-----------------|-------------------|---------------|--------------|
| Failed | Yes | Yes | Solution + Feedback |
| Failed | Yes | No | Solution only |
| Failed | No | Yes | Feedback only |
| Failed | No | No | Original prompt (no distillation) |
| Succeeded | N/A | N/A | Original prompt (no distillation for successful samples) |

### Other Relevant Settings

```bash
rollout.n=8                    # 8 rollouts per question (more chances for solutions)
alpha=1.0                      # Reverse KL divergence
teacher_update_rate=0.01       # Slow EMA update of teacher
distillation_topk=20           # Top-20 logit distillation
```
