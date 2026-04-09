# Training Metrics Reference

This document explains key training metrics logged during SDPO and GRPO training.

## Core Training Metrics (All Algorithms)

| Metric | Description | What to Watch For |
|--------|-------------|-------------------|
| `critic/score/mean` | Average reward per batch | Should increase over training |
| `actor/pg_loss` | Policy gradient loss - optimization signal strength | Fluctuates based on advantage variance |
| `actor/entropy` | Policy diversity/randomness | Avoid collapse to 0 (mode collapse) |
| `actor/grad_norm` | Gradient magnitude | Spikes indicate training instability |
| `rollout_corr/kl` | KL divergence from reference policy | Stay small (<0.1); large values = policy drift |
| `response_length/mean` | Average output token length | Track verbosity changes |

---

## SDPO-Specific Metrics

### `actor/kl_loss`
**Student-teacher KL divergence** - measures the divergence between:
- **Student**: Policy on the original prompt
- **Teacher**: Policy on the re-prompted input (with solution/feedback)

Lower values mean the student is successfully imitating the teacher's behavior.

---

### `self_distillation/success_sample_fraction`
**% of samples with a successful solution available for demonstration.**

A sample has a solution if another sample in the same prompt group achieved `reward >= success_reward_threshold` (default: **0.5**).

Note: If `dont_reprompt_on_self_success=True` (default), a sample's own success is excluded.

---

### `self_distillation/success_group_fraction`
**% of prompt groups that have at least one successful sample.**

A prompt group is all samples generated from the same prompt (controlled by `n` rollouts per prompt).

---

### `self_distillation/feedback_available_fraction`
**% of samples that have environment feedback available.**

Feedback comes from the environment (e.g., compiler errors, test results, execution traces).

---

### `self_distillation/feedback_used_fraction`
**% of samples where environment feedback is actually used for re-prompting.**

#### How it's computed:
```python
feedback_used = [
    feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
    for i in range(batch_size)
]
feedback_used_fraction = sum(feedback_used) / batch_size
```

#### Why `feedback_used` ≤ `feedback_available`:
When `environment_feedback_only_without_solution=True` (default), feedback is only used when **no solution exists**. This avoids redundancy — if a successful demonstration is available, it's preferred over error feedback.

| Scenario | Solution Available | Feedback Available | Feedback Used? |
|----------|-------------------|-------------------|----------------|
| A | ✓ | ✓ | ❌ (solution preferred) |
| B | ✓ | ❌ | ❌ |
| C | ❌ | ✓ | ✓ |
| D | ❌ | ❌ | ❌ |

---

### `self_distillation/reprompt_sample_fraction`
**% of samples that receive any form of re-prompting (solution OR feedback).**

This is the `self_distillation_mask` — samples with `mask=True` will have distillation loss applied.

```python
reprompt_sample_fraction = mean(solution_available OR feedback_used)
```

---

## When Solution/Feedback is NOT Used

### Solution NOT Available:
1. **No success in group** — No sample achieved `reward >= success_reward_threshold` (default: 0.5)
2. **`dont_reprompt_on_self_success=True`** — Sample's own success is excluded
3. **Only self-success** — Sample is the only successful one in its group

### Feedback NOT Used:
1. **No feedback from environment** — No error messages/test results returned
2. **`include_environment_feedback=False`** — Feedback collection disabled
3. **`environment_feedback_only_without_solution=True` + Has Solution** — Solution takes priority

---

## Validation Metrics

| Metric | Description |
|--------|-------------|
| `val-core/livecodebench/acc/best@{k}/mean` | Best-of-k accuracy on validation set |
| `val-core/livecodebench/acc/worst@{k}/mean` | Worst-of-k accuracy (conservative estimate) |
| `val-core/livecodebench/score/mean@{k}` | Average score across k samples |

---

## Plotting Metrics

Use the plotting script to visualize training:
```bash
python scripts/plot_training_metrics.py \
    /path/to/log1.log /path/to/log2.log \
    --labels "Seed 1" "Seed 2" \
    --output /path/to/output.png
```

The script auto-detects SDPO vs GRPO and selects appropriate metrics.
