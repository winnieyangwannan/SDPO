# Complete SDPO Execution Flow

This document traces the complete execution flow of Self-Distilled Policy Optimization (SDPO) on top of verl.

---

## Overview

SDPO treats the current model conditioned on feedback as a "self-teacher" and distills its feedback-informed next-token predictions back into the student policy. The key insight is that the model can retrospectively identify its own mistakes in-context.

---

## Step 1: Configuration Loading

**`run_sdpo.sh` → `verl_training.sh` → `main_ppo.py --config-name sdpo`**

The `verl/trainer/config/sdpo.yaml` config sets:
```yaml
policy_loss:
  loss_mode: sdpo   # ← Enables SDPO
self_distillation:
  max_reprompt_len: 10240
  is_clip: 2.0      # Importance sampling clipping
rollout:
  calculate_log_probs: True  # Required for rollout correction
```

**Scripts using `sdpo.yaml`:**
- `experiments/rich_feedback/run_sdpo.sh` (CONFIG_NAME="sdpo")
- `experiments/generalization/run_sdpo_all.sh`
- `run_local_sdpo.sh`
- `run_local_test.sh`

---

## Step 2: Worker Initialization (Teacher Model Setup)

**Location:** `verl/workers/fsdp_workers.py` (lines 895-907)

```python
if self_distillation_cfg is not None and loss_mode == "sdpo":
    if teacher_regularization == "trust-region":
        # Logits = lerp(ref_logits, student_logits, mix_coef)
        self.actor.teacher_module = TrustRegionTeacher(
            ref_module=self.ref_module_fsdp,
            student_module=self.actor_module_fsdp,
            mix_coef=self_distillation_cfg.get("teacher_update_rate", 0.0),
        )
    else:  # "ema" (default)
        # Teacher = reference model, updated via EMA
        self.actor.teacher_module = self.ref_module_fsdp
```

**Two teacher regularization modes:**

| Mode | Description |
|------|-------------|
| **EMA** (default) | Teacher = reference model, updated as EMA of student |
| **Trust-region** | Teacher logits = linear interpolation of ref + student logits |

The `TrustRegionTeacher` class in `verl/workers/actor/dp_actor.py` (lines 51-62):
```python
class TrustRegionTeacher(nn.Module):
    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)
```

---

## Step 3: Training Loop - UID Assignment

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 1626-1628)

```python
# Each question gets a unique ID (used to find successful solutions from same question)
batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
)
```

Then the batch is **repeated** for multiple rollouts (e.g., n=8) with `interleave=True` — samples with the same `uid` share the same question.

**Note:** UID itself is not SDPO-specific (used by GRPO too), but SDPO uniquely uses it to find cross-rollout demonstrations via `_collect_solutions_by_uid()`.

---

## Step 4: Reward Computation & Feedback Collection

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 1786-1790)

```python
self_distillation_data = self._maybe_build_self_distillation_batch(
    batch, reward_tensor, reward_extra_infos_dict
)
```

This calls two helper functions:

### 4a. Collect successful solutions per UID

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 636-643)

```python
def _collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold):
    # Groups successful responses by their uid
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:  # default: 0.5
            success_by_uid[uid].append(idx)
```

### 4b. Collect environment feedback

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 611-635)

```python
def _collect_feedback(include_environment_feedback, reward_extra_infos_dict, batch_size):
    # Extracts "feedback" field from reward_extra_infos_dict (e.g., test errors)
    feedback_list[i] = reward_extra_infos_dict["feedback"][i]
```

---

## Step 5: Reprompted Teacher Input Construction

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 708-745)

```python
def _build_teacher_message(i):
    solution_section = ""
    if has_solution:
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = re.sub(r'<think>.*?</think>\s*', '', solution_str)  # Remove thinking traces
        solution_section = solution_template.format(successful_previous_attempt=solution_str)
    
    feedback_section = ""
    if use_feedback:
        feedback_section = feedback_template.format(feedback_raw=feedback_list[i])
    
    reprompt_text = reprompt_template.format(prompt, solution, feedback)
    return system_messages + [{"role": "user", "content": reprompt_text}]
```

**Output tensors:**
- `teacher_input_ids`: Original prompt + solution/feedback context + original response
- `self_distillation_mask`: `True` if sample has a solution OR feedback available

---

## Step 6: Rollout Correction (IS Weights)

**Location:** `verl/trainer/ppo/ray_trainer.py` (lines 1813-1818)

If `calculate_log_probs=True` and `rollout_is: token`:

```python
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
```

This computes importance sampling weights to correct for off-policy shifts.

---

## Step 7: Actor Update - KL Computation

**Location:** `verl/workers/actor/dp_actor.py` (lines 810-850)

```python
# 7a. Forward pass on STUDENT
outputs = self._forward_micro_batch(model_inputs, distill_topk=distill_topk)
student_log_prob = outputs["log_probs"]
student_topk_logps = outputs["topk_logps"]

# 7b. Forward pass on TEACHER (with reprompted inputs!)
teacher_inputs = {
    "input_ids": model_inputs["teacher_input_ids"],  # ← Has solution/feedback context
    "attention_mask": model_inputs["teacher_attention_mask"],
    ...
}
with torch.no_grad():
    teacher_outputs = self._forward_micro_batch(
        teacher_inputs, 
        topk_indices=student_topk_indices, 
        module=teacher_model
    )
teacher_log_prob = teacher_outputs["log_probs"]
```

---

## Step 8: Self-Distillation Loss Calculation

**Location:** `verl/trainer/ppo/core_algos.py` (lines 1085-1180)

```python
def compute_self_distillation_loss(...):
    # Apply self_distillation_mask (only samples with solution/feedback)
    loss_mask = response_mask * self_distillation_mask.unsqueeze(1)
    
    # Top-k distillation with tail bucket
    if distillation_topk and distillation_add_tail:
        student_distill_log_probs = add_tail(student_topk_log_probs)
        teacher_distill_log_probs = add_tail(teacher_topk_log_probs)
    
    # KL computation based on alpha
    if alpha == 0.0:  # Forward KL
        kl_loss = F.kl_div(student_probs, teacher_probs, log_target=True)
    elif alpha == 1.0:  # Reverse KL  
        kl_loss = F.kl_div(teacher_probs, student_probs, log_target=True)
    else:  # Jensen-Shannon Divergence
        mixture = logsumexp([student + log(1-α), teacher + log(α)])
        kl_loss = lerp(KL(mixture||student), KL(mixture||teacher), alpha)
    
    # Importance sampling clipping (is_clip)
    if is_clip is not None:
        ratio = exp(student_log_probs - old_log_probs).clamp(max=is_clip)
        per_token_loss *= ratio
    
    # Apply rollout correction weights
    if rollout_is_weights is not None:
        per_token_loss *= rollout_is_weights
```

---

## Step 9: EMA Teacher Update

**Location:** `verl/workers/actor/dp_actor.py` (lines 132-156, called at line 920-921)

After each successful gradient step:

```python
def _update_teacher(self):
    if teacher_regularization == "ema" and update_rate > 0:
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            # teacher = (1 - rate) * teacher + rate * student
            teacher_param.data.mul_(1 - update_rate).add_(student_data, alpha=update_rate)
```

---

## SDPO Component Summary Table

| Component | Location | Purpose |
|-----------|----------|---------|
| `loss_mode: sdpo` | Config | Enables SDPO |
| `self_distillation` config | `verl/trainer/config/actor/actor.yaml` (lines 84-144) | All distillation params |
| Teacher initialization | `verl/workers/fsdp_workers.py` (lines 895-907) | EMA or Trust-region teacher |
| UID tracking | `verl/trainer/ppo/ray_trainer.py` (lines 1626-1628) | Groups responses by question |
| Solution collection | `verl/trainer/ppo/ray_trainer.py` (lines 636-643) | Finds successful demos |
| Feedback collection | `verl/trainer/ppo/ray_trainer.py` (lines 611-635) | Extracts env feedback |
| Reprompt construction | `verl/trainer/ppo/ray_trainer.py` (lines 708-745) | Builds teacher input |
| Thinking removal | `verl/trainer/ppo/ray_trainer.py` (lines 647-649) | Cleans `<think>` tags |
| Teacher forward pass | `verl/workers/actor/dp_actor.py` (lines 820-830) | Gets teacher logits |
| KL loss | `verl/trainer/ppo/core_algos.py` (lines 1085-1180) | Distillation objective |
| Top-k + tail bucket | `verl/trainer/ppo/core_algos.py` (lines 1110-1135) | Efficient distillation |
| IS clipping | `verl/trainer/ppo/core_algos.py` (lines 1168-1175) | Variance reduction |
| Rollout correction | `verl/trainer/ppo/rollout_corr_helper.py` | Off-policy correction |
| EMA update | `verl/workers/actor/dp_actor.py` (lines 132-156) | Teacher adaptation |

---

## Visual Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Iteration                       │
├─────────────────────────────────────────────────────────────┤
│  1. Rollout: Generate responses from student                │
│  2. Reward: Identify successful responses (reward >= thresh)│
│  3. Reprompt: Build teacher inputs with solution/feedback   │
│                                                             │
│  ┌─────────────┐        ┌─────────────┐                     │
│  │   Student   │   KL   │   Teacher   │                     │
│  │  (actor)    │ ◄───── │ (ref + ctx) │                     │
│  └─────────────┘        └─────────────┘                     │
│        │                      │                             │
│        │  Gradient via KL     │  Contextual guidance        │
│        ▼                      │  (successful demo/feedback) │
│  ┌─────────────┐              │                             │
│  │ Optimizer   │              │                             │
│  │   Step      │              │                             │
│  └─────────────┘              │                             │
│        │                      │                             │
│        └── EMA update ────────┘                             │
│            (teacher params updated toward student)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Configuration Options

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo  # Enable SDPO
    self_distillation:
      full_logit_distillation: true    # Use full logit KL (vs token-level)
      alpha: 0.0                       # 0=forward KL, 1=reverse KL, 0.5=JSD
      success_reward_threshold: 0.5    # Min reward to count as successful
      teacher_regularization: ema      # "ema" or "trust-region"
      teacher_update_rate: 0.05        # EMA coefficient
      distillation_topk: 100           # Top-k logits for efficiency
      distillation_add_tail: true      # Add tail bucket for top-k
      max_reprompt_len: 10240          # Max tokens for reprompted prompt
      dont_reprompt_on_self_success: true  # Don't use own success as demo
      remove_thinking_from_demonstration: true  # Strip <think> tags
      include_environment_feedback: true  # Include test errors etc.
      is_clip: 2.0                     # Clip IS ratio
```
