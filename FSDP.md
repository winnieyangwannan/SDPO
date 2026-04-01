# FSDP (Fully Sharded Data Parallel) Guide

## What is FSDP?

FSDP (Fully Sharded Data Parallel) is a PyTorch distributed training technique that shards model parameters, gradients, and optimizer states across multiple GPUs. Unlike standard Data Parallel (DP) which replicates the entire model on each GPU, FSDP:
- Reduces memory usage per GPU by distributing model weights
- Only gathers full parameters when needed for forward/backward passes
- Enables training of larger models that wouldn't fit on a single GPU

---

## Why FSDP Needs Model-Specific Code

**The core principle:** FSDP itself is model-agnostic, but practical issues arise from:

### 1. Wrapping Policy — Which modules to shard?

FSDP doesn't shard individual parameters—it wraps entire `nn.Module` submodules. You must tell it:
- Which layer types to wrap (e.g., `Qwen3DecoderLayer`)
- Wrapping too granularly = excessive communication overhead
- Wrapping too coarsely = memory not distributed well

### 2. Device Placement — Where do tensors live during CPU offload?

When `cpu_offload=True`, FSDP moves weights to CPU and back. But some tensors (like positional embeddings, vision encoders) may be created/accessed outside the normal forward flow, causing device mismatches.

### 3. Non-Standard Architectures — Special handling needed for:
- **Vision-Language Models**: Visual encoder may have different sharding needs
- **MoE models**: Expert routing creates dynamic compute graphs
- **Custom embeddings**: Positional encodings that interpolate or are computed differently

### 4. Parameter Tying — Shared weights across modules

If `lm_head` shares weights with `embed_tokens`, FSDP must know to avoid double-sharding.

---

## What Parts of Model Architecture Affect FSDP?

| Architecture Feature | FSDP Impact |
|---------------------|-------------|
| **Transformer layers** | Primary wrap target (`_no_split_modules`) |
| **Vision encoder** | May need separate wrap policy |
| **Positional embeddings** | Device placement during CPU offload |
| **Tied weights** | Must be handled to avoid corruption |
| **Dynamic shapes** | Can break FSDP's static sharding assumptions |
| **Custom forward methods** | May access tensors before FSDP gathers them |

---

## Qwen3.5 + FSDP Code Walkthrough

### File 1: Wrap Policy (`verl/utils/fsdp_utils.py`)

This determines which modules FSDP wraps:

```python
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    # HuggingFace models define _no_split_modules
    # For Qwen3.5, this is typically ["Qwen3_5DecoderLayer"]
    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    
    # Use transformer_auto_wrap_policy to wrap each decoder layer
    transformer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,  # e.g., {Qwen3_5DecoderLayer}
    )
```

**Key concept:** `_no_split_modules` is defined in HuggingFace model classes. For Qwen3.5, it's in the transformers library:

```python
class Qwen3_5ForConditionalGeneration(PreTrainedModel):
    _no_split_modules = ["Qwen3_5DecoderLayer"]  # FSDP wraps at this granularity
```

Each `Qwen3_5DecoderLayer` becomes an FSDP unit—its params are sharded across GPUs and gathered when needed.

---

### File 2: FSDP2 CPU Offload Fix (`verl/models/transformers/qwen3_5.py`)

This is the **critical FSDP-specific fix** for Qwen3.5:

```python
def fast_pos_embed_interpolate(self, grid_thw):
    # ... 
    # THE BUG: When FSDP2 uses cpu_offload, self.pos_embed.weight lives on CPU
    # But grid_thw (input) is on GPU. Creating tensors without specifying device
    # causes them to go to CPU, then GPU operations fail.
    
    # THE FIX: Explicitly get device from input tensor
    device = grid_thw.device  # <-- This line fixes the bug
    
    # Now create all tensors on the correct device
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
```

**Why this happens:**
1. FSDP2 with `cpu_offload=True` moves model weights to CPU
2. When forward runs, FSDP gathers weights to GPU temporarily
3. But `self.pos_embed` is an `nn.Embedding`—when accessed, its weight may still be on CPU during the gather
4. Creating tensors without explicit `device=` defaults to CPU
5. Mix of CPU/GPU tensors causes errors

---

### File 3: Monkey Patching (`verl/models/transformers/monkey_patch.py`)

This applies the fix at runtime:

```python
elif model.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
    from verl.models.transformers.qwen3_5 import (
        fast_pos_embed_interpolate,
        forward_with_normal_backend,
        qwen3_5_base_forward
    )

    # Patch the forward methods
    Qwen3_5Model.forward = qwen3_5_base_forward
    Qwen3_5ForConditionalGeneration.forward = forward_with_normal_backend
    
    # FSDP2 CPU offload fix: patch the vision model's pos_embed interpolation
    Qwen3_5VisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    Qwen3_5MoeVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
```

---

## Guide: Adding FSDP Support to a New Model

### Checklist

```
□ 1. WRAP POLICY
   - Check if HuggingFace model has _no_split_modules defined
   - If not, add it: _no_split_modules = ["YourDecoderLayer"]
   - Or configure wrap_policy.transformer_layer_cls_to_wrap in config

□ 2. DEVICE PLACEMENT (Critical for CPU offload)
   - Search for torch.tensor(...) or torch.zeros(...) without device=
   - Search for .to() calls that might fail with offloaded params
   - Fix: Always derive device from input tensors, not model params
   
□ 3. VISION ENCODERS (for VLMs)
   - Vision models often have special positional embeddings
   - These are accessed before main transformer layers
   - May need separate device handling

□ 4. TIED WEIGHTS
   - If lm_head shares weights with embed_tokens, ensure FSDP handles it
   - Check model config: tie_word_embeddings

□ 5. CUSTOM FORWARD METHODS
   - If you override forward(), ensure it works with FSDP gathered params
   - Avoid caching tensors between forward calls
```

---

### Example Template for New Model

```python
# your_model_fsdp.py

import torch
from transformers.models.your_model.modeling_your_model import YourModelForCausalLM

def fixed_pos_embed_forward(self, grid_info):
    """Fix device placement for FSDP2 CPU offload"""
    # ALWAYS get device from input, not from self.weight
    device = grid_info.device
    
    # Create tensors with explicit device
    indices = torch.arange(..., device=device)
    result = self.embedding(indices).to(device)
    return result


def apply_fsdp_patches(model):
    """Apply FSDP compatibility patches to model"""
    if model.config.model_type == "your_model":
        # Patch any methods that create tensors without device
        model.encoder.pos_embed_layer.forward = fixed_pos_embed_forward
        
    return model


# In your training code:
model = AutoModelForCausalLM.from_pretrained("your-model")
model = apply_fsdp_patches(model)

# Then wrap with FSDP
fsdp_model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(...),
    cpu_offload=CPUOffload(offload_params=True),  # This triggers the issues
    ...
)
```

---

## Qwen3.5 FSDP Code Reference (with File Paths & Line Numbers)

### Part 1: Wrap Policy

**File:** `verl/utils/fsdp_utils.py` (Lines 76-143)

```python
# Line 98: Gets _no_split_modules from HuggingFace model
default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)

# Lines 99-101: Uses configured or default wrap policy  
fsdp_transformer_layer_cls_to_wrap = _get_attr(
    "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
)

# Lines 133-136: Creates transformer wrap policy
transformer_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=transformer_cls_to_wrap,  # e.g., {Qwen3_5DecoderLayer}
)
```

**File:** `verl/utils/fsdp_utils.py` (Lines 534-560) - FSDP2

```python
def apply_fsdp2(model, fsdp_kwargs, config):
    # Line 537: Get wrap targets from model
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    
    # Line 549: Select modules to wrap
    modules = _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap)
    
    # Lines 554-556: Wrap each transformer layer
    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    
    # Line 560: Wrap root model
    fully_shard(model, **fsdp_kwargs)
```

---

### Part 2: Device Placement (CPU Offload Fix)

**File:** `verl/models/transformers/qwen3_5.py` (Lines 17-79)

```python
def fast_pos_embed_interpolate(self, grid_thw):
    # ...
    # Line 23: THE CRITICAL FIX - Get device from input tensor
    device = grid_thw.device
    
    # Line 61: Create index tensor on correct device
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    
    # Line 62: Create weight tensor on correct device  
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    
    # Line 63: Explicitly move pos_embed output to device
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
```

---

### Part 3: Vision Encoders

**File:** `verl/models/transformers/monkey_patch.py` (Lines 487-516)

```python
elif model.config.model_type in ["qwen3_5", "qwen3_5_moe"]:
    # Lines 489-494: Import vision models
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
        Qwen3_5TextModel,
    )
    
    # Lines 501-505: Import FSDP-compatible functions
    from verl.models.transformers.qwen3_5 import (
        fast_pos_embed_interpolate,
        forward_with_normal_backend,
        qwen3_5_base_forward
    )
    
    # Lines 515-516: Patch vision model for FSDP2 CPU offload
    Qwen3_5VisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    Qwen3_5MoeVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
```

**File:** `verl/models/transformers/qwen3_5.py` (Lines 82-136)

```python
def _get_input_embeds(model, input_ids, pixel_values=None, pixel_values_videos=None, ...):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    
    # Line 92: Process image inputs through vision encoder
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        # ... merge image embeddings with text embeddings
```

---

### Part 4: Tied Weights

**File:** `verl/utils/fsdp_utils.py` (Lines 510-530)

```python
def _select_fsdp2_wrap_targets(model, fsdp_transformer_layer_cls_to_wrap):
    """When tie_word_embeddings is True, embed_tokens and lm_head share weights 
    and must not be wrapped separately."""
    
    # Line 519: Check if weights are tied
    _tie = getattr(model.config, "tie_word_embeddings", False)
    
    # Line 520: If tied, don't wrap embed_tokens/lm_head separately
    _wrap_by_name = set() if _tie else {"embed_tokens", "lm_head"}
    
    for name, module in model.named_modules():
        if (isinstance(module, nn.Embedding) and not _tie):  # Don't wrap tied embeddings
            modules.append(module)
```

**File:** `verl/workers/fsdp_workers.py` (Line 439)

```python
# Meta tensor init disabled for tied weights (causes issues)
init_context = get_init_weight_context_manager(
    use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
)
```

---

### Part 5: Custom Forward Methods

**File:** `verl/models/transformers/qwen3_5.py` (Lines 138-163)

```python
def qwen3_5_base_forward(self, input_ids, pixel_values=None, ...):
    """Base model forward that handles vision inputs"""
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, 
        image_grid_thw, video_grid_thw
    )
    kwargs.update(input_kwargs)
    return self.language_model(input_ids=None, **kwargs)
```

**File:** `verl/models/transformers/qwen3_5.py` (Lines 192-222)

```python
def forward_with_torch_backend(self, input_ids=None, temperature=1.0, **kwargs):
    """Forward with PyTorch fused kernels for PPO"""
    from verl.utils.experimental.torch_functional import FusedLinearForPPO
    
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Fused log_probs/entropy calculation
    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states, self.lm_head.weight, rolled_labels, temperature
    )
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, ...)
```

**File:** `verl/models/transformers/qwen3_5.py` (Lines 225-251)

```python
def forward_with_triton_backend(self, input_ids=None, temperature=1.0, **kwargs):
    """Forward with Triton fused kernels for PPO"""
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
    
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    
    # Triton fused log_probs/entropy
    log_probs, entropy = linear_cross_entropy(
        hidden_states, self.lm_head.weight, rolled_labels, temperature, "none"
    )
    return Qwen3_5CausalLMOutputForPPO(log_probs=log_probs, entropy=entropy, ...)
```

**File:** `verl/models/transformers/monkey_patch.py` (Lines 507-512) - Applying Patches

```python
# Patch forward methods for Qwen3.5
Qwen3_5Model.forward = qwen3_5_base_forward
Qwen3_5MoeModel.forward = qwen3_5_base_forward
Qwen3_5ForConditionalGeneration.forward = forward_with_normal_backend
Qwen3_5MoeForConditionalGeneration.forward = forward_with_normal_backend
```

---

### Summary: Qwen3.5 FSDP Code Locations

| Part | File | Lines | Key Code |
|------|------|-------|----------|
| **1. Wrap Policy** | `verl/utils/fsdp_utils.py` | 76-143, 534-560 | `_no_split_modules` → `transformer_auto_wrap_policy` |
| **2. Device Placement** | `verl/models/transformers/qwen3_5.py` | 17-79 | `device = grid_thw.device` (Line 23) |
| **3. Vision Encoders** | `verl/models/transformers/monkey_patch.py` | 487-516 | Patches `Qwen3_5VisionModel.fast_pos_embed_interpolate` |
| **4. Tied Weights** | `verl/utils/fsdp_utils.py` | 510-530 | Checks `tie_word_embeddings` |
| **5. Custom Forwards** | `verl/models/transformers/qwen3_5.py` | 138-251 | `qwen3_5_base_forward`, `forward_with_torch_backend`, `forward_with_triton_backend` |

---

## Common FSDP Bugs and Fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| Device mismatch | `RuntimeError: expected device cuda:0 but got cpu` | Derive device from input, not model params |
| Hang during forward | Training freezes | Check wrap policy, ensure all ranks have same modules |
| NaN gradients | Loss becomes NaN | Check mixed precision config, reduce gradient accumulation |
| OOM after FSDP | More memory than expected | Ensure reshard_after_forward=True (ZeRO-3) |
| Checkpoint corruption | Can't load saved model | Use FSDP state_dict methods, not vanilla torch.save |

---

## FSDP vs Tensor Parallelism (TP)

See [FSDP_TP.md](FSDP_TP.md) for a detailed comparison of FSDP and Tensor Parallelism.

### Quick Summary

| Aspect | FSDP | Tensor Parallelism |
|--------|------|-------------------|
| **What's sharded** | Full parameters, gradients, optimizer states | Individual tensors (weights) within layers |
| **Data processed** | Different batches per GPU | Same batch on all GPUs |
| **Single-sample speedup** | No (must gather full weights first) | Yes (computation itself is split) |
| **Best for** | Training memory efficiency | Inference speed, very large models |

---

## FSDP Configuration in VERL

In VERL, FSDP is configured through the actor config:

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2  # or "fsdp" for FSDP1
    fsdp_config:
      param_offload: False      # CPU offload for params
      optimizer_offload: False  # CPU offload for optimizer states
      reshard_after_forward: True  # ZeRO-3 style, reshards after forward
      wrap_policy:
        transformer_layer_cls_to_wrap: ["Qwen3_5DecoderLayer"]
```

### Key Files in VERL Codebase

| File | Purpose |
|------|---------|
| `verl/utils/fsdp_utils.py` | Core FSDP utilities, wrap policies, offload functions |
| `verl/workers/fsdp_workers.py` | FSDP worker implementations for training |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | FSDP checkpoint saving/loading |
| `verl/models/transformers/qwen3_5.py` | Qwen3.5-specific FSDP fixes |
| `verl/models/transformers/monkey_patch.py` | Applies model patches at runtime |
