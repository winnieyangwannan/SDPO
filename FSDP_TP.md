# FSDP vs Tensor Parallelism (TP)

## Overview

Both FSDP and TP are techniques to distribute model training across GPUs, but they work differently.

| Aspect | FSDP | Tensor Parallelism |
|--------|------|-------------------|
| **What's sharded** | Full parameters, gradients, optimizer states | Individual tensors (weights) within layers |
| **When parameters gather** | Before forward/backward, then re-shard | Always split—each GPU holds a slice |
| **Communication pattern** | All-gather before compute, reduce-scatter after | All-reduce within each layer's matmul |
| **Data parallelism** | Yes—each GPU processes different data batches | No—all GPUs process the same input |
| **Scaling** | Works across nodes (tolerates higher latency) | Best within a node (requires low latency NVLink) |
| **Code changes** | Minimal—wraps existing model | Requires model architecture changes |

---

## Fundamental Difference: How They Split the Model

Think of a neural network layer as a matrix multiplication: `output = input × Weight`

### FSDP (Fully Sharded Data Parallel) — *Each GPU stores a fraction of every layer*

```
Weight matrix W (full):     GPU0 stores:    GPU1 stores:
┌─────────────────┐         ┌────────┐      ┌────────┐
│ a  b  c  d  e  f│   →     │ a  b  c│      │ d  e  f│
│ g  h  i  j  k  l│         │ g  h  i│      │ j  k  l│
│ m  n  o  p  q  r│         │ m  n  o│      │ p  q  r│
└─────────────────┘         └────────┘      └────────┘
                            (rows 1-3,      (rows 1-3,
                             cols 1-3)       cols 4-6)
```

When computing:
1. **Gather** all shards → reconstruct full W on each GPU
2. Each GPU computes with **different data samples** (data parallel)
3. **Scatter** gradients back to shards

### Tensor Parallelism — *Each GPU computes part of every layer's output*

```
Weight matrix W:            GPU0 gets:      GPU1 gets:
┌─────────────────┐         ┌────────┐      ┌────────┐
│ a  b  c  d  e  f│   →     │ a  b  c│      │ d  e  f│
│ g  h  i  j  k  l│         │ g  h  i│      │ j  k  l│
│ m  n  o  p  q  r│         │ m  n  o│      │ p  q  r│
└─────────────────┘         └────────┘      └────────┘
                            (left half)     (right half)
```

When computing (same input on both GPUs):
1. GPU0: `partial_out_0 = input × W_left`
2. GPU1: `partial_out_1 = input × W_right`
3. **All-reduce** to combine: `output = [partial_out_0 | partial_out_1]`

---

## Key Intuition

| | FSDP | Tensor Parallelism |
|---|---|---|
| **What's different per GPU** | The **data** (different batches) | The **computation** (partial matmul) |
| **Same input on all GPUs?** | No — each sees different samples | Yes — all process same input |
| **When communication happens** | Before/after entire forward pass | Inside every layer |
| **Memory saving** | By not storing full params | By splitting weight matrices |

---

## FSDP Shards Both Weights AND Data

FSDP shards:
1. **Model weights** — each GPU stores only 1/N of the parameters
2. **Data** — each GPU processes different batches (like regular Data Parallel)

FSDP is called "Data Parallel" because each GPU ultimately processes **different data samples**. But unlike vanilla Data Parallel (which stores the full model on each GPU), FSDP also shards the weights.

### What happens in FSDP:

```
STORAGE (persistent):
  GPU0: [shard 0 of weights]  ← only 1/4 of model
  GPU1: [shard 1 of weights]  ← only 1/4 of model
  GPU2: [shard 2 of weights]  ← only 1/4 of model
  GPU3: [shard 3 of weights]  ← only 1/4 of model

COMPUTE (temporary, per layer):
  1. All-gather: reconstruct full weights for this layer
  2. Each GPU runs forward on its OWN data batch
  3. Each GPU runs backward on its OWN data batch
  4. Reduce-scatter: average gradients & re-shard weights
```

### Comparison with DDP:

| Method | Weights stored | Data processed | 
|--------|---------------|----------------|
| **Data Parallel (DDP)** | Full copy on each GPU | Different batches per GPU |
| **FSDP** | Sharded across GPUs | Different batches per GPU |
| **Tensor Parallel** | Sharded across GPUs | Same batch on all GPUs |

---

## Why TP Speeds Up Inference But FSDP Does Not

**TP speeds up a single sample** because it splits the *computation* itself:

```
Single input "Hello" with TP=4:
  GPU0: computes output[:, 0:25%]    ─┐
  GPU1: computes output[:, 25:50%]   ─┼─→ combine → full output
  GPU2: computes output[:, 50:75%]   ─┤     (faster than 1 GPU)
  GPU3: computes output[:, 75:100%]  ─┘
```

**FSDP does NOT speed up a single sample** because it must first reconstruct full weights:

```
Single input "Hello" with FSDP=4:
  Step 1: All-gather weights (communication overhead!)
          GPU0,1,2,3 now each have FULL weights temporarily
  
  Step 2: Only ONE GPU actually processes "Hello"
          (or each GPU processes different data)
  
  Step 3: Re-shard weights back
```

### Why the difference?

| | FSDP | TP |
|---|---|---|
| **Goal** | Save memory during training | Speed up single-sample compute |
| **Weights at compute time** | Full copy (gathered temporarily) | Still split |
| **How it "parallelizes"** | Multiple samples across GPUs | Single sample split across GPUs |
| **Single-sample latency** | Same as 1 GPU (+ gather overhead) | ~1/N of 1 GPU |

---

## Analogies

### FSDP Analogy
4 people each store 1/4 of a cookbook. To cook ONE dish, they first photocopy all pages to reconstruct the full book, then ONE person cooks. No speedup for one dish.

### TP Analogy
4 chefs in a kitchen, each responsible for different parts of every dish (one does protein, one does sauce, etc.). ONE dish gets cooked 4x faster because they work simultaneously.

### Library vs Factory
- **FSDP** = A library where each person stores different pages of every book. To read, you gather all pages, read your own chapter (your data batch), then redistribute pages.
- **TP** = A factory assembly line where each worker does a specific part of building every single car. Workers must coordinate on every car.

---

## When to Use Each

- **FSDP**: Large models that don't fit on one GPU, multi-node training, when you want minimal code changes
- **TP**: Very large models (100B+), single-node with fast interconnect (NVLink), often combined with FSDP (FSDP+TP hybrid)

---

## In VERL/SDPO Codebase

In this codebase, FSDP is the primary training backend (`verl/workers/fsdp_workers.py`), while the Megatron backend supports TP for very large MoE models like DeepSeek-V3 and Qwen3-235B.

Example from `run_baseline_grpo_27b.sh`:
- **Actor (training)**: Uses FSDP — shards across all 16 GPUs
- **Rollout (inference)**: Uses TP=4 — each model replica spans 4 GPUs → 4 replicas total

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

This is a common hybrid pattern: FSDP handles training memory efficiently across nodes, while TP accelerates inference by splitting weight matrices within fast NVLink-connected GPUs.
