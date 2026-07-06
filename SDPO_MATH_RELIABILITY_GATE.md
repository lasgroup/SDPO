# SDPO-Math Reliability Gate

This note documents the SDPO-Math improvement used for the thesis runs: reliability-gated self-distillation. It is the main comparison against Base RL and vanilla SDPO on DAPO-Math.

## Goal

Vanilla SDPO uses feedback-reprompted model outputs as distillation targets. In math, those targets are not equally reliable:

- A correct peer solution is a strong target.
- Safe correctness feedback can be useful, but it is weaker than a verified solution.
- Format feedback is weaker again.
- Truncated outputs are unsafe to imitate.

Reliability-gated SDPO keeps the SDPO mechanism but makes the target selection feedback-aware. The model still learns from RL reward, but the SDPO loss is applied only to reliable self-distillation targets.

## Compared Variants

| Variant | Objective | Feedback used | Reliability weighting | Reliability gate |
|---|---|---:|---:|---:|
| `base_rl` | RL only | no | no | no |
| `sdpo_vanilla` | RL + SDPO | yes | no | no |
| `sdpo_reliability_gate` | RL + gated weighted SDPO | yes | yes | yes |

`sdpo_reliability` is still implemented for analysis, but the thesis runbook focuses on the three variants above to save training time.

## Pipeline

For each training prompt, the rollout worker samples multiple answers:

```text
prompt x
  -> rollout answers y_1, ..., y_K
  -> math verifier / reward function scores each answer
  -> feedback is produced for incorrect or malformed answers
  -> SDPO builds a teacher reprompt from:
       original problem
       successful peer solution, if available
       correctness/format feedback, if used
  -> teacher target is used for self-distillation
```

The math reward expects the final answer in `\boxed{...}`. The validation metric is exact/math-verify score on the held-out DAPO-Math validation subset.

## Objective

Let:

- `x` be the math problem.
- `y` be the student response.
- `r(x, y)` be the math reward.
- `A(x, y)` be the RL advantage.
- `q_phi(. | x, f)` be the SDPO teacher distribution after feedback reprompting.
- `pi_theta(. | x)` be the current student policy.
- `m_i` be the SDPO target mask for sample `i`.
- `w_i` be the reliability weight.
- `g_i` be the binary reliability gate.

The base RL objective is the usual clipped policy-gradient objective:

```text
L_base = L_RL(pi_theta; A)
```

Vanilla SDPO adds self-distillation on every available target:

```text
L_vanilla = L_RL + lambda * L_SDPO
```

where the implemented SDPO loss is full-logit top-k generalized Jensen-Shannon distillation:

```text
L_SDPO =
  mean_{i,t} m_i * D_GJS(
    pi_theta(. | x_i, y_{i,<t}),
    q_phi(. | x_i, f_i, y_{i,<t})
  )
```

Reliability-gated SDPO changes the mask and weight:

```text
L_gate =
  L_RL
  + lambda * mean_{i,t}
      m_i * g_i * w_i *
      D_GJS(
        pi_theta(. | x_i, y_{i,<t}),
        q_phi(. | x_i, f_i, y_{i,<t})
      )
```

The gate is:

```text
g_i = 1[w_i >= tau] and selected_by_batch_budget(i)
```

with:

```text
tau = 0.4
max selected fraction = 0.5
```

The batch budget keeps the highest-reliability eligible rows when too many samples pass the threshold.

## Reliability Weights

The implementation assigns a reliability weight to each SDPO target:

| Case | Weight | Reason |
|---|---:|---|
| Verified successful peer solution | `1.0` | Strongest target; it passed reward verification. |
| Safe correctness feedback | `0.4` | Useful correction signal, but not a verified solution. |
| Format feedback | `0.2` | Helps formatting, but weak for reasoning quality. |
| Truncated output | `0.0` | Unsafe target; should not be imitated. |
| No solution and no feedback | `0.0` | No useful SDPO target. |

With the default gate threshold `tau = 0.4`, the gate keeps verified peer solutions and safe correctness-feedback targets, but skips pure format-feedback and truncated targets.

## Why This Improves SDPO

Vanilla SDPO assumes that every reprompted target is beneficial. That assumption is weak in math because an incorrect answer can still produce a long, fluent, misleading solution. Distilling from that target can reinforce wrong reasoning.

Reliability-gated SDPO improves the algorithm in two ways:

1. **Higher-quality SDPO targets**
   The SDPO loss is concentrated on verified or safer feedback-derived targets. Low-confidence targets are removed from the distillation objective.

2. **Lower SDPO compute**
   Sparse target execution avoids computing the expensive SDPO branch for many low-reliability samples. This is important because SDPO adds teacher inputs, teacher logits, and actor updates on top of RL rollout generation.

The thesis claim should be phrased as:

```text
Reliability-gated SDPO improves math-domain self-distillation by making feedback-derived targets selective and reliability-weighted, reducing noisy imitation while preserving the useful correction signal.
```

## Active Training Configuration

The centralized entry point is:

```bash
bash experiments/math/run_sdpo_math_benchmark.sh
```

The active thesis run uses:

| Field | Value |
|---|---|
| Dataset | `open-r1/DAPO-Math-17k-Processed`, English subset |
| Prepared files | `data/dapo_math_en/train.parquet`, `data/dapo_math_en/val.parquet` |
| Model | `Qwen/Qwen3-8B` |
| Python | `3.12` |
| Attention | SDPA |
| Rollout backend | vLLM |
| Rollout tensor parallel | `2` |
| LoRA | rank `32`, alpha `32` |
| Quantization | `null` |
| Qwen3 thinking mode | disabled |
| Validation sampling | greedy, `n=1`, `temperature=0.01` |
| Train seed | `42` |
| Variants | `base_rl sdpo_vanilla sdpo_reliability_gate` |

### H200 Thesis Profile

Used when:

```bash
export PHASE=thesis
export HARDWARE_PROFILE=h200
```

| Setting | Value |
|---|---:|
| Train steps | `15` |
| Train max samples | `1536` |
| Val max samples | `128` |
| Train batch size | `64` |
| Rollouts per prompt | `2` |
| Effective rollouts per step | `128` |
| Agent workers | `16` |
| Response length | `2048` |
| Rollout max model length | `6144` |
| Actor max token length | `8192` |
| SDPO reprompt length | `4096` |
| Base rollout batched tokens | `196608` |
| Base rollout max seqs | `96` |
| Base rollout GPU utilization | `0.70` |
| SDPO batched tokens | `131072` |
| SDPO max seqs | `64` |
| SDPO GPU utilization | `0.58` |
| Activation offload for SDPO | `True` |
| Distillation top-k | `50` |
| Reliability gate threshold | `0.4` |
| Reliability gate max fraction | `0.5` |

Reasoning:

- `Qwen3-8B` is strong enough for math while still trainable with 2 GPUs.
- `train_batch_size=64` improves GPU utilization on H200 without making SDPO target tensors too large.
- `val_max_samples=128` reduces validation cost while still giving a useful signal during constrained thesis runs.
- `response_len=2048` keeps enough room for math reasoning, but the prompt asks for concise reasoning and boxed final answers.
- Qwen3 thinking mode is disabled to reduce rambling and improve throughput.
- SDPO rollout memory settings are lower than base RL settings because SDPO carries teacher inputs and distillation tensors.
- `gpu_memory_utilization` is intentionally below maximum. Hybrid FSDP + vLLM needs free memory for model transitions, LoRA synchronization, teacher tensors, and CUDA workspace.

### A100/H100 Thesis Profile

Used when:

```bash
export PHASE=thesis
export HARDWARE_PROFILE=a100
```

or:

```bash
export HARDWARE_PROFILE=h100
```

| Setting | Value |
|---|---:|
| Train steps | `10` |
| Train max samples | `1024` |
| Val max samples | `128` |
| Train batch size | `32` |
| Rollouts per prompt | `2` |
| Effective rollouts per step | `64` |
| Response length | `2048` |
| Rollout max model length | `6144` |
| Actor max token length | `8192` |
| SDPO reprompt length | `3072` on A100, `4096` on H100 |

Reasoning:

- A100/H100 memory headroom is lower than H200 in the current 2-GPU setup.
- Smaller batch size and lower SDPO memory settings reduce vLLM startup and LoRA wake-up OOM risk.
- The shorter run is intended for a time-constrained thesis comparison, not a full-scale arXiv-grade training sweep.

## Variant-Specific Overrides

### `base_rl`

```text
policy_loss.loss_mode = vanilla
include_environment_feedback = False
reliability_weighting = False
reliability_gate_threshold = 0.0
```

This is the RL baseline. It uses math reward only and does not use SDPO.

### `sdpo_vanilla`

```text
policy_loss.loss_mode = sdpo
include_environment_feedback = True
sparse_target_execution = True
reliability_weighting = False
reliability_gate_threshold = 0.0
reliability_gate_max_fraction = null
reliability_gate_sparse_execution = False
```

This is the original SDPO-style math adaptation. It uses feedback-reprompted targets but does not judge target reliability.

### `sdpo_reliability_gate`

```text
policy_loss.loss_mode = sdpo
include_environment_feedback = True
sparse_target_execution = True
reliability_weighting = True
reliability_gate_threshold = 0.4
reliability_gate_max_fraction = 0.5
reliability_gate_sparse_execution = True
```

This is the proposed improvement. It uses the same feedback source as vanilla SDPO, but applies a reliability weight and a sparse reliability gate.

## Metrics To Report

Primary result metrics:

| Metric | Meaning |
|---|---|
| `val_acc_mean` | Main validation accuracy. |
| `incorrect_format_mean` | Fraction of validation outputs with bad answer format. |
| `truncated_mean` | Fraction of outputs clipped by max response length. |
| `response_length_mean` | Average generated response length. |
| `response_length_clip_ratio` | How often responses hit the length cap. |

Training and efficiency metrics:

| Metric | Meaning |
|---|---|
| `throughput_tokens_per_s` | Overall training throughput. |
| `time_per_step_s` | Wall time per optimization step. |
| `gen_s` | Rollout generation time. |
| `old_log_prob_s` | Old log-probability computation time. |
| `update_actor_s` | Actor update time. |

SDPO metrics:

| Metric | Meaning |
|---|---|
| `sdpo_reprompt_fraction` | Fraction of samples with SDPO reprompt targets. |
| `sdpo_feedback_used_fraction` | Fraction of samples where feedback was used. |
| `sdpo_reliability_weight_mean` | Average reliability weight. |
| `sdpo_reliability_gate_threshold` | Active threshold, normally `0.4`. |
| `sdpo_reliability_gate_max_fraction` | Batch budget, normally `0.5`. |
| `sdpo_reliability_gate_eligible_fraction` | Fraction passing the reliability threshold. |
| `sdpo_reliability_gate_fraction` | Fraction actually selected as SDPO targets. |
| `sdpo_reliability_gate_compute_fraction` | Fraction computed after DP alignment. |
| `sdpo_reliability_gate_compute_token_fraction` | Token fraction spent on gated SDPO compute. |

## Expected Interpretation

Good reliability-gate behavior looks like:

- `sdpo_reliability_gate_fraction` is below vanilla SDPO reprompt fraction.
- `sdpo_reliability_gate_compute_token_fraction` is meaningfully below `1.0`.
- Validation accuracy is competitive with or better than vanilla SDPO.
- Incorrect-format and truncation rates do not increase.
- Actor update time is lower than vanilla SDPO, or the method reaches similar validation accuracy with fewer reliable SDPO targets.

If reliability-gated SDPO is faster but slightly lower accuracy, report it as an efficiency-quality tradeoff. If it is both faster and more accurate than vanilla SDPO, that is the strongest thesis result.

## Code References

- Runner: `experiments/math/run_sdpo_math_benchmark.sh`
- Hardware/profile config: `experiments/math/phase_common.sh`
- Manifest writer: `experiments/math/write_phase_manifest.py`
- Reliability weights and gate scheduling: `verl/trainer/ppo/ray_trainer.py`
- SDPO loss: `verl/trainer/ppo/core_algos.py`
- Config defaults: `verl/trainer/config/sdpo_math_a100.yaml`
