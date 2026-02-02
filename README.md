<div align="center">

# Reinforcement Learning via Self-Distillation (SDPO)

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.20802)  [![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/lasgroup/SDPO) [![W&B Logs](https://img.shields.io/badge/WandB%20Logs-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/jonhue/SDPO?nw=mgotcx6kk7)

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#-introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a> •
    <a href="#-main-results" style="text-decoration: none; font-weight: bold;">📊 Main Results</a> •
    <a href="#-getting-started" style="text-decoration: none; font-weight: bold;">🚀 Getting Started</a>
  </p>
  <p>
    <a href="#usage-documentation" style="text-decoration: none; font-weight: bold;">Usage Documentation</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a>
  </p>
</div>

## 📖 Introduction

Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain *why* an attempt failed. We formalize this setting as *Reinforcement Learning with Rich Feedback* (RLRF):

<p align="center">
<img src="figures/sdpo-fig-training-loop.png" alt="Reinforcement Learning from Rich Feedback" width="80%">
</p>

**We propose Self-Distilled Policy Optimization (SDPO)**, a reinforcement learning framework that augments on-policy optimization with self-distillation from the model’s own high-reward trajectories.

SDPO converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context.

<p align="center">
<img src="figures/sdpo-fig.png" alt="SDPO" width="80%">
</p>

---

## 📊 Main Results

### Learning without Rich Environment Feedback

<!-- SDPO turns successful rollouts into reusable supervision, allowing the policy to directly learn from its own best generations—without requiring external demonstrations or additional datasets. -->

When environment feedback is sparse or rule-based, standard reinforcement learning methods struggle to propagate learning signals efficiently. SDPO addresses this by reusing high-reward rollouts as implicit feedback, providing dense supervision even in the absence of rich environment feedback.

<p align="center">
<img src="figures/chemistry-accuracy-response.png" alt="SDPO Performance vs. Training Steps" width="80%">
</p>

*Training progression of Olmo3-7B-Instruct on Chemistry. We report the average accuracy across 16 samples per question and a rolling average of response lengths over 5 steps. We report GRPO with the optimal hyperparameters for this model and task.*

<p align="center">
<img src="figures/table-no-rich-feedback.png" alt="SDPO Performance without Rich Environment Feedback" width="80%">
</p>

***Comparison of SDPO and GRPO on reasoning-related benchmarks.** We report the highest achieved avg@16 within 1 hour and 5 hours of wall-clock training time, respectively.
Both SDPO and on-policy GRPO perform one gradient step per generation batch, while GRPO performs 4 off-policy mini batch steps. We select optimal hyperparameters for SDPO and baselines based on 5h accuracy. Each run is performed on a node with 4 NVIDIA GH200 GPUs. Together with initialization and validation, each run takes approximately 6 hours.*

---

### Learning with Rich Environment Feedback

In settings where environments provide structured or textual feedback, SDPO naturally incorporates this information into self-distillation. By conditioning future attempts on both successful demonstrations and feedback from failed attempts, SDPO achieves faster convergence and more stable training.

<p align="center">
<img src="figures/lcbv6-accuracy.png" alt="SDPO Performance with Rich Environment Feedback" width="80%">
</p>

***SDPO with rich environment feedback.**
Left: SDPO benefits from denser credit assignment (logit > token > sequence-level) and consistently outperforms GRPO when rich feedback is available.
Right: The self-teacher improves throughout training, and the final student substantially surpasses the initial teacher. Error bars show variability across seeds.*

---

### Solving Hard Questions via Test-Time Self-Distillation

SDPO also enables **test-time self-distillation**. By generating multiple candidate solutions, identifying high-quality responses, and reusing them as demonstrations, the model can iteratively refine its outputs at inference time.  This leads to substantial gains on hard reasoning tasks without additional training.

<p align="center">
<img src="figures/very-hard-questions.png" alt="Test-Time Self-Distillation" width="80%">
</p>

***Test-time self-distillation on hard coding problems.**
SDPO solves questions that neither the base model nor multi-turn interaction can solve, achieving higher solution discovery rates across generation budgets.*

---

## 🚀 Getting Started

### System Requirements
*   **Operating System:** Linux (Tested on SLES 15 SP5 and Ubuntu 22.04)
*   **Hardware:** NVIDIA GPUs (CUDA compatible)
*   **Python:** 3.12 (Tested on 3.12.3)
*   **CUDA Driver:** Compatible with the PyTorch version installed (see below).

### Installation

1. **Install PyTorch:**
Choose the version matching your GPU architecture:

*   **For Ampere/Hopper (RTX 30/40, H100):**
    ```bash
    # PyTorch 2.5.1 (Stable for CUDA 12.4)
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

*   **For Blackwell (RTX 50, RTX PRO 2000 Blackwell):**
    ```bash
    # PyTorch 2.7.0 (Support for CUDA 12.8 / sm_120)
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

2. **Install SDPO and Dependences:**
From the root of the repository:
```bash
# Install core dependencies (includes Blackwell fixes)
pip install -r requirements.txt

# Install SGLang/vLLM stack (recommended for local high-perf)
pip install -r requirements_sglang.txt

# Install SDPO (verl) in editable mode
pip install -e .

# Install Flash Attention 2 (compiled from source)
pip install -r requirements-cuda.txt --no-build-isolation
```

### Advanced Components (vLLM & SGLang)
This codebase supports vLLM and SGLang for high-throughput inference, which significantly accelerates the rollout phase of reinforcement learning. While optional for basic usage, they are recommended for large-scale training.

```bash
# Install SGLang/vLLM stack
pip install -r requirements_sglang.txt
```
*Note: This command installs specific versions of SGLang and vLLM compatible with this codebase. Ensure your NVIDIA drivers are compatible with the installed CUDA toolkit.*

> [!NOTE]
> For more specific instructions on `verl` architecture and advanced configuration, refer to the [official verl repository](https://github.com/volcengine/verl).

### Data Preparation

#### Data Loading
The detailed instructions for loading the data are provided in `data/README.md`.

One example is provided below:
```bash
python data/load_dataset.py \
    --dataset_name Chemistry \
    --output_path datasets/chemistry.json
```

To split the data into train and test sets, run the following command:
```bash
python data/split_tasks.py \
    --json_path datasets/chemistry.json \
    --output_dir datasets/sciknoweval/chemistry \
    --test_ratio 0.1 \
    --seed 42
```

For `LiveCodeBenchv6` split the _unit tests_ into train and test sets, run the following command:
```bash
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
```


#### Data Preprocessing
Our implementation uses the `parquet` format for the data. To preprocess the data, run the following command:

```bash
python data/preprocess.py \
    --data_source DATASET_PATH
```
`DATASET_PATH` should contain the `train.json` and `test.json` files.

### Configuration
Before running experiments, adapt the paths in `verl/trainer/config/user.yaml` to your environment:

```yaml
vars:
  dir: /path/to/your/SDPO              # Path to the SDPO repository
  log_dir: /path/to/your/logs          # Directory for logs
  ckpt_dir: /path/to/your/checkpoints  # Directory for model checkpoints
```

### Training

#### Reproducing Results (Without Rich Environment Feedback)

Run the following commands to reproduce the results without rich environment feedback.

**GRPO baseline:**
```bash
bash experiments/generalization/run_baseline_grpo_all.sh
```

**SDPO:**
```bash
bash experiments/generalization/run_sdpo_all.sh
```

#### Reproducing Results (With Rich Environment Feedback)
Run the following commands to reproduce the results with rich environment feedback.

**GRPO baseline:**
```bash
bash experiments/rich_feedback/run_baseline_grpo.sh
```

**SDPO:**
```bash
bash experiments/rich_feedback/run_sdpo.sh
```

### Multi-turn Baseline of Section 5

Prepare the data by splitting it into individual tasks:

```
export MY_DATA_SPLITS_DIR=lcb_v6
export MY_DATA_SINGLES_DIR=lcb_v6_singles
bash dat/prepare_data_splits.sh datasets/lcb_v6.json
```

Run the multi-turn baseline for, e.g., question 120:

```
python baseline_multiturn/multiturn.py --data-dir=lcb_v6_singles/q_120 --run-name multiturn_q120
```

Or, for all hard questions:

```
bash experiments/ttt/run_multiturn_all.sh
```

---

## Usage Documentation

This section documents the configuration options added by SDPO on top of the base verl framework.

### Policy Loss Configuration

Located at `actor.policy_loss` in the config.

- **loss_mode** (str, default: `"vanilla"`): Loss function mode. Set to `"sdpo"` to enable self-distillation. Options: `vanilla`, `sdpo`.

### Self-Distillation Configuration

Located at `actor.self_distillation` in the config. Only active when `actor.policy_loss.loss_mode = "sdpo"`.

#### Core Settings

- **full_logit_distillation** (bool, default: `True`): Whether to use full-logit KL distillation.

- **alpha** (float, default: `0.5`): KL interpolation coefficient. `0.0` = forward KL, `1.0` = reverse KL, `0.5` = JSD.

- **success_reward_threshold** (float, default: `1.0`): Minimum sequence reward to be considered a successful demonstration.

- **teacher_regularization** (str, default: `"ema"`): Teacher regularization mode. Options: `ema`, `trust-region`. Note: if `ema` is used, the model on the `RefWorker` is updated as an exponential moving average. `trust-region` requires `use_fused_kernels = False`.

- **teacher_update_rate** (float, default: `0.05`): EMA update rate for teacher weights, or trust-region mixing coefficient.

- **distillation_topk** (int | None, default: `100`): If set, use top-k logits for distillation instead of full distribution.

- **distillation_add_tail** (bool, default: `True`): Whether to add a tail bucket for top-k distillation.

- **is_clip** (float | None, default: `2.0`): Clip value for importance sampling ratio. `None` disables IS weighting.

#### Reprompting Settings

- **max_reprompt_len** (int, default: `10240`): Maximum token length of the reprompted prompt.

- **reprompt_truncation** (str, default: `"right"`): Truncation method for reprompted prompts. Options: `left`, `right`, `error`.

- **dont_reprompt_on_self_success** (bool, default: `True`): If `True`, don't use a sample's own successful response as demonstration.

- **remove_thinking_from_demonstration** (bool, default: `True`): Whether to remove `<think>...</think>` tags from demonstrations.

#### Template Settings

- **reprompt_template** (str): Main template for reprompting. Placeholders: `{prompt}`, `{solution}`, `{feedback}`.

- **solution_template** (str): Template for the solution section. Placeholder: `{successful_previous_attempt}`.

- **feedback_template** (str): Template for the feedback section. Placeholder: `{feedback_raw}`.

#### Feedback Settings

- **include_environment_feedback** (bool, default: `True`): Whether to include environment feedback (e.g., test errors) in reprompting.

- **environment_feedback_only_without_solution** (bool, default: `True`): If `True`, only use feedback when no successful solution is available.

---

## Citation
If you find this work helpful, please cite us.

```bibtex
@article{hubotter2026reinforcement,
  title = {Reinforcement Learning via Self-Distillation},
  author = {Hübotter, Jonas and Lübeck, Frederike and Behric, Lejs and Baumann, Anton and Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas},
  year = {2026},
  journal = {arXiv preprint arXiv:2601.20802},
}
```

## Attribution

Our implementation is based on a recent version of [verl](https://github.com/verl-project/verl).
