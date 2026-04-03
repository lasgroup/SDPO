"""
Run TRL SDPOTrainer on Modal with H100 GPU(s).

Usage (single model):
    python3 -m modal run --detach trl_modal.py --model Qwen/Qwen2.5-3B-Instruct --max-steps 100

Usage (multi-GPU for large models):
    python3 -m modal run --detach trl_modal.py --model Qwen/Qwen2.5-7B-Instruct --gpu-count 2

Usage (multiple models overnight):
    python3 -m modal run --detach trl_modal.py --run-all True --max-steps 100

Usage (disable wandb):
    python3 -m modal run trl_modal.py --use-wandb false

Wandb: Create a Modal secret named 'wandb-secret' with WANDB_API_KEY:
    python3 -m modal secret create wandb-secret WANDB_API_KEY=<your-key>

With --detach, the job runs on Modal's servers. You can close your laptop
and check results later at modal.com.
"""

import modal
import os

app = modal.App("sdpo-trl")

SDPO_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "trl[vllm]>=1.0.0",
        "datasets",
        "accelerate",
        "tqdm",
        "wandb",
    )
    .add_local_dir(SDPO_LOCAL_DIR, remote_path="/root/sdpo-local")
)

# Persistent volume to store results across runs
volume = modal.Volume.from_name("sdpo-results", create_if_missing=True)


_common_kwargs = dict(
    image=image,
    volumes={"/root/results": volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("adam-hf-token"),
    ],
    timeout=18000,  # 5 hours max
)


@app.function(gpu="H100", **_common_kwargs)
def train_1gpu(
    model: str = "Qwen/Qwen2.5-3B-Instruct", **kwargs,
):
    return _train(model=model, **kwargs)


@app.function(gpu="H100:2", **_common_kwargs)
def train_2gpu(
    model: str = "Qwen/Qwen2.5-3B-Instruct", **kwargs,
):
    return _train(model=model, **kwargs)


@app.function(gpu="H100:4", **_common_kwargs)
def train_4gpu(
    model: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs,
):
    return _train(model=model, **kwargs)


def _train(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_generations: int = 8,
    max_completion_length: int = 4096,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    lr: float = 1e-5,
    max_samples: int = 0,
    max_steps: int = 100,
    use_vllm: str = "true",
    teacher_reg: str = "ema",
    dataset: str = "gpqa_physics",
    use_wandb: str = "true",
):
    use_vllm_bool = use_vllm.lower() == "true"
    use_wandb_bool = use_wandb.lower() == "true"
    import sys
    sys.path.insert(0, "/root/sdpo-local")

    import json
    import math
    import time
    import torch
    start = time.time()

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus} x {torch.cuda.get_device_name(0)}")
    print(f"Memory per GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")

    # --- Wandb setup ---
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if use_wandb_bool and wandb_api_key:
        import wandb
        model_short = model.split("/")[-1]
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        run_name = f"{model_short}_{dataset}_{max_steps}steps_gen{num_generations}_{timestamp}"
        wandb.init(
            project="sdpo",
            name=run_name,
            config={
                "model": model,
                "dataset": dataset,
                "num_generations": num_generations,
                "max_completion_length": max_completion_length,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lr": lr,
                "max_steps": max_steps,
                "use_vllm": use_vllm_bool,
                "teacher_reg": teacher_reg,
                "n_gpus": n_gpus,
            },
        )
        report_to = "wandb"
        print(f"Wandb: logging to project 'sdpo', run '{run_name}'")
    else:
        report_to = "none"
        if use_wandb_bool and not wandb_api_key:
            print("Wandb: skipped (no WANDB_API_KEY — create Modal secret 'wandb-secret')")
        else:
            print("Wandb: disabled")

    # --- HuggingFace auth for gated datasets ---
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print(f"HuggingFace: authenticated (token: {hf_token[:8]}...)")
    else:
        print("HuggingFace: no token found (HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")

    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
    from trl_train import prepare_gsm8k, gsm8k_reward, prepare_gpqa, gpqa_reward
    from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

    if dataset == "gsm8k":
        train_dataset, test_dataset = prepare_gsm8k(max_samples=max_samples)
        reward_fn = gsm8k_reward
    elif dataset == "gpqa":
        train_dataset, test_dataset = prepare_gpqa(max_samples=max_samples)
        reward_fn = gpqa_reward
    elif dataset == "gpqa_physics":
        train_dataset, test_dataset = prepare_gpqa(max_samples=max_samples, physics_only=True)
        reward_fn = gpqa_reward
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model and dataset-specific output dir
    model_tag = model.replace("/", "--")
    output_dir = f"/root/results/{dataset}/{model_tag}"
    os.makedirs(output_dir, exist_ok=True)

    config = SDPOConfig(
        output_dir=output_dir,
        # Generation
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=512,
        temperature=0.7,
        top_p=0.95,
        # Training
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        max_steps=max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SDPO-specific
        distillation_alpha=0.5,          # JSD (symmetric, matching the paper)
        full_logit_distillation=True,
        distillation_topk=100,
        sdpo_policy_loss_mode="distillation_only",
        use_successful_as_teacher=True,
        success_reward_threshold=1.0,
        teacher_regularization=teacher_reg,
        ema_update_rate=0.05 if teacher_reg == "ema" else 0.0,
        dont_reprompt_on_self_success=True,
        use_vllm=use_vllm_bool,
        # Logging
        logging_steps=1,
        report_to=report_to,
        save_strategy="no",
    )

    trainer = SDPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Patch log method to include self_distillation metrics
    _original_log = trainer.log
    def patched_log(logs, start_time=None):
        mode = "train" if trainer.model.training else "eval"
        metrics = {}
        for key, val in trainer._metrics[mode].items():
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        trainer._metrics[mode].clear()
        _original_log(logs, start_time)
    trainer.log = patched_log

    # --- Helper: evaluate on test set ---
    def evaluate_test(n_samples=50):
        """Run quick eval on a subset of the test set with batched generation."""
        import random as _rng
        model_obj = trainer.model
        _tokenizer = trainer.processing_class
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        model_obj.eval()

        samples = list(test_dataset)
        if len(samples) > n_samples:
            samples = _rng.sample(samples, n_samples)

        from trl_train import extract_number

        n_correct = 0
        eval_batch_size = 16  # process 16 questions at a time

        for i in range(0, len(samples), eval_batch_size):
            batch_samples = samples[i : i + eval_batch_size]
            prompts = [
                s["prompt"][-1]["content"] if isinstance(s["prompt"], list) else s["prompt"]
                for s in batch_samples
            ]

            inputs = _tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to("cuda")

            with torch.no_grad():
                output_ids = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_completion_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=_tokenizer.pad_token_id,
                )

            for j, sample in enumerate(batch_samples):
                prompt_len = inputs["attention_mask"][j].sum().item()
                response = _tokenizer.decode(
                    output_ids[j][prompt_len:], skip_special_tokens=True,
                )
                predicted = extract_number(response)
                expected = sample["solution"].strip().replace(",", "")
                if predicted is not None and expected:
                    try:
                        if abs(float(predicted) - float(expected)) < 1e-3:
                            n_correct += 1
                    except ValueError:
                        if predicted.strip() == expected:
                            n_correct += 1

        model_obj.train()
        return n_correct / len(samples)

    # --- Callback for periodic test eval ---
    from transformers import TrainerCallback

    class TestEvalCallback(TrainerCallback):
        def __init__(self, eval_every=10):
            self.eval_every = eval_every
            self.test_accuracies = []

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_every == 0 and state.global_step > 0:
                acc = evaluate_test(n_samples=200)
                self.test_accuracies.append({
                    "step": state.global_step,
                    "test_accuracy": acc,
                })
                print(f"  [Step {state.global_step}] Test accuracy: {acc:.1%}")
                if report_to == "wandb":
                    import wandb
                    wandb.log({"eval/test_accuracy": acc}, step=state.global_step)

    # --- Pre-training evaluation (fixed seed for consistent test set) ---
    import random as _eval_rng
    _eval_rng.seed(123)  # fixed seed so pre and post eval use same questions
    eval_samples_fixed = _eval_rng.sample(list(test_dataset), min(200, len(test_dataset)))

    def evaluate_fixed(samples):
        """Eval on the fixed sample set. Works for both GSM8K and GPQA."""
        import re as _re
        from trl_train import extract_number
        _tokenizer = trainer.processing_class
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        model_obj = trainer.model
        model_obj.eval()

        # Detect if GPQA (solution is a single letter A-D) or GSM8K (solution is a number)
        is_mcq = len(samples[0]["solution"]) == 1 and samples[0]["solution"] in "ABCD"

        n_correct = 0
        eval_batch_size = 16
        for i in range(0, len(samples), eval_batch_size):
            batch_samples = samples[i : i + eval_batch_size]
            prompts = [
                s["prompt"][-1]["content"] if isinstance(s["prompt"], list) else s["prompt"]
                for s in batch_samples
            ]
            inputs = _tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to("cuda")
            with torch.no_grad():
                output_ids = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_completion_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=_tokenizer.pad_token_id,
                )
            for j, sample in enumerate(batch_samples):
                prompt_len = inputs["attention_mask"][j].sum().item()
                response = _tokenizer.decode(output_ids[j][prompt_len:], skip_special_tokens=True)
                expected = sample["solution"].strip()

                if is_mcq:
                    # GPQA: extract last A/B/C/D
                    matches = _re.findall(r'\b([A-D])\b', response.upper())
                    predicted = matches[-1] if matches else None
                    if predicted == expected.upper():
                        n_correct += 1
                else:
                    # GSM8K: extract number
                    predicted = extract_number(response)
                    expected_clean = expected.replace(",", "")
                    if predicted is not None and expected_clean:
                        try:
                            if abs(float(predicted) - float(expected_clean)) < 1e-3:
                                n_correct += 1
                        except ValueError:
                            if predicted.strip() == expected_clean:
                                n_correct += 1
        model_obj.train()
        return n_correct / len(samples)

    print(f"\nPre-training evaluation on {len(eval_samples_fixed)} fixed test questions...")
    pre_accuracy = evaluate_fixed(eval_samples_fixed)
    print(f"Pre-training test accuracy: {pre_accuracy:.1%}")
    if report_to == "wandb":
        import wandb
        wandb.log({"eval/pre_training_accuracy": pre_accuracy}, step=0)

    eval_callback = TestEvalCallback(eval_every=10)
    if max_steps > 2:
        trainer.add_callback(eval_callback)

    trainer.train()

    train_elapsed = time.time() - start
    print(f"\nTraining done! {model} in {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

    # --- Post-training evaluation on SAME test questions ---
    print(f"\nPost-training evaluation on same {len(eval_samples_fixed)} test questions...")
    post_accuracy = evaluate_fixed(eval_samples_fixed)
    print(f"Post-training test accuracy: {post_accuracy:.1%}")
    print(f"Accuracy change: {pre_accuracy:.1%} -> {post_accuracy:.1%} ({post_accuracy - pre_accuracy:+.1%})")
    test_accuracy = post_accuracy

    elapsed = time.time() - start

    # Save log to persistent volume
    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Log saved to {log_path}")

    # Save summary
    summary = {
        "model": model,
        "dataset": dataset,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr": lr,
        "total_time_seconds": elapsed,
        "pre_training_test_accuracy": pre_accuracy,
        "post_training_test_accuracy": test_accuracy,
        "periodic_test_accuracies": eval_callback.test_accuracies,
        "log_history": log_history,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Wandb: log final metrics and finish ---
    if report_to == "wandb":
        import wandb
        wandb.log({
            "eval/pre_training_accuracy": pre_accuracy,
            "eval/post_training_accuracy": test_accuracy,
            "eval/accuracy_change": test_accuracy - pre_accuracy,
            "total_time_seconds": elapsed,
        })
        # Log periodic test accuracies as a table
        if eval_callback.test_accuracies:
            table = wandb.Table(columns=["step", "test_accuracy"],
                                data=[[e["step"], e["test_accuracy"]]
                                      for e in eval_callback.test_accuracies])
            wandb.log({"eval/periodic_accuracy": table})
        wandb.finish()

    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    num_generations: int = 8,
    max_completion_length: int = 4096,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    lr: float = 1e-5,
    max_samples: int = 0,
    max_steps: int = 100,
    run_all: bool = False,
    use_vllm: str = "true",
    teacher_reg: str = "ema",
    dataset: str = "gpqa",
    gpu_count: int = 1,
    use_wandb: str = "true",
):
    def _get_train_fn(n_gpus):
        return {1: train_1gpu, 2: train_2gpu, 4: train_4gpu}[n_gpus]

    if run_all:
        # Launch multiple models in parallel
        # 7B gets 2 GPUs so it can use vLLM + EMA like the smaller models
        model_configs = [
            # (model, batch_size, accum_steps, use_vllm, teacher_reg, gpu_count)
            # Full fine-tuning: model + AdamW (2x) + grads + vLLM copy + EMA
            # (model, batch_size, accum_steps, use_vllm, teacher_reg, gpu_count)
            ("Qwen/Qwen2.5-0.5B-Instruct", 16, 8,  True,  "ema",  1),
            ("Qwen/Qwen2.5-1.5B-Instruct", 8,  16, True,  "ema",  2),
            ("Qwen/Qwen2.5-3B-Instruct",   8,  16, True,  "ema",  2),
            ("Qwen/Qwen2.5-7B-Instruct",   16, 8,  True,  "ema",  4),
        ]
        print(f"Launching {len(model_configs)} models in parallel...")

        handles = []
        for m, bs, accum, vllm, treg, gpus in model_configs:
            print(f"  Launching {m} (batch={bs}, accum={accum}, vllm={vllm}, teacher={treg}, gpus={gpus})...")
            handle = _get_train_fn(gpus).spawn(
                model=m,
                num_generations=num_generations,
                max_completion_length=max_completion_length,
                batch_size=bs,
                gradient_accumulation_steps=accum,
                lr=lr,
                use_vllm=str(vllm).lower(),
                teacher_reg=treg,
                max_samples=max_samples,
                max_steps=max_steps,
                dataset=dataset,
                use_wandb=use_wandb,
            )
            handles.append((m, handle))

        print(f"\nAll {len(model_configs)} jobs spawned! Safe to close your laptop.")
        print("Results will be saved to the 'sdpo-results' Modal volume.")
        print("Check progress at modal.com or run:")
        print("  python3 -m modal volume ls sdpo-results")
        print("\nTo download results later:")
        print(f"  python3 -m modal volume get sdpo-results {dataset}/ ~/sdpo-local/results/{dataset}/")

    else:
        # Single model run — use .remote() for live output
        # Add --detach to the modal run command to close your laptop
        result = _get_train_fn(gpu_count).remote(
            model=model,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=lr,
            max_samples=max_samples,
            max_steps=max_steps,
            use_vllm=use_vllm,
            teacher_reg=teacher_reg,
            dataset=dataset,
            use_wandb=use_wandb,
        )

        import json
        model_tag = model.replace("/", "--")
        output_path = os.path.expanduser("~/sdpo-local/results/")
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, f"trl_{model_tag}.json")
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {filepath}")
        print(f"Total time: {result['total_time_seconds']:.0f}s")
