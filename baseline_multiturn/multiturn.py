import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
import wandb
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
from verl.utils.reward_score import multi_source_reward


MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 8192
TEMPERATURE = 1.0
TOP_P = 1
TOP_K = -1
TENSOR_PARALLEL_SIZE = 1
DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.7
ENFORCE_EAGER = False
EVAL_SPLIT = "train"
template_name = "feedback_only"
MAX_TURNS = 2880
OUTPUT_DIR = "outputs/multiturn_baseline"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "multiturn")

MAX_CONTEXT_LEN = 32000


@dataclass
class QuestionData:
    question_id: str
    prompt: str
    tests: Any
    split: str


def _load_from_parquet(question_dir: Path, split: str) -> Optional[QuestionData]:
    parquet_path = question_dir / f"{split}.parquet"
    dataset = load_dataset("parquet", data_files=str(parquet_path))["train"]
    row = dataset[0]

    prompt = str(row["prompt"])
    reward_model = row["reward_model"]
    tests = reward_model["ground_truth"]
    return QuestionData(
        question_id=question_dir.name, prompt=prompt, tests=tests, split=split
    )


def _load_question(question_dir: Path, split: str) -> QuestionData:
    parquet_data = _load_from_parquet(question_dir, split)
    if parquet_data is None:
        raise FileNotFoundError(
            f"Missing parquet file for split={split} in {question_dir}"
        )
    return parquet_data


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def _token_length(tokenizer, text: str) -> int:
    tokenized = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return len(tokenized["input_ids"])


def _build_prompt(
    tokenizer,
    question: str,
    feedback_history: list[str],
    response_history: list[str],
    template: str,
) -> str:

    feedback_prefix = (
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
    )
    feedback_suffix = "\n\nCorrectly solve the original question."

    if len(feedback_history) == 0:  # first turn
        messages = [{"role": "user", "content": question}]
    else:  # multiturns
        if template == "feedback_only":
            # include only feedback history
            all_feedback = "\n\n".join(feedback_history)
            messages = [
                {
                    "role": "user",
                    "content": f"{question}\n\n{feedback_prefix}{all_feedback}{feedback_suffix}",
                }
            ]
        elif template == "attempt_and_feedback":
            # include full conversation history
            messages = [{"role": "user", "content": question}]
            for idx, feedback in enumerate(feedback_history):
                messages.append({"role": "assistant", "content": response_history[idx]})
                messages.append(
                    {
                        "role": "user",
                        "content": (f"{feedback_prefix}{feedback}{feedback_suffix}"),
                    }
                )
        else:
            raise ValueError(f"Unknown template: {template}")
    return _apply_chat_template(tokenizer, messages)


def _log_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _get_job_id() -> str:
    for key in ("SLURM_JOB_ID", "JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        value = os.environ.get(key)
        if value:
            return value
    return "local"


def run_question(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    question: QuestionData,
    max_turns: int,
    output_path: Path,
    run_name: str,
    template_name: str,
    max_test_cases: Optional[int],
) -> Dict[str, Any]:

    turn = 0
    start_time = time.time()
    final_reward = 0.0
    context_len = 0
    max_context_len_exceeded = False
    history = []

    response_history = []
    feedback_history = []

    while True:

        prompt = _build_prompt(
            tokenizer=tokenizer,
            question=question.prompt,
            feedback_history=feedback_history,
            response_history=response_history,
            template=template_name,
        )

        prompt_length = _token_length(tokenizer, prompt)

        # trim earliest feedback if prompt exceeds max context length
        if prompt_length > MAX_CONTEXT_LEN:
            trimmed = 0
            while feedback_history and prompt_length > MAX_CONTEXT_LEN:
                feedback_history.pop(0)
                if response_history:
                    response_history.pop(0)
                trimmed += 1
                prompt = _build_prompt(
                    tokenizer=tokenizer,
                    question=question.prompt,
                    feedback_history=feedback_history,
                    response_history=response_history,
                    template=template_name,
                )
                prompt_length = _token_length(tokenizer, prompt)
            if trimmed > 0:
                print(
                    f"Prompt length exceeded {MAX_CONTEXT_LEN}. "
                    f"Trimmed {trimmed} earliest feedback entries."
                )

        context_len = prompt_length

        outputs = llm.generate([prompt], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        response_history.append(response)

        response_length = _token_length(tokenizer, response)

        # compute reward
        score = multi_source_reward.compute_score(
            solution=response,
            ground_truth=question.tests,
            reward_style="code",
            extra_info={"split": question.split, "truncated": False},
            max_test_cases=max_test_cases,
            sparse_rewards=True,
        )

        reward = float(score.get("score", 0.0))
        final_reward = reward
        feedback = score.get("feedback", "")
        now = time.time()
        feedback_history.append(feedback)

        record = {
            "run_name": run_name,
            "question_id": question.question_id,
            "split": question.split,
            "turn": turn,
            "timestamp": now,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_length,
            "response_tokens": response_length,
            "reward": reward,
            "metrics": {k: v for k, v in score.items() if k != "score"},
        }

        history.append(record)
        _log_jsonl(output_path, record)

        wandb_payload = {
            "question_id": question.question_id,
            "turn": turn,
            "reward": reward,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "acc": score.get("acc", None),
            "timed_out": score.get("timed_out", None),
            "incorrect_format": score.get("incorrect_format", None),
            "feedback_buffer_len": len(feedback_history),
        }

        wandb.log(wandb_payload, step=turn)

        if reward >= 1.0:
            # log final solution to wandb
            wandb.log(
                {
                    "final_solution": wandb.Html(
                        f"<pre style='white-space: pre-wrap;'>{response}</pre>"
                    )
                },
                step=turn,
            )
            break
        if turn >= max_turns - 1:
            break
        turn += 1

    duration = time.time() - start_time

    summary = {
        "final_reward": final_reward,
        "total_turns": turn + 1,  # started at 0
        "total_duration_sec": duration,
        "max_context_len_exceeded": max_context_len_exceeded,
        "max_context_len": context_len,
    }
    wandb.log({f"summary/{k}": v for k, v in summary.items()}, step=turn)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiturn sampling for coding questions with environment feedback."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to a directory containing a single question as a parquet file.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--template",
        type=str,
        default="feedback_only",
        help="Template name to use for prompt construction.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("PYTHONBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    question_dir = Path(args.data_dir)
    if not question_dir.exists():
        raise SystemExit(f"Question directory not found: {question_dir}")

    run_name = args.run_name
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = _get_job_id()
    seed = args.seed

    template_name = args.template
    output_path = output_dir / f"multiturn-{run_name}-{job_id}.jsonl"

    config = {
        "experiment": run_name,
        "data_dir": args.data_dir,
        "eval_split": EVAL_SPLIT,
        "template": template_name,
        "max_turns": MAX_TURNS,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
    }

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
    )
    wandb_run_name = wandb.run.name if wandb.run else run_name
    _log_jsonl(
        output_path,
        {
            "config": {**config},
            "wandb_run_name": wandb_run_name,
        },
    )
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
        seed=seed,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )
    tokenizer = hf_tokenizer(MODEL, trust_remote_code=True)

    question = _load_question(question_dir, EVAL_SPLIT)
    result = run_question(
        llm=llm,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        question=question,
        max_turns=MAX_TURNS,
        output_path=output_path,
        run_name=run_name,
        max_test_cases=None,
        template_name=template_name,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
