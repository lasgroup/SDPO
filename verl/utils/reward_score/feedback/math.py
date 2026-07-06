# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import signal
from typing import Optional

try:
    from math_verify import parse as mv_parse, verify as mv_verify
except ImportError:  # math-verify is optional for integer-only DAPO-Math training.
    mv_parse = None
    mv_verify = None

FORMAT_PENALTY = False
FORMAT_FEEDBACK = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
TRUNCATION_FEEDBACK = "Your response was truncated because it exceeded the maximum length."
SAFE_WRONG_ANSWER_FEEDBACK = (
    "Your boxed final answer was parsed, but it is incorrect. Recheck the reasoning and final calculation."
)
VALID_FEEDBACK_MODES = {"none", "safe", "oracle"}


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""#None


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    #assert s[: len(left)] == left, f"box error: {s}"
    #assert s[-1] == "}", f"box error: {s}"
    if s[: len(left)] == left and  s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


def normalize_answer(answer: str) -> str:
    """Normalize simple final answers before exact comparison.

    DAPO-Math processed English answers are all integer strings, but model outputs
    may include harmless wrappers or separators.
    """
    answer = str(answer or "").strip()
    answer = answer.strip("$")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace(",", "")
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    boxed = last_boxed_only_string(answer)
    if boxed is not None:
        answer = remove_boxed(boxed)
    return re.sub(r"\s+", "", answer)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return normalize_answer(extracted_pred or "") == normalize_answer(gt), extracted_pred


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    # try Math-Verify equivalence check
    if not correct and pred != "" and mv_parse is not None and mv_verify is not None:
        try:
            with timeout(seconds=5):
                gold_expr = mv_parse(answer)
                pred_expr = mv_parse(pred)
                correct = mv_verify(gold_expr, pred_expr)
        except Exception:  # ignore any parsing/verification errors
            pass
    return correct, pred


def resolve_feedback_mode(extra_info: Optional[dict], format_feedback: bool, correctness_feedback: bool) -> str:
    """Resolve feedback behavior while preserving legacy keyword controls."""
    if not format_feedback:
        return "none"
    if correctness_feedback:
        return "oracle"

    feedback_mode = "safe"
    if extra_info is not None:
        feedback_mode = str(extra_info.get("feedback_mode", feedback_mode)).lower()
    if feedback_mode not in VALID_FEEDBACK_MODES:
        raise ValueError(f"Invalid math feedback_mode={feedback_mode!r}. Expected one of {sorted(VALID_FEEDBACK_MODES)}.")
    return feedback_mode


def build_feedback(
    *,
    correct: bool,
    incorrect_format: bool,
    was_truncated: bool,
    ground_truth: str,
    feedback_mode: str,
) -> str:
    if feedback_mode == "none" or correct:
        return ""
    if was_truncated:
        return TRUNCATION_FEEDBACK
    if incorrect_format:
        return FORMAT_FEEDBACK
    if feedback_mode == "oracle":
        return f"Your answer is incorrect. The correct answer is {ground_truth}."
    return SAFE_WRONG_ANSWER_FEEDBACK


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    extra_info = extra_info or {}
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    feedback_mode = resolve_feedback_mode(extra_info, format_feedback, correctness_feedback)
    feedback = build_feedback(
        correct=correct,
        incorrect_format=incorrect_format,
        was_truncated=was_truncated,
        ground_truth=ground_truth,
        feedback_mode=feedback_mode,
    )

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback_mode": feedback_mode,
        "math_verify_available": 1 if mv_parse is not None and mv_verify is not None else 0,
        "feedback": feedback,
    }
