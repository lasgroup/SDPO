from datasets import load_dataset, Dataset


def _format_cot_example(example, including_answer: bool = True) -> str:
    choices = [chr(ord("A") + i) for i in range(16)]
    prompt = "Question:\n" + example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += f"{choices[i]}. {opt}\n"
    if including_answer:
        cot = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def _format_mmlu_pro__evalchemy(problem) -> str:
    initial_prompt = "The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice."
    subject = problem["category"]
    prompt = initial_prompt.replace("{$}", subject) + "\n" + _format_cot_example(problem, including_answer=False)
    return prompt


def _format_mmlu_pro__webinstruct(problem) -> str:
    choices = [chr(ord("A") + i) for i in range(16)]
    prompt = problem["question"] + "\n"
    prompt += "Options are:\n"
    for i, opt in enumerate(problem["options"]):
        prompt += f"{choices[i]}: {opt}\n"
    prompt +='\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the option letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
    return prompt


def _format_mmlu_pro(problem, implementation: str = "evalchemy") -> str:
    if implementation == "evalchemy":
        return _format_mmlu_pro__evalchemy(problem)
    elif implementation == "webinstruct":
        return _format_mmlu_pro__webinstruct(problem)
    else:
        raise ValueError(f"Invalid implementation: {implementation}")


def load_mmlu_pro(category: str = None, implementation: str = "evalchemy") -> Dataset:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if category is not None:
        if category == "computer_science":
            ds = ds.filter(lambda ex: ex["category"] == "computer science")
        else:
            ds = ds.filter(lambda ex: ex["category"] == category)

    return ds.map(lambda ex: {
        "kind": "mmlu_pro",
        "dataset": "mmlu_pro",
        "description": ex["question"],
        "problem": ex["question"],
        "prompt": _format_mmlu_pro(ex, implementation),
        "answer": ex["answer"],
    }, remove_columns=ds.column_names, batched=False)
