PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

CODE_PROMPT = """You are a coding expert. You will be given a coding problem, and you need to write a correct Python program that matches the specification and passes all tests. The time limit is 1 second. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```.\n\n{problem}"""


# Codeforces
# instruction = """You are a coding expert. Given a competition-level coding problem, you need to write a Python program to solve it. You may start by outlining your thought process. In the end, please provide the complete code in a code block enclosed with ``` ```. The code should take stdin as input and print the output. Your program should be a Python function generated from the given prompt. Simply call the function after the definition."""

# LCB
# def prepare_livecode_bench(example):
#     if example["is_stdin"]:
#         prompt_text = (
#             "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
#             + example["question_content"]
#         )
#     else:
#         prompt_text = (
#             "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
#             + example["question_content"]
#         )
#     #print(len(prompt_text))
#     return {"prompt": prompt_text}
