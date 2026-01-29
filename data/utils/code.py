import re

I = r"(?:input[ \t]*(?:format|specification|section)|input[ \t]*(?:and|/|&)[ \t]*output|input)"
EXAMPLE = "example"
EXAMPLES = "examples"
NL = r"(?:\r?\n)" # require newline (Unix/Windows)

SEPARATOR_PATTERN = re.compile(
    rf"""
    ^[ \t]*(?:\#[ \t]*)*               # optional leading markdown hashes
    (?:
         [ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*(?:{I})[ \t]*:?[ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*{NL}              # -----Input:-----, Input and Output, Input/Output, etc.
       | [ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*(?:{I}):?[ \t]*(?:[-–—−])[^\n]*{NL}   # Input - description (require dash if inline)
       | [ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*{EXAMPLES}[ \t]*(?:\d+)?[ \t]*:?[ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*{NL} # ------ Examples: ------ or Examples 1:
       | [ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*{EXAMPLE}[ \t]*(?:\d+)?[ \t]*:?[ \t]*[-=_()\[\]<>|~*–—−]*[ \t]*{NL} # --- Example 1: --- or Example:
       | .*for[ \t]+example[ \t]*:[^\n]*{NL}                                              # any line containing "For example:"
       | .*```[^\n]*{NL}                                                                   # any line containing triple backticks
       | .*-----[^\n]*{NL}                                                                   # any line containing -----
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def parse_description(problem: str, min_length: int = 10) -> str:
    """
    Return everything before the first separator line.
    The separator must be followed by a newline; matching ignores spaces inside tokens.
    """
    m = SEPARATOR_PATTERN.search(problem)
    cut = m.start() if m else len(problem)
    prefix = problem[:cut]

    # If the last non-space char is '-', drop that final line
    rprefix = prefix.rstrip()
    if rprefix.endswith('-'):
        last_nl = rprefix.rfind('\n')
        if last_nl != -1:
            prefix = prefix[:last_nl]

    if len(prefix) < min_length:
        prefix = ""
    return prefix.strip()


def float_with_default(val: str | int | float | None, default: int | float) -> float:
    if val is None:
        return default
    if isinstance(val, int) or isinstance(val, float):
        return float(val)
    if isinstance(val, str):
        if val.strip() == "":
            return default
        else:
            return float(val.split()[0])
    return default
