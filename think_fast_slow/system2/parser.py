import re
from typing import List, Optional

def extract_answer(
    text: str,
    options: List[str],
    reasoning_trace: Optional[str] = None,
) -> str:
    """
    Parse final answer from model output or reasoning trace according to Table 1:
    "free text; answer parsed from final line".
    """
    if not options:
        return ""

    options_clean = [opt.strip() for opt in options]
    opt_lower_map = {opt.lower(): opt for opt in options_clean}

    # 1. Check final lines of text
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        last_line = lines[-1]

        # Pattern: Answer: [Choice] or Option: [Choice] or The answer is [Choice]
        match = re.search(r'(?:answer|choice|option|result)(?:\s*(?:is|:|=)\s*)([A-Za-z0-9_\-\.\s]+)', last_line, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("[]().'\"")
            if candidate.lower() in opt_lower_map:
                return opt_lower_map[candidate.lower()]
            for opt in options_clean:
                if opt.lower() in candidate.lower():
                    return opt

        # Direct exact match on last line
        clean_last = last_line.strip("[]().'\"* \t")
        if clean_last.lower() in opt_lower_map:
            return opt_lower_map[clean_last.lower()]

        for opt in options_clean:
            if re.search(r'\b' + re.escape(opt) + r'\b', last_line, re.IGNORECASE):
                return opt

    # 2. Check full text if last line didn't yield an exact match
    for opt in options_clean:
        if re.search(r'(?:final answer|answer is|correct choice is)\s*[:=]?\s*[\*\*_]*' + re.escape(opt) + r'\b', text, re.IGNORECASE):
            return opt

    # 3. Fallback to reasoning trace if text was truncated at budget boundary
    if reasoning_trace:
        r_lines = [l.strip() for l in reasoning_trace.strip().split("\n") if l.strip()]
        if r_lines:
            for line in reversed(r_lines[-5:]):
                for opt in options_clean:
                    if re.search(r'(?:therefore|thus|hence|conclude|answer is)\s*.*?\b' + re.escape(opt) + r'\b', line, re.IGNORECASE):
                        return opt

    # 4. Fallback to first option if nothing matched
    return options_clean[0]
