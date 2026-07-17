# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def extract_information_blocks(solution_str):
    """Extract the text of all <information>...</information> blocks that were
    inserted into the rollout (i.e. the documents the model actually retrieved).
    Returns a single normalized string for grounding checks."""
    blocks = re.findall(r"<information>(.*?)</information>", solution_str, re.DOTALL)
    return normalize_answer(" ".join(blocks)) if blocks else ""


def compute_score_grounded(solution_str, ground_truth, method='strict',
                           format_score=0., score=1., ungrounded_score=0.3):
    """behavcf groundedness-gated reward (single-forward proxy of behavioral counterfactual).

    Idea (mirrors behav_cf rwd_A): an answer is only fully rewarded if it is BOTH
    correct (EM=1) AND actually supported by the documents the model retrieved in
    its own rollout (<information> blocks). A correct answer that does NOT appear in
    the retrieved context is treated as a suspected PARAM_HALL (parametric-memory /
    tool hallucination) and is discounted, creating a gradient that prefers genuinely
    tool-grounded behavior over memorized guessing.

    This is Goodhart-resistant: to obtain the full reward the policy must make the
    answer appear in the retrieved evidence (a grounded behavior), which cannot be
    faked by manipulating internal activations.

        answer is None                         -> 0
        EM=1 and answer in retrieved docs      -> score (1.0)   [TRUE_TOOL]
        EM=1 and answer NOT in retrieved docs  -> ungrounded_score (0.3) [PARAM_HALL]
        EM=0                                   -> format_score (0)
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if answer is None:
        if do_print:
            print(f"--------------------------------")
            print(f"[grounded] no answer extracted")
        return 0

    correct = em_check(answer, ground_truth['target'])
    if not correct:
        return format_score

    info_norm = extract_information_blocks(solution_str)
    ans_norm = normalize_answer(answer)
    grounded = bool(ans_norm) and (ans_norm in info_norm)

    if do_print:
        print(f"--------------------------------")
        print(f"[grounded] golden={ground_truth['target']} answer={answer} "
              f"grounded={grounded} reward={score if grounded else ungrounded_score}")

    return score if grounded else ungrounded_score


# ---------------------------------------------------------------------------
# Legacy behavioral-trust reward variants
# ---------------------------------------------------------------------------

def tool_use_causal_proxy(solution_str: str) -> float:
    """Method B: causal/temporal proxy for tool use.

    Logic:
      - Did the model generate <search> AND receive <information>?
      - Was the <answer> generated AFTER the last </information> block?
        (i.e., the answer generation was causally downstream of retrieval)
      - Was at least one search query non-trivial (has word-overlap with context)?

    Returns a float in [0, 1]:
      1.0  : answer after retrieval, search query meaningful       [TRUE_TOOL]
      0.6  : answer after retrieval, but query overlap weak        [PROBABLE_TOOL]
      0.2  : searched but answer before last info / no info block  [FORMAT_FAKE]
      0.0  : no search at all                                      [NO_TOOL]
    """
    searches = re.findall(r"<search>(.*?)</search>", solution_str, re.DOTALL)
    info_ends = [m.end() for m in re.finditer(r"</information>", solution_str)]
    answer_start = solution_str.rfind("<answer>")

    if not searches:
        return 0.0  # no tool call at all

    if not info_ends or answer_start == -1:
        return 0.2  # called search but no info injected / no answer

    last_info_end = info_ends[-1]
    if answer_start < last_info_end:
        return 0.2  # answer was generated before the last info block -> format fake

    # answer is causally downstream of retrieval
    # check query quality: does any query share content words with the rollout context?
    context_words = set(normalize_answer(solution_str[:answer_start]).split())
    for q in searches:
        q_words = set(normalize_answer(q).split())
        if len(q_words & context_words) >= 2:  # meaningful overlap
            return 1.0
    return 0.6  # answer after info, but queries are weak/generic


def compute_score_eco_b(solution_str, ground_truth, floor=0.4, budget=2,
                         format_score=0., bonus=0.2, penalty=0.1):
    """Legacy reward using Method B (causal tool-use proxy) for D_trust.

    R = D_trust_B / (1 + n_search/budget)   if correct & searched
      = 1 + bonus                            if correct & not searched
      = -penalty                             if wrong & not searched
      = 0                                    if wrong & searched
    """
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0
    n_search = len(re.findall(r"<search>", solution_str))
    correct = bool(em_check(answer, ground_truth['target']))
    if n_search == 0:
        return (1.0 + bonus) if correct else (-penalty)
    if not correct:
        return format_score
    d_trust = tool_use_causal_proxy(solution_str)
    # if proxy says "format fake" (0.2) treat as floor
    d_trust = max(d_trust, floor)
    C = n_search / max(1e-6, budget)
    return d_trust / (1.0 + C)


def compute_score_eco_bc(solution_str, ground_truth, floor=0.2, budget=2,
                          format_score=0., bonus=0.2, penalty=0.1):
    """Legacy reward combining Method B (causal proxy) + Method A (substring).

    D_trust levels (richer than either alone):
      B=1.0 & A=1.0  → 1.0   (causally grounded + string confirmed)
      B=1.0 & A=0    → 0.7   (causally grounded, multi-hop style)
      B=0.6 & A=1.0  → 0.7   (string in doc, but query overlap weak)
      B=0.6 & A=0    → 0.4   (weakly grounded)
      B=0.2 (fake)   → floor (searched but answer before info)
    """
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0.0
    n_search = len(re.findall(r"<search>", solution_str))
    correct = bool(em_check(answer, ground_truth['target']))
    if n_search == 0:
        return (1.0 + bonus) if correct else (-penalty)
    if not correct:
        return format_score

    b_score = tool_use_causal_proxy(solution_str)
    ans_norm = normalize_answer(answer)
    info_norm = extract_information_blocks(solution_str)
    a_score = 1.0 if (bool(ans_norm) and ans_norm in info_norm) else 0.0

    if b_score >= 1.0 and a_score == 1.0:
        d_trust = 1.0
    elif b_score >= 1.0 or a_score == 1.0:
        d_trust = 0.7
    elif b_score >= 0.6:
        d_trust = 0.4
    else:
        d_trust = floor  # format fake

    C = n_search / max(1e-6, budget)
    return d_trust / (1.0 + C)


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
