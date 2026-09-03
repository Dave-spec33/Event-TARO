from math_verify import parse, verify
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig,
)

from verl.utils.reward_score.math_reward import (
    last_boxed_only_string,
    remove_boxed,
    is_equiv,
)


MATH_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
}


def _parse_answer(answer: str):
    """
    Parse a short mathematical answer using Math-Verify.

    We wrap the extracted answer in a LaTeX environment so that both
    ordinary expressions and LaTeX ground truths can be parsed.
    """
    return parse(
        f"${answer}$",
        extraction_config=[
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
        # We only parse the short boxed answer, rather than the whole CoT.
        # This also avoids signal.alarm/threading issues in some verl setups.
        parsing_timeout=None,
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """
    MATH reward:
      1. Require a final \\boxed{...}, preserving verl's existing behavior.
      2. Extract the LAST boxed answer.
      3. Use Math-Verify for mathematical equivalence.
      4. Fall back to verl's original normalized-string comparison.

    Returns:
        1.0 if correct, otherwise 0.0.
    """

    # This custom reward is intended for MATH-style datasets.
    if data_source not in MATH_SOURCES:
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    # Preserve the original verl requirement:
    # no boxed answer -> reward 0.
    boxed = last_boxed_only_string(solution_str)

    if boxed is None:
        return 0.0

    try:
        pred_answer = remove_boxed(boxed)
    except Exception:
        return 0.0

    if pred_answer is None:
        return 0.0

    # Prevent pathological model output from giving SymPy an enormous
    # symbolic expression.
    if len(pred_answer) > 1024:
        return 0.0

    # ---------------------------------------------------------
    # 1. Preferred path: symbolic/numerical equivalence
    # ---------------------------------------------------------
    try:
        gold_parsed = _parse_answer(str(ground_truth))
        pred_parsed = _parse_answer(str(pred_answer))

        if gold_parsed and pred_parsed:
            if verify(gold_parsed, pred_parsed):
                return 1.0

    except Exception as e:
        # Don't crash RL training because one expression cannot be parsed.
        # Fall through to verl's original evaluator.
        pass

    # ---------------------------------------------------------
    # 2. Fallback: verl original normalization/string comparison
    # ---------------------------------------------------------
    try:
        if is_equiv(pred_answer, ground_truth):
            return 1.0
    except Exception:
        pass

    return 0.0