from __future__ import annotations
from core.reasoner import Formula


def parse_formula(text: str, agents: list[str]) -> Formula:
    """
    Parses a human-readable epistemic formula string into a Formula object.

    Supported syntax:
      K_agent(phi)           → agent knows phi
      NOT(phi)               → negation
      AND(phi, psi)          → conjunction
      OR(phi, psi)           → disjunction
      fact_name              → atomic fact (e.g. book_at_drawer)

    Examples:
      K_emma(book_at_drawer)
      K_emma(NOT K_liam(book_at_drawer))
      AND(K_emma(book_at_drawer), NOT K_liam(book_at_drawer))
    """
    text = text.strip()

    # K_agent(phi) — knowledge formula
    if text.startswith("K_"):
        # find agent name — everything between K_ and first (
        paren_idx = text.index("(")
        agent     = text[2:paren_idx].strip()
        inner     = _extract_inner(text, paren_idx)
        return Formula.knows(agent, parse_formula(inner, agents))

    # NOT(phi) or NOT phi
    if text.startswith("NOT"):
        rest = text[3:].strip()
        if rest.startswith("("):
            inner = _extract_inner(rest, 0)
            return Formula.not_(parse_formula(inner, agents))
        else:
            return Formula.not_(parse_formula(rest, agents))

    # AND(phi, psi)
    if text.startswith("AND("):
        inner = _extract_inner(text, 3)
        left, right = _split_args(inner)
        return Formula.and_(parse_formula(left, agents), parse_formula(right, agents))

    # OR(phi, psi)
    if text.startswith("OR("):
        inner = _extract_inner(text, 2)
        left, right = _split_args(inner)
        return Formula.or_(parse_formula(left, agents), parse_formula(right, agents))

    # atomic fact
    return Formula.atom(text.lower())


def _extract_inner(text: str, open_idx: int) -> str:
    """Extracts content between matching parentheses starting at open_idx."""
    depth = 0
    start = None
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
            if start is None:
                start = i + 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    raise ValueError(f"Unmatched parentheses in: {text}")


def _split_args(text: str) -> tuple[str, str]:
    """Splits 'phi, psi' into ('phi', 'psi') respecting nested parentheses."""
    depth   = 0
    split_i = None
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            split_i = i
            break
    if split_i is None:
        raise ValueError(f"Cannot split args in: {text}")
    return text[:split_i].strip(), text[split_i + 1:].strip()


def formula_examples(agents: list[str], facts: list[str]) -> list[str]:
    """
    Generates example formulas for the given agents and facts.
    Used in the demo to help the user understand the syntax.
    """
    examples = []
    if len(agents) >= 1 and facts:
        a = agents[0]
        f = facts[0]
        examples.append(f"K_{a}({f})")
        examples.append(f"NOT K_{a}({f})")
    if len(agents) >= 2 and facts:
        a1, a2 = agents[0], agents[1]
        f = facts[0]
        examples.append(f"K_{a1}(NOT K_{a2}({f}))")
        examples.append(f"K_{a1}(K_{a2}({f}))")
    return examples