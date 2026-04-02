from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
from core.kripke import KripkeModel


class FormulaType(Enum):
    ATOM        = "atom"         # a basic fact: "ball_in_box"
    NOT         = "not"          # negation: ¬φ
    AND         = "and"          # conjunction: φ ∧ ψ
    OR          = "or"           # disjunction: φ ∨ ψ
    KNOWS       = "knows"        # Kᵢφ  — agent i knows φ
    BELIEVES    = "believes"     # Bᵢφ  — agent i believes φ is possible
    NESTED      = "nested"       # Kᵢ(Kⱼ φ) — nested knowledge


@dataclass
class Formula:
    """
    A recursive epistemic logic formula.
    Examples:
      Formula.atom("ball_in_box")
      Formula.knows("sally", Formula.atom("ball_in_box"))
      Formula.knows("anne", Formula.not_(Formula.knows("sally", Formula.atom("ball_in_box"))))
    """
    type: FormulaType
    atom: str | None = None
    agent: str | None = None
    sub: Formula | None = None
    left: Formula | None = None
    right: Formula | None = None

    # ── constructors (clean API) ──────────────────────────────────
    @staticmethod
    def atom(name: str) -> Formula:
        return Formula(type=FormulaType.ATOM, atom=name)

    @staticmethod
    def not_(sub: Formula) -> Formula:
        return Formula(type=FormulaType.NOT, sub=sub)

    @staticmethod
    def and_(left: Formula, right: Formula) -> Formula:
        return Formula(type=FormulaType.AND, left=left, right=right)

    @staticmethod
    def or_(left: Formula, right: Formula) -> Formula:
        return Formula(type=FormulaType.OR, left=left, right=right)

    @staticmethod
    def knows(agent: str, sub: Formula) -> Formula:
        return Formula(type=FormulaType.KNOWS, agent=agent, sub=sub)

    @staticmethod
    def believes(agent: str, sub: Formula) -> Formula:
        return Formula(type=FormulaType.BELIEVES, agent=agent, sub=sub)

    def __repr__(self) -> str:
        match self.type:
            case FormulaType.ATOM:     return self.atom
            case FormulaType.NOT:      return f"¬({self.sub})"
            case FormulaType.AND:      return f"({self.left} ∧ {self.right})"
            case FormulaType.OR:       return f"({self.left} ∨ {self.right})"
            case FormulaType.KNOWS:    return f"K_{self.agent}({self.sub})"
            case FormulaType.BELIEVES: return f"B_{self.agent}({self.sub})"


@dataclass
class ProofStep:
    formula: str
    world: str
    result: bool
    reason: str


class Reasoner:
    """
    Evaluates epistemic logic formulas against a KripkeModel.
    Returns both a boolean result and a full proof trace.
    """

    def __init__(self, model: KripkeModel):
        self.model = model
        self.trace: list[ProofStep] = []

    def reset_trace(self):
        self.trace = []

    def evaluate(self, formula: Formula, world: str | None = None) -> bool:
        world = world or self.model.actual_world
        return self._eval(formula, world)

    def _eval(self, formula: Formula, world: str) -> bool:
        match formula.type:

            case FormulaType.ATOM:
                result = self.model.worlds[world].holds(formula.atom)
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason=f"fact lookup in world '{world}'"
                ))
                return result

            case FormulaType.NOT:
                result = not self._eval(formula.sub, world)
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason="negation"
                ))
                return result

            case FormulaType.AND:
                left  = self._eval(formula.left, world)
                right = self._eval(formula.right, world)
                result = left and right
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason=f"conjunction: {left} ∧ {right}"
                ))
                return result

            case FormulaType.OR:
                left  = self._eval(formula.left, world)
                right = self._eval(formula.right, world)
                result = left or right
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason=f"disjunction: {left} ∨ {right}"
                ))
                return result

            case FormulaType.KNOWS:
                reachable = self.model.accessible_worlds(formula.agent, world)
                results = {w: self._eval(formula.sub, w) for w in reachable}
                result = all(results.values())
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason=(
                        f"{formula.agent} can reach {reachable}. "
                        f"Sub-formula values: {results}. "
                        f"All true? {result}"
                    )
                ))
                return result

            case FormulaType.BELIEVES:
                reachable = self.model.accessible_worlds(formula.agent, world)
                results = {w: self._eval(formula.sub, w) for w in reachable}
                result = any(results.values())
                self.trace.append(ProofStep(
                    formula=str(formula),
                    world=world,
                    result=result,
                    reason=(
                        f"{formula.agent} can reach {reachable}. "
                        f"Sub-formula values: {results}. "
                        f"Any true? {result}"
                    )
                ))
                return result

    def proof_report(self) -> str:
        lines = ["── Proof trace ──────────────────────────"]
        for i, step in enumerate(self.trace):
            tick = "✓" if step.result else "✗"
            lines.append(f"[{i+1}] {tick} {step.formula}  @{step.world}")
            lines.append(f"     reason: {step.reason}")
        lines.append("─────────────────────────────────────────")
        return "\n".join(lines)