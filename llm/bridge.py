from __future__ import annotations
import os
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
from nlp.parser import parse_story, extract_story_structure, build_tracker_from_structure, build_query_from_tracker
from core.reasoner import Reasoner, Formula

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """
    Stores the result of evaluating one story under all 3 systems.
    """
    story:              str
    question:           str
    agent:              str
    fact:               str

    # ground truth — computed by our formal system
    expected:           bool

    # system answers
    baseline_answer:    bool
    cot_answer:         bool
    epistemic_answer:   bool

    # correctness flags
    baseline_correct:   bool
    cot_correct:        bool
    epistemic_correct:  bool

    # evidence
    proof_trace:        str
    kripke_snapshot:    str

    def report(self) -> str:
        lines = [
            f"{'='*54}",
            f"Story    : {self.story[:80]}...",
            f"Query    : Does {self.agent} know '{self.fact}'?",
            f"Expected : {self.expected}",
            f"{'─'*54}",
            f"Baseline    : {self.baseline_answer}  {'✓' if self.baseline_correct  else '✗'}",
            f"CoT         : {self.cot_answer}  {'✓' if self.cot_correct       else '✗'}",
            f"EpistemicLLM: {self.epistemic_answer}  {'✓' if self.epistemic_correct else '✗'}",
            f"{'='*54}",
        ]
        return "\n".join(lines)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip().lower()


def _parse_yes_no(raw: str) -> bool:
    """
    Parses a yes/no answer from the LLM.
    Returns True if the LLM answered yes, False otherwise.
    """
    raw = raw.lower()
    if "yes" in raw:
        return True
    if "no" in raw:
        return False
    # fallback — if ambiguous, treat as False
    return False


# ── the 3 systems ─────────────────────────────────────────────────────────────

def run_baseline(story: str, agent: str, fact: str) -> bool:
    """
    System 1 — Zero-shot LLM.
    Ask the LLM directly, no reasoning instructions.
    """
    obj, location = fact.split("_at_")
    prompt = (
        f"{story}\n\n"
        f"Question: Does {agent} know that the {obj} is in the {location}?\n"
        f"Answer only 'yes' or 'no'."
    )
    raw = _ask_llm(prompt)
    return _parse_yes_no(raw)


def run_cot(story: str, agent: str, fact: str) -> bool:
    """
    System 2 — Chain-of-Thought LLM with explicit ToM instructions.
    We give the LLM the best possible prompt to be fair.
    """
    obj, location = fact.split("_at_")
    prompt = (
        f"{story}\n\n"
        f"To answer correctly, follow these steps:\n"
        f"1. List every agent in the story.\n"
        f"2. For each agent, list which events they were physically present for.\n"
        f"3. An agent can only know facts from events they directly witnessed.\n"
        f"4. If an agent left before an event, they did not witness it.\n"
        f"5. If an agent was told something by another agent, they know it.\n"
        f"6. Based only on what {agent} witnessed or was told, does {agent} know "
        f"that the {obj} is in the {location}?\n\n"
        f"Think step by step, then end your response with exactly:\n"
        f"Answer: yes\nor\nAnswer: no"
    )
    raw = _ask_llm(prompt)
    for line in reversed(raw.split("\n")):
        if "answer:" in line:
            return _parse_yes_no(line)
    return _parse_yes_no(raw)


def run_epistemic(story: str) -> tuple[bool, str, str, str, str, bool]:
    """
    System 3 — EpistemicLLM (our system).
    Returns: (answer, agent, fact, proof_trace, kripke_snapshot, expected)
    """
    tracker, structure, query = parse_story(story)

    agent    = query["agent"]
    fact     = query["fact"]
    expected = query["expected_answer"]

    r = Reasoner(tracker.model)
    answer = r.evaluate(Formula.knows(agent, Formula.atom(fact)))
    trace  = r.proof_report()
    snap   = tracker.snapshot()

    return answer, agent, fact, trace, snap, expected


# ── main evaluation function ──────────────────────────────────────────────────

def evaluate_story(story: str) -> TrialResult:
    """
    Runs all 3 systems on a single story and returns a TrialResult.
    This is the main function called by the benchmark.
    """
    # run our system first to get agent, fact, expected
    epistemic_answer, agent, fact, proof_trace, kripke_snapshot, expected = run_epistemic(story)

    # extract question string
    structure = extract_story_structure(story)
    question  = structure.get("question", "")

    # run baseline and CoT on the same agent/fact
    baseline_answer = run_baseline(story, agent, fact)
    cot_answer      = run_cot(story, agent, fact)

    return TrialResult(
        story             = story.strip(),
        question          = question,
        agent             = agent,
        fact              = fact,
        expected          = expected,
        baseline_answer   = baseline_answer,
        cot_answer        = cot_answer,
        epistemic_answer  = epistemic_answer,
        baseline_correct  = (baseline_answer  == expected),
        cot_correct       = (cot_answer       == expected),
        epistemic_correct = (epistemic_answer == expected),
        proof_trace       = proof_trace,
        kripke_snapshot   = kripke_snapshot,
    )