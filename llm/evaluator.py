from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datasets import load_dataset
from llm.bridge import evaluate_story, TrialResult


# ── results aggregator ────────────────────────────────────────────────────────

@dataclass
class BenchmarkResults:
    trials:             list[TrialResult] = field(default_factory=list)

    def add(self, trial: TrialResult):
        self.trials.append(trial)

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def baseline_acc(self) -> float:
        if self.n == 0: return 0.0
        return sum(t.baseline_correct for t in self.trials) / self.n

    @property
    def cot_acc(self) -> float:
        if self.n == 0: return 0.0
        return sum(t.cot_correct for t in self.trials) / self.n

    @property
    def epistemic_acc(self) -> float:
        if self.n == 0: return 0.0
        return sum(t.epistemic_correct for t in self.trials) / self.n
    
    def summary_table(self) -> str:
        if self.n == 0:
            return "\nNo results to display.\n"
        lines = [
            "",
            "╔══════════════════════════════════════════════╗",
            "║         EpistemicLLM — Benchmark Results     ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Stories evaluated   : {self.n:<22} ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Baseline (zero-shot): {self.baseline_acc*100:>6.1f}%                ║",
            f"║  Chain-of-Thought    : {self.cot_acc*100:>6.1f}%                ║",
            f"║  EpistemicLLM (ours) : {self.epistemic_acc*100:>6.1f}%                ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Improvement vs baseline : +{(self.epistemic_acc - self.baseline_acc)*100:.1f}%             ║",
            f"║  Improvement vs CoT      : +{(self.epistemic_acc - self.cot_acc)*100:.1f}%             ║",
            "╚══════════════════════════════════════════════╝",
            "",
        ]
        return "\n".join(lines)

    def failure_analysis(self) -> str:
        """
        Shows stories where our system failed — for error analysis section
        of the technical report.
        """
        failures = [t for t in self.trials if not t.epistemic_correct]
        if not failures:
            return "No failures — perfect score.\n"
        lines = [f"\n── Failure analysis ({len(failures)} cases) ──"]
        for i, t in enumerate(failures):
            lines.append(f"\n[{i+1}] Query: Does {t.agent} know '{t.fact}'?")
            lines.append(f"     Expected : {t.expected}")
            lines.append(f"     Got      : {t.epistemic_answer}")
            lines.append(f"     Story    : {t.story[:120]}...")
        return "\n".join(lines)

    def save_json(self, path: str):
        data = []
        for t in self.trials:
            data.append({
                "story"            : t.story[:200],
                "agent"            : t.agent,
                "fact"             : t.fact,
                "expected"         : t.expected,
                "baseline_correct" : t.baseline_correct,
                "cot_correct"      : t.cot_correct,
                "epistemic_correct": t.epistemic_correct,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


# ── ToMi dataset loader ───────────────────────────────────────────────────────

def load_tomi_stories(
    txt_path: str = "data/fb_all_test.txt",
    trace_path: str = "data/fb_all_test.trace",
    max_stories: int = 50,
    question_types: list[str] | None = None
) -> list[str]:
    """
    Loads and parses ToMi stories from local files.

    Args:
        txt_path       : path to the .txt file
        trace_path     : path to the .trace file
        max_stories    : max number of stories to load
        question_types : filter by question type
                         None = load all
                         ["first_order_1_tom"] = only false belief questions
                         ["first_order_0_no_tom"] = only true belief questions
    """
    if question_types is None:
        question_types = ["first_order_1_tom", "first_order_0_no_tom"]

    print(f"Loading ToMi from {txt_path}...")

    # read both files
    with open(txt_path, "r", encoding="utf-8") as f:
        txt_lines = f.readlines()
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_lines = f.readlines()

    stories     = []
    story_lines = []
    trace_idx   = 0

    for line in txt_lines:
        line = line.strip()
        if not line:
            continue

        # extract line number and content
        parts    = line.split(" ", 1)
        line_num = int(parts[0])
        content  = parts[1] if len(parts) > 1 else ""

        if line_num == 1 and story_lines:
            # new story starting — save previous if any
            story_lines = []

        story_lines.append((line_num, content))

        # check if this is the question line (contains tab)
        if "\t" in content:
            question_part = content.split("\t")
            question_text = question_part[0].strip()
            answer        = question_part[1].strip() if len(question_part) > 1 else ""

            # get the trace for this story
            story_type = ""
            if trace_idx < len(trace_lines):
                trace      = trace_lines[trace_idx].strip()
                story_type = trace
                trace_idx += 1

            # filter by question type
            matched = any(qt in story_type for qt in question_types)
            if not matched:
                story_lines = []
                continue

            # build the story narrative
            narrative_lines = []
            for num, text in story_lines[:-1]:  # exclude question line
                # skip irrelevant lines (dislikes, etc.)
                if any(skip in text for skip in ["dislikes", "likes", "is bored"]):
                    continue
                narrative_lines.append(text)

            narrative = "\n".join(narrative_lines)
            full      = f"{narrative}\n{question_text}"

            stories.append(full)
            story_lines = []

            if len(stories) >= max_stories:
                break

    print(f"Loaded {len(stories)} ToMi stories (types: {question_types}).")
    return stories


# ── manual fallback stories ───────────────────────────────────────────────────

MANUAL_STORIES = [
    """Sally and Anne are in the room. Sally puts the ball in the basket.
Sally leaves the room. Anne moves the ball from the basket to the box.
Sally comes back into the room.
Where does Sally think the ball is?""",

    """John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Peter enters the kitchen and sees the milk in the cupboard.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?""",

    """Emma and Liam are in the living room. Emma puts the book on the shelf.
Liam leaves the living room. Emma moves the book from the shelf to the drawer.
Noah enters the living room and sees the book in the drawer.
Emma leaves the living room.
Where does Liam think the book is?""",

    """Oliver and Sophia are in the garage. Oliver puts the keys in the toolbox.
Oliver goes to the backyard. Sophia moves the keys from the toolbox to the cabinet.
Oliver comes back to the garage.
Where does Oliver think the keys are?""",

    """Lucas and Mia are in the office. Lucas puts the document in the folder.
Lucas leaves the office. Mia moves the document from the folder to the safe.
Lucas calls Mia on the phone. Mia tells Lucas that the document is in the safe.
Where does Lucas think the document is?""",

    """Ava and Ethan are in the classroom. Ava puts the pencil in the box.
Ava leaves the classroom. Ethan moves the pencil from the box to the bag.
Ava returns to the classroom.
Where does Ava think the pencil is?""",

    """Sofia and Noah are in the library. Sofia puts the notebook on the desk.
Noah leaves the library. Sofia moves the notebook from the desk to the shelf.
Noah comes back to the library.
Where does Noah think the notebook is?""",

    """James and Lily are in the garden. James puts the toy in the basket.
James goes inside the house. Lily moves the toy from the basket to the pot.
James comes back to the garden.
Where does James think the toy is?""",

    """Chloe and Ben are in the bedroom. Chloe puts the watch on the table.
Ben leaves the bedroom. Chloe moves the watch from the table to the drawer.
Ben returns to the bedroom.
Where does Ben think the watch is?""",

    """Tom and Anna are in the kitchen. Tom puts the apple in the bowl.
Tom goes to the garden. Anna moves the apple from the bowl to the fridge.
Anna tells Tom that the apple is in the fridge.
Where does Tom think the apple is?""",
]


# ── main benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(
    max_stories: int = 10,
    use_tomi: bool = False,
    delay_seconds: float = 1.0,
    save_path: str = "results.json"
) -> BenchmarkResults:
    """
    Runs the full 3-way benchmark.

    Args:
        max_stories    : number of stories to evaluate
        use_tomi       : if True, loads from HuggingFace ToMi dataset
                         if False, uses manual stories
        delay_seconds  : pause between stories (avoids API rate limits)
        save_path      : where to save results JSON
    """
    if use_tomi:
        print("Data source : HuggingFace ToMi dataset")
        stories = load_tomi_stories(
            txt_path       = "data/fb_all_test.txt",
            trace_path     = "data/fb_all_test.trace",
            max_stories    = max_stories,
            question_types = ["first_order_1_tom", "first_order_0_no_tom"]
        )
        if not stories:
            print("Falling back to manual stories.")
            stories = MANUAL_STORIES[:max_stories]
    else:
        print(f"Data source : Manual stories ({len(MANUAL_STORIES)} available)")
        stories = MANUAL_STORIES[:max_stories]

    results = BenchmarkResults()
    total   = len(stories)

    print(f"\nRunning benchmark on {total} stories...")
    print("="*54)

    errors = 0
    for i, story in enumerate(stories):
        print(f"\n[{i+1}/{total}] Evaluating...")
        print(f"  Story: {story[:70].strip()}...")

        try:
            trial = evaluate_story(story)
            results.add(trial)
            print(f"  Query    : Does {trial.agent} know '{trial.fact}'?")
            print(f"  Expected : {trial.expected}")
            print(f"  Baseline : {'✓' if trial.baseline_correct  else '✗'}  ")
            print(f"  CoT: {'✓' if trial.cot_correct else '✗'}  ")
            print(f"  EpistemicLLM: {'✓' if trial.epistemic_correct else '✗'}")

        except Exception as e:
            errors += 1
            print(f"  ERROR: {e} — skipping.")
            if errors >= 3:
                print("  WARNING: 3+ errors — check API or story format.")

        if i < total - 1:
            time.sleep(delay_seconds)

    print(f"\nCompleted: {results.n} evaluated, {errors} skipped.")

    print("\n" + results.summary_table())
    print(results.failure_analysis())

    results.save_json(save_path)
    return results