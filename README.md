# EpistemicLLM

**Theory of Mind Reasoning via Epistemic Logic and Large Language Models**

---

## What this is

Large Language Models fail at Theory of Mind (ToM) — they confuse what *they*
know with what *agents in a story* know. EpistemicLLM fixes this by combining:

- A **Kripke model** that formally encodes each agent's belief state
- An **epistemic reasoner** that evaluates knowledge formulas with mathematical guarantees
- An **LLM** (GPT-4o-mini) used only for what it does well: natural language parsing

The result: answers that are not just correct, but **provably correct**, with full
proof traces.

---

## Results on ToMi benchmark (44 stories)

| System | Accuracy | vs Baseline |
|---|---|---|
| Baseline (zero-shot GPT-4o-mini) | 54.5% | — |
| Chain-of-Thought (GPT-4o-mini) | 97.7% | +43.2% |
| **EpistemicLLM (this system)** | **100.0%** | **+45.5%** |

EpistemicLLM is the **only system with zero failures** on the benchmark.
It also provides a **formal proof trace** for every answer.

---

## How it works
```
Story in natural language
        │
        ▼
┌─────────────────┐
│   LLM Parser    │  GPT-4o-mini extracts agents, objects,
│  (parser.py)    │  locations, and event sequence → JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BeliefTracker  │  Replays events, builds Kripke model:
│  (tracker.py)   │  absent agents get indistinguishability edges
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Reasoner     │  Evaluates K_i(φ) formally:
│  (reasoner.py)  │  ∀v accessible to i: φ holds in v?
└────────┬────────┘
         │
         ▼
   Answer (True/False) + proof trace
```

---

## Project structure
```
EpistemicLLM/
│
├── core/
│   ├── kripke.py          # Kripke model — worlds, accessibility relations, valuations
│   ├── reasoner.py        # Epistemic query engine — evaluates K_i(φ) with proof traces
│   ├── tracker.py         # Dynamic belief state updater — processes event sequences
│   └── formula_parser.py  # Manual formula parser — supports nested formulas K_i(¬K_j(φ))
│
├── nlp/
│   ├── parser.py          # Story text → structured JSON → BeliefTracker
│   └── templates.py       # LLM prompt templates for story extraction
│
├── llm/
│   ├── bridge.py          # 3-way comparison engine (baseline vs CoT vs EpistemicLLM)
│   └── evaluator.py       # Benchmark runner — loads ToMi, runs evaluation, saves results
│
├── demo/
│   └── app.py             # Streamlit interactive demo
│
├── data/
│   ├── fb_all_test.txt    # ToMi test set (Nematzadeh et al., EMNLP 2019)
│   └── fb_all_test.trace  # ToMi metadata — story and question type labels
│
├── benchmark.py           # Main benchmark entry point
├── results_tomi.json      # Benchmark results (44 stories)
│
├── test_kripke.py         # Unit tests — Kripke model
├── test_reasoner.py       # Unit tests — epistemic reasoner
├── test_tracker.py        # Unit tests — belief tracker
├── test_correctness.py    # Formal verification — all 5 S5 axioms
├── test_parser.py         # Integration tests — full pipeline
└── test_bridge.py         # System tests — 3-way comparison
```

---

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/EpistemicLLM.git
cd EpistemicLLM

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install openai python-dotenv streamlit datasets
```

Create `.env` at the project root:
```
OPENAI_API_KEY=sk-proj-...
```

---

## Usage

### Interactive demo
```bash
streamlit run demo/app.py
```

Opens at `http://localhost:8501`. Enter any ToM story, get:
- Extracted Kripke model
- Formal epistemic query
- Proof trace
- 3-way comparison (baseline vs CoT vs EpistemicLLM)

**Advanced mode**: enter nested formulas like `K_emma(NOT K_liam(book_at_drawer))`
and evaluate them directly on the constructed model.

### Run the benchmark
```bash
python benchmark.py
```

### Run formal verification tests
```bash
python test_correctness.py
```

All 5 S5 axioms (T, 4, 5, K, consistency) are verified programmatically.

---

## Formal foundation

The system implements **epistemic logic S5** over Kripke models:

$$M = (W, \pi, \mathcal{R}_1, \ldots, \mathcal{R}_n)$$

$$M, w \models K_i\varphi \iff \forall v \in W: w\,\mathcal{R}_i\,v \Rightarrow M, v \models \varphi$$

**Key implementation decisions:**

| Concept | Implementation |
|---|---|
| Possible world | `World` — dict of boolean facts |
| Accessibility relation | `set[tuple[str,str]]` per agent |
| Agent witnesses event | No edge added between before/after worlds |
| Agent misses event | Edge added → indistinguishability |
| Communication | `KnowledgeGrant` removes contradicting edges |
| Nested formula | Recursive `_eval()` in `Reasoner` |

The engine is verified against all 5 S5 axioms in `test_correctness.py`.

---

## Supported epistemic formulas

In the advanced demo, you can type:

| Formula | Meaning |
|---|---|
| `K_sally(ball_at_box)` | Sally knows the ball is in the box |
| `NOT K_sally(ball_at_box)` | Sally does not know |
| `K_anne(NOT K_sally(ball_at_box))` | Anne knows that Sally doesn't know |
| `AND(K_anne(ball_at_box), NOT K_sally(ball_at_box))` | Anne knows, Sally doesn't |
| `K_a(K_b(NOT K_c(fact)))` | 3rd-order nested belief |

---

## Research context

> *How can formal logical reasoning be integrated with LLMs to overcome their
> limitations in theory of mind, causal reasoning, and explanation?*

**Open research directions identified:**

1. **Second-order formula extraction** — can an LLM automatically generate
   `K_i(¬K_j(φ))` from natural language questions? (currently requires manual input)
2. **Causal reasoning integration** — extend the engine with Pearl's do-calculus
3. **Argumentation frameworks** — integrate Dung's argumentation for explanation
4. **Constrained decoding** — use logic to mask invalid tokens during LLM generation

---

## References

- Lorini, E. (2024). Designing Artificial Reasoners for Communication. *AAMAS 2024*.
- Nematzadeh, A. et al. (2018). Evaluating Theory of Mind in Question Answering. *EMNLP 2019*.
- Fagin, R. et al. (1995). *Reasoning About Knowledge*. MIT Press.
- Hintikka, J. (1962). *Knowledge and Belief*. Cornell University Press.
- Ullman, T. (2023). Large Language Models Fail on Trivial Alterations to Theory-of-Mind Tasks. *arXiv:2302.08399*.
- Dung, P. M. (1995). On the acceptability of arguments. *Artificial Intelligence*, 77(2).
- Pearl, J. (2000). *Causality*. Cambridge University Press.

---

## License

MIT License — free to use, modify, and distribute.