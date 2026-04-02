import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from nlp.parser import parse_story
from llm.bridge import evaluate_story
from core.reasoner import Reasoner, Formula
from core.formula_parser import parse_formula, formula_examples

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EpistemicLLM",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EpistemicLLM")
st.markdown("**Theory of Mind reasoning via Epistemic Logic + LLMs**")

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("About")
st.sidebar.markdown("""
This system combines **formal epistemic logic** with LLMs to reason about
what agents know and believe.

**3 systems compared:**
- 🔴 Baseline — zero-shot LLM
- 🟡 CoT — chain-of-thought LLM
- 🟢 EpistemicLLM — Kripke model + LLM
""")

st.sidebar.header("Example stories")
examples = {
    "Sally-Anne (classic)": """Sally and Anne are in the room. Sally puts the ball in the basket.
Sally leaves the room. Anne moves the ball from the basket to the box.
Sally comes back into the room.
Where does Sally think the ball is?""",

    "Communication": """John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Peter enters the kitchen and sees the milk in the cupboard.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?""",

    "Lying agent": """John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Mary then moves the milk from the cupboard to the drawer.
Peter enters the kitchen and sees the milk in the drawer.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?""",

    "Three agents": """Emma and Liam are in the living room. Emma puts the book on the shelf.
Liam leaves the living room. Emma moves the book from the shelf to the drawer.
Noah enters the living room and sees the book in the drawer.
Emma leaves the living room.
Where does Liam think the book is?""",
}

selected = st.sidebar.selectbox("Load an example", ["— write your own —"] + list(examples.keys()))

# ── main input ────────────────────────────────────────────────────────────────

st.header("Story input")

if selected != "— write your own —":
    default_story = examples[selected]
else:
    default_story = ""

story = st.text_area(
    "Enter a Theory of Mind story (end with a question):",
    value=default_story,
    height=180,
    placeholder="Sally and Anne are in the room...\nWhere does Sally think the ball is?"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_epistemic = st.button("🟢 Run EpistemicLLM", type="primary")
with col2:
    run_all = st.button("⚡ Run 3-way comparison")

# ── epistemic only ────────────────────────────────────────────────────────────

if run_epistemic and story.strip():
    with st.spinner("Parsing story and building Kripke model..."):
        try:
            tracker, structure, query = parse_story(story)

            st.header("Results")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Extracted structure")
                st.json(structure)

            with col2:
                st.subheader("Belief state — Kripke model")
                st.code(tracker.snapshot(), language=None)

            st.subheader("Formal query")
            agent = query["agent"]
            fact  = query["fact"]
            obj, location = fact.split("_at_", 1)

            st.markdown(f"**Does `{agent}` know that the `{obj}` is in the `{location}`?**")
            st.markdown(f"Formal notation: `K_{agent}({fact})`")

            expected = query["expected_answer"]
            r2 = Reasoner(tracker.model)
            actual = r2.evaluate(Formula.knows(agent, Formula.atom(fact)))
            r2.reset_trace()

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.metric("🟢 EpistemicLLM answer", "True" if actual else "False")
            with col_v2:
                if actual == expected:
                    st.success("✓ Correct")
                else:
                    st.error("✗ Wrong")
            st.subheader("Proof trace")
            r = Reasoner(tracker.model)
            r.evaluate(Formula.knows(agent, Formula.atom(fact)))
            st.code(r.proof_report(), language=None)

        except Exception as e:
            st.error(f"Error: {e}")

# ── 3-way comparison ──────────────────────────────────────────────────────────

elif run_all and story.strip():
    with st.spinner("Running all 3 systems... (this may take 15-30 seconds)"):
        try:
            result = evaluate_story(story)

            st.header("3-way comparison results")

            # summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "normal" if result.baseline_correct else "inverse"
                st.metric(
                    "🔴 Baseline",
                    "✓ Correct" if result.baseline_correct else "✗ Wrong",
                    delta=None
                )
            with col2:
                st.metric(
                    "🟡 Chain-of-Thought",
                    "✓ Correct" if result.cot_correct else "✗ Wrong",
                )
            with col3:
                st.metric(
                    "🟢 EpistemicLLM",
                    "✓ Correct" if result.epistemic_correct else "✗ Wrong",
                )

            st.divider()

            # details
            st.subheader("Query details")
            obj, location = result.fact.split("_at_", 1)
            st.markdown(f"**Agent queried:** `{result.agent}`")
            st.markdown(f"**Question:** Does `{result.agent}` know that the `{obj}` is in the `{location}`?")
            st.markdown(f"**Formal notation:** `K_{result.agent}({result.fact})`")
            st.markdown(f"**Expected answer:** `{result.expected}`")

            st.divider()

            # answers breakdown
            st.subheader("System answers")
            data = {
                "System": ["Baseline (zero-shot)", "Chain-of-Thought", "EpistemicLLM"],
                "Answer": [
                    str(result.baseline_answer),
                    str(result.cot_answer),
                    str(result.epistemic_answer),
                ],
                "Correct": [
                    "✓" if result.baseline_correct else "✗",
                    "✓" if result.cot_correct else "✗",
                    "✓" if result.epistemic_correct else "✗",
                ]
            }
            st.table(data)

            st.divider()

            # kripke model
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Kripke model")
                st.code(result.kripke_snapshot, language=None)
            with col2:
                st.subheader("Proof trace")
                st.code(result.proof_trace, language=None)

        except Exception as e:
            st.error(f"Error: {e}")

elif (run_epistemic or run_all) and not story.strip():
    st.warning("Please enter a story first.")


# ── advanced formula mode ─────────────────────────────────────────────────────

st.divider()
st.header("🔬 Advanced — nested formula evaluation")
st.markdown(
    "Build the Kripke model from your story, then evaluate any epistemic formula manually. "
    "This supports **second-order** and higher-order reasoning."
)

with st.expander("Open advanced formula evaluator"):

    if not story.strip():
        st.warning("Enter a story above first.")
    else:
        # build model silently
        with st.spinner("Building Kripke model..."):
            try:
                tracker_adv, structure_adv, _ = parse_story(story)
                agents_adv = [a.lower() for a in structure_adv.get("agents", [])]
                facts_adv  = [
                    f for f, v in tracker_adv.current_facts.items() if v
                ] + [
                    f for f, v in tracker_adv.current_facts.items() if not v
                ]
                # deduplicate and sort
                all_facts = sorted(set(tracker_adv.current_facts.keys()))
                built_ok  = True
            except Exception as e:
                st.error(f"Could not build model: {e}")
                built_ok = False

        if built_ok:
            # show current model
            st.subheader("Current Kripke model")
            st.code(tracker_adv.snapshot(), language=None)

            # show syntax help
            st.subheader("Formula syntax")
            st.markdown("""
| Syntax | Meaning |
|---|---|
| `K_agent(fact)` | agent knows fact |
| `NOT K_agent(fact)` | agent does NOT know fact |
| `K_a(NOT K_b(fact))` | a knows that b does not know fact |
| `AND(phi, psi)` | phi AND psi |
| `OR(phi, psi)` | phi OR psi |
""")

            # auto-generate examples
            examples_adv = formula_examples(agents_adv, all_facts)
            if examples_adv:
                st.markdown("**Generated examples for this story:**")
                for ex in examples_adv:
                    st.code(ex, language=None)

            # formula input
            formula_input = st.text_input(
                "Enter formula:",
                placeholder=f"K_{agents_adv[0]}(NOT K_{agents_adv[1]}({all_facts[0]}))" if len(agents_adv) >= 2 and all_facts else "K_agent(fact)"
            )

            if st.button("▶ Evaluate formula"):
                if formula_input.strip():
                    try:
                        formula_obj = parse_formula(formula_input.strip(), agents_adv)
                        r_adv       = Reasoner(tracker_adv.model)
                        result_adv  = r_adv.evaluate(formula_obj)

                        st.subheader("Result")
                        if result_adv:
                            st.success(f"✓ TRUE — `{formula_input}` holds in the actual world")
                        else:
                            st.error(f"✗ FALSE — `{formula_input}` does not hold")

                        st.subheader("Proof trace")
                        st.code(r_adv.proof_report(), language=None)

                    except Exception as e:
                        st.error(f"Formula parse error: {e}")
                else:
                    st.warning("Enter a formula first.")

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "*EpistemicLLM — combining formal epistemic logic with LLMs for Theory of Mind reasoning. "
)