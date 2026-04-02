import os
import json
from nlp.parser import parse_story, extract_story_structure
from core.reasoner import Reasoner, Formula

from dotenv import load_dotenv
load_dotenv()

assert os.environ.get("OPENAI_API_KEY"), \
    "Set OPENAI_API_KEY environment variable first"


SALLY_ANNE_STORY = """
Sally and Anne are in the room. Sally puts the ball in the basket.
Sally leaves the room. Anne moves the ball from the basket to the box.
Sally comes back into the room.
Where does Sally think the ball is?
"""

SECOND_ORDER_STORY = """
John and Mary are in the kitchen. John puts the milk in the fridge.
John goes to the garden. Mary moves the milk from the fridge to the cupboard.
Peter enters the kitchen and sees the milk in the cupboard.
Peter tells John that the milk is in the cupboard.
Where does John think the milk is?
"""


def run_pipeline(name: str, story: str):
    print(f"\n{'═'*50}")
    print(f"  Story : {name}")
    print(f"{'═'*50}")

    # Step 1 — extraction
    print("\n── Step 1 : Extracted structure ────────────")
    tracker, structure, question = parse_story(story)
    print(json.dumps(structure, indent=2))

    # Step 2 — question classification
    print("\n── Step 2 : Question classification ────────")
    print(json.dumps(question, indent=2))

    # Step 3 — belief state
    print("\n── Step 3 : Belief state (Kripke model) ────")
    print(tracker.snapshot())

    # Step 4 — formal reasoning
    print("\n── Step 4 : Formal reasoning ───────────────")
    r = Reasoner(tracker.model)
    agent = question["agent"].lower()
    fact  = question["fact"].lower()

    f = Formula.knows(agent, Formula.atom(fact))
    result = r.evaluate(f)
    print(f"Query   : Does {agent} know '{fact}' ?")
    print(f"Answer  : {result}")
    print(r.proof_report())

    # Step 5 — verdict
    expected = question.get("expected_answer")
    correct  = (result == expected)
    print(f"── Step 5 : Verdict ────────────────────────")
    print(f"Expected : {expected}")
    print(f"Got      : {result}")
    print(f"{'✓ CORRECT' if correct else '✗ WRONG'}")


if __name__ == "__main__":
    run_pipeline("Sally-Anne",    SALLY_ANNE_STORY)
    run_pipeline("Communication", SECOND_ORDER_STORY)
    print(f"\n{'═'*50}")
    print("  All parser tests complete.")
    print(f"{'═'*50}")